"""
Chatbot Arena (Jupyter) tab.
Users chat with two chosen models in Jupyter notebooks.
"""

import json
import time
import os
from pathlib import Path
import requests
import traceback

import gradio as gr
import numpy as np
from huggingface_hub import InferenceClient
from e2b_code_interpreter import Sandbox
from transformers import AutoTokenizer
from gradio_sandboxcomponent import SandboxComponent
import nbformat
from nbconvert import HTMLExporter

from fastchat.constants import (
    MODERATION_MSG,
    CONVERSATION_LIMIT_MSG,
    INPUT_CHAR_LEN_LIMIT,
    CONVERSATION_TURN_LIMIT,
    SURVEY_LINK,
)
from fastchat.serve.gradio_web_server import (
    State,
    bot_response,
    get_conv_log_filename,
    no_change_btn,
    enable_btn,
    disable_btn,
    invisible_btn,
    acknowledgment_md,
    get_ip,
    get_model_description_md,
)
from fastchat.serve.remote_logger import get_remote_logger
from fastchat.serve.jupyter_utils import (
    create_base_notebook,
    update_notebook_display,
)
from fastchat.serve.sandbox.code_runner import (
    DEFAULT_DATA_SCIENCE_PROMPT,
    SandboxEnvironment,
    CodeSandbox,
)
from fastchat.utils import (
    build_logger,
    moderation_filter,
)
from fastchat.model.model_adapter import get_conversation_template
from fastchat.conversation import Conversation

logger = build_logger("gradio_web_server_multi", "gradio_web_server_multi.log")

# Environment variables and constants
E2B_API_KEY = os.environ.get("E2B_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")
TMP_DIR = './tmp/'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

# Global variables
num_sides = 2
enable_moderation = False
models = []  # Will be populated by load_demo_side_by_side_jupyter
SANDBOXES = {}  # Global sandbox sessions

html_exporter = HTMLExporter()
html_exporter.template_name = 'classic'

def parse_exec_result_nb(execution):
    """Convert an E2B Execution object to Jupyter notebook cell output format"""
    outputs = []
    
    # Handle stdout if present
    if execution.logs.stdout:
        outputs.append({
            'output_type': 'stream',
            'name': 'stdout',
            'text': ''.join(execution.logs.stdout)
        })
    
    # Handle stderr if present
    if execution.logs.stderr:
        outputs.append({
            'output_type': 'stream',
            'name': 'stderr',
            'text': ''.join(execution.logs.stderr)
        })

    # Handle error if present
    if execution.error:
        outputs.append({
            'output_type': 'error',
            'ename': execution.error.name,
            'evalue': execution.error.value,
            'traceback': [line for line in execution.error.traceback.split('\n')]
        })

    # Handle execution results
    for result in execution.results:
        output = {
            'output_type': 'execute_result' if result.is_main_result else 'display_data',
            'metadata': {},
            'data': {}
        }
        
        # Handle different result types
        if result.text:
            output['data']['text/plain'] = [result.text]  # Array for text/plain
        if result.html:
            output['data']['text/html'] = result.html
        if result.png:
            output['data']['image/png'] = result.png
        if result.svg:
            output['data']['image/svg+xml'] = result.svg
        if result.jpeg:
            output['data']['image/jpeg'] = result.jpeg
        if result.pdf:
            output['data']['application/pdf'] = result.pdf
        if result.latex:
            output['data']['text/latex'] = result.latex
        if result.json:
            output['data']['application/json'] = result.json
        if result.javascript:
            output['data']['application/javascript'] = result.javascript

        # Add execution count for main results
        if result.is_main_result and execution.execution_count is not None:
            output['execution_count'] = execution.execution_count

        # Only add output if it has data
        if output['data']:
            outputs.append(output)

    return outputs

def parse_exec_result_llm(execution):
    """Convert an E2B Execution object to markdown format for LLM consumption"""
    output = ""
    
    # Add stdout if present
    if execution.logs.stdout:
        output += "### Stdout:\n```\n" + "".join(execution.logs.stdout) + "\n```\n\n"
    
    # Add stderr if present
    if execution.logs.stderr:
        output += "### Stderr:\n```\n" + "".join(execution.logs.stderr) + "\n```\n\n"
    
    # Add error if present
    if execution.error:
        output += "### Error:\n```\n" + execution.error.traceback + "\n```\n\n"
    
    # Add results
    results = []
    for result in execution.results:
        if result.text:
            results.append(result.text)
        if result.html:
            results.append(result.html)
        if result.markdown:
            results.append(f"```markdown\n{result.markdown}\n```")
        if result.latex:
            results.append(f"```latex\n{result.latex}\n```")
        if result.json:
            results.append(f"```json\n{result.json}\n```")
        if result.javascript:
            results.append(result.javascript)
        if result.png:
            results.append(f"![png image](data:image/png;base64,{result.png})")
        if result.jpeg:
            results.append(f"![jpeg image](data:image/jpeg;base64,{result.jpeg})")
        if result.svg:
            results.append(f"![svg image](data:image/svg+xml;base64,{result.svg})")
    
    if results:
        output += "### Results:\n" + "\n".join(results)
    
    return output

def normalize_role(role: str) -> str:
    """Normalize different role formats to standard roles."""
    role = role.lower()
    if "user" in role or "human" in role:
        return "user"
    elif "assistant" in role or "bot" in role or "model" in role:
        return "assistant"
    elif "system" in role:
        return "system"
    else:
        return role

def extract_python_code_from_markdown(message: str) -> str | None:
    """Extract Python code from markdown code blocks.
    Returns the code if found, None otherwise."""
    # Match ```python ... ``` blocks
    import re
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, message, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def execute_jupyter_agent_multi(
    state0, state1,
    temperature,
    top_p,
    max_new_tokens,
    sandbox_state0,
    sandbox_state1,
    request: gr.Request
):
    """Execute Jupyter notebooks for both models in parallel"""
    try:
        ip = get_ip(request)
        logger.info(f"execute_jupyter_agent_multi. ip: {ip}")
        
        states = [state0, state1]
        sandbox_states = [sandbox_state0, sandbox_state1]

        if state0 is not None and state1 is not None: 
            if state0.skip_next or state1.skip_next:
                notebook_data = create_base_notebook([])
                empty_html = update_notebook_display(notebook_data)
                return states + [empty_html] * num_sides + [""] + [no_change_btn] * 4

        # Create generators for both models
        gen = []
        for i in range(num_sides):
            gen.append(
                bot_response(
                    states[i],
                    temperature,
                    top_p,
                    max_new_tokens,
                    sandbox_states[i],
                    request
                )
            )

        # Execute for both models with streaming
        notebooks = [None] * num_sides
        notebook_datas = [None] * num_sides
        collecting_code = [False] * num_sides
        
        # Initialize notebooks with current conversation (excluding last message)
        for i in range(num_sides):
            messages = []
            for role, content in states[i].conv.messages[:-1]:  # Exclude last message
                if content is not None:
                    messages.append({"role": normalize_role(role), "content": content})
            notebook_data = create_base_notebook(messages)
            # Add initial markdown cell for streaming
            notebook_data["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": "",
                "streaming": True
            })
            notebook_datas[i] = notebook_data
            notebooks[i] = update_notebook_display(notebook_data)

        # Create sandbox with longer timeout
        sandbox = CodeSandbox(api_key=E2B_API_KEY, timeout=300)
        try:
            while True:
                stop = True
                for i in range(num_sides):
                    try:
                        ret = next(gen[i])
                        states[i] = ret[0]  # Get updated state
                        
                        # Handle only the last message (streaming)
                        last_message = states[i].conv.messages[-1]
                        if last_message[1] is not None:
                            content = last_message[1]
                            
                            # Check for code block start/end
                            if "```python" in content:
                                parts = content.split("```python", 1)
                                pre_code = parts[0]
                                code_part = parts[1]
                                
                                # Update streaming markdown cell with pre-code content
                                if notebook_datas[i]["cells"][-1]["cell_type"] == "markdown":
                                    notebook_datas[i]["cells"][-1]["source"] = pre_code
                                
                                # If we find a complete code block
                                if "```" in code_part:
                                    code, post_code = code_part.split("```", 1)
                                    code = code.strip()
                                    
                                    # Add and execute code cell
                                    code_cell = {
                                        "cell_type": "code",
                                        "execution_count": len([c for c in notebook_datas[i]["cells"] if c["cell_type"] == "code"]) + 1,
                                        "metadata": {},
                                        "source": code,
                                        "outputs": []
                                    }
                                    notebook_datas[i]["cells"].append(code_cell)
                                    
                                    # Execute code
                                    execution = sandbox.run_code(code, language='python')
                                    code_cell["outputs"] = parse_exec_result_nb(execution)
                                    
                                    # Add new markdown cell for post-code content
                                    notebook_datas[i]["cells"].append({
                                        "cell_type": "markdown",
                                        "metadata": {},
                                        "source": post_code,
                                        "streaming": True
                                    })
                                else:
                                    # Code block is incomplete, show it in a code cell
                                    if notebook_datas[i]["cells"][-1]["cell_type"] != "code":
                                        notebook_datas[i]["cells"].append({
                                            "cell_type": "code",
                                            "execution_count": len([c for c in notebook_datas[i]["cells"] if c["cell_type"] == "code"]) + 1,
                                            "metadata": {},
                                            "source": code_part,
                                            "outputs": []
                                        })
                                    else:
                                        notebook_datas[i]["cells"][-1]["source"] = code_part
                            else:
                                # Regular message content - update streaming markdown cell
                                if notebook_datas[i]["cells"][-1]["cell_type"] == "markdown":
                                    notebook_datas[i]["cells"][-1]["source"] = content
                                else:
                                    notebook_datas[i]["cells"].append({
                                        "cell_type": "markdown",
                                        "metadata": {},
                                        "source": content,
                                        "streaming": True
                                    })
                        
                        # Update notebook display
                        notebooks[i] = update_notebook_display(notebook_datas[i])
                        stop = False
                    except StopIteration:
                        # Remove streaming flag from last cell when done
                        if notebook_datas[i]["cells"]:
                            notebook_datas[i]["cells"][-1].pop("streaming", None)
                        pass
                    except Exception as e:
                        error_msg = f"Error in execute_jupyter_agent_multi for model {i}: {str(e)}"
                        logger.error(error_msg)
                        # Create error notebook with assistant role
                        notebook_data = create_base_notebook([{"role": "assistant", "content": error_msg}])
                        notebooks[i] = update_notebook_display(notebook_data)
                
                yield states + notebooks + [""] + [enable_btn] * 4
                if stop:
                    break

        finally:
            # Clean up sandbox
            sandbox.close()

    except Exception as e:
        error_msg = f"Error in execute_jupyter_agent_multi: {str(e)}"
        logger.error(error_msg)
        # Create error notebooks for both sides
        notebook_data = create_base_notebook([{"role": "assistant", "content": error_msg}])
        error_html = update_notebook_display(notebook_data)
        return states + [error_html] * num_sides + [""] + [enable_btn] * 4

def set_global_vars_jupyter(enable_moderation_):
    global enable_moderation
    enable_moderation = enable_moderation_

def load_demo_side_by_side_jupyter(models_, url_params):
    global models
    models = models_  # Store models in global variable
    
    states = [None] * num_sides

    model_left = models[0] if len(models) > 0 else ""
    if len(models) > 1:
        weights = ([8] * 4 + [4] * 8 + [1] * 64)[: len(models) - 1]
        weights = weights / np.sum(weights)
        model_right = np.random.choice(models[1:], p=weights)
    else:
        model_right = model_left

    selector_updates = [
        gr.Dropdown(choices=models, value=model_left, visible=True),
        gr.Dropdown(choices=models, value=model_right, visible=True),
    ]

    return states + selector_updates

def vote_last_response(states, vote_type, model_selectors, request: gr.Request):
    if states[0] is None or states[1] is None:
        return
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "models": [x for x in model_selectors],
            "states": [x.dict() for x in states],
            "ip": get_ip(request),
        }
        fout.write(json.dumps(data) + "\n")
    get_remote_logger().log(data)

def leftvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"leftvote (jupyter). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "leftvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4

def rightvote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"rightvote (jupyter). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "rightvote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4

def tievote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"tievote (jupyter). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "tievote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4

def bothbad_vote_last_response(
    state0, state1, model_selector0, model_selector1, request: gr.Request
):
    logger.info(f"bothbad_vote (jupyter). ip: {get_ip(request)}")
    vote_last_response(
        [state0, state1], "bothbad_vote", [model_selector0, model_selector1], request
    )
    return ("",) + (disable_btn,) * 4

def clear_history(request: gr.Request):
    logger.info(f"clear_history (jupyter). ip: {get_ip(request)}")
    return [None] * num_sides + [None] * num_sides + [""] + [invisible_btn] * 4 + [disable_btn] * 1 + [invisible_btn] * 2 + [disable_btn] * 1

def share_click(state0, state1, model_selector0, model_selector1, request: gr.Request):
    logger.info(f"share (jupyter). ip: {get_ip(request)}")
    if state0 is not None and state1 is not None:
        vote_last_response(
            [state0, state1], "share", [model_selector0, model_selector1], request
        )

def add_text_multi(
    state0, state1,
    model_selector0, model_selector1,
    sandbox_state0, sandbox_state1,
    text, request: gr.Request
):
    ip = get_ip(request)
    logger.info(f"add_text_multi (jupyter). ip: {ip}. len: {len(text)}")
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]
    sandbox_states = [sandbox_state0, sandbox_state1]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = State(model_selectors[i])

    if len(text) <= 0:
        for i in range(num_sides):
            states[i].skip_next = True
        return (
            states
            + [None] * num_sides  # Return None to keep current notebook display
            + sandbox_states
            + [""]
            + [no_change_btn] * 4
        )

    model_list = [states[i].model_name for i in range(num_sides)]
    all_conv_text_left = states[0].conv.get_prompt()
    all_conv_text_right = states[1].conv.get_prompt()
    all_conv_text = (
        all_conv_text_left[-1000:] + all_conv_text_right[-1000:] + "\nuser: " + text
    )
    flagged = moderation_filter(all_conv_text, model_list)
    if flagged:
        logger.info(f"violate moderation (jupyter). ip: {ip}. text: {text}")
        text = MODERATION_MSG

    conv = states[0].conv
    if (len(conv.messages) - conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        for i in range(num_sides):
            states[i].skip_next = True
        notebook = create_base_notebook([{"role": "assistant", "content": CONVERSATION_LIMIT_MSG}])
        return (
            states
            + [update_notebook_display(notebook)] * num_sides
            + sandbox_states
            + [CONVERSATION_LIMIT_MSG]
            + [no_change_btn] * 4
        )

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    for i in range(num_sides):
        states[i].conv.append_message(states[i].conv.roles[0], text)
        states[i].conv.append_message(states[i].conv.roles[1], None)
        states[i].skip_next = False

    # Convert current conversation to notebook format
    notebooks = []
    for i in range(num_sides):
        messages = []
        for role, content in states[i].conv.messages:
            if content is not None:
                messages.append({"role": normalize_role(role), "content": content})
        notebook_data = create_base_notebook(messages)
        notebooks.append(update_notebook_display(notebook_data))

    return (
        states
        + notebooks
        + sandbox_states
        + [""]
        + [disable_btn] * 4
    )

def update_sandbox_system_messages_multi(state0, state1, sandbox_state0, sandbox_state1, model_selector0, model_selector1):
    '''Add sandbox instructions to the system message.'''
    states = [state0, state1]
    model_selectors = [model_selector0, model_selector1]
    sandbox_states = [sandbox_state0, sandbox_state1]

    # Init states if necessary
    for i in range(num_sides):
        if states[i] is None:
            states[i] = State(model_selectors[i])
        sandbox_state = sandbox_states[i]
        if sandbox_state is None or sandbox_state['enable_sandbox'] is False or sandbox_state["enabled_round"] > 0:
            continue
        sandbox_state['enabled_round'] += 1 # avoid dup
        environment_instruction = sandbox_state['sandbox_instruction']
        current_system_message = states[i].conv.get_system_message(states[i].is_vision)
        new_system_message = f"{current_system_message}\n\n{environment_instruction}"
        states[i].conv.set_system_message(new_system_message)

    # Convert current conversation to notebook format
    notebooks = []
    for i in range(num_sides):
        messages = []
        for role, content in states[i].conv.messages:
            if content is not None:
                messages.append({"role": normalize_role(role), "content": content})
        notebook_data = create_base_notebook(messages)
        notebooks.append(update_notebook_display(notebook_data))

    return states + notebooks

def build_side_by_side_ui_jupyter(models):
    """Build side-by-side Jupyter notebook interface."""
    states = [gr.State() for _ in range(num_sides)]
    model_selectors = [None] * num_sides
    notebooks = [None] * num_sides
    
    # Initialize sandbox states
    sandbox_states = [
        gr.State({
            "enable_sandbox": False,
            "sandbox_environment": SandboxEnvironment.JUPYTER,
            "auto_selected_sandbox_environment": None,
            "sandbox_instruction": DEFAULT_DATA_SCIENCE_PROMPT,
            "code_to_execute": "",
            "code_language": None,
            "code_dependencies": ([], []),
            "enabled_round": 0,
            "btn_list_length": 4
        }) for _ in range(num_sides)
    ]

    notice_markdown = (
        "# ⚔️  Chatbot Arena - Jupyter Notebook Edition\n\n"
        "![Jupyter Logo](assets/Jupyter_logo.png)\n\n"
        "Compare and test AI models in interactive Jupyter notebooks side by side.\n\n"
        f"{SURVEY_LINK}\n\n"
        "## 📜 How It Works\n\n"
        "- Ask any question to two chosen models and see their responses in Jupyter notebooks\n"
        "- Compare their code execution, visualizations and outputs\n"
        "- Vote for the better one!\n\n"
        "## 👇 Choose two models to compare\n"
    )
    notice = gr.Markdown(notice_markdown, elem_id="notice_markdown")

    with gr.Group(elem_id="share-region-jupyter"):
        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    model_selectors[i] = gr.Dropdown(
                        choices=models,
                        value=models[i] if len(models) > i else models[0],
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
        with gr.Row():
            with gr.Accordion(
                f"🔍 Expand to see the descriptions of {len(models)} models", open=False
            ):
                model_description_md = get_model_description_md(models)
                gr.Markdown(model_description_md, elem_id="model_description_markdown")

        with gr.Row():
            for i in range(num_sides):
                with gr.Column():
                    notebooks[i] = gr.HTML(
                        value=update_notebook_display(create_base_notebook([])),
                        visible=True,
                        elem_id=f"notebook_{i}",
                        container=True,
                        elem_classes=["notebook-wrapper"]
                    )

        with gr.Row():
            textbox = gr.Textbox(
                show_label=False,
                placeholder="👉 Enter your prompt and press ENTER",
                elem_id="input_box",
            )
            send_btn = gr.Button(value="Send", variant="primary", scale=0)
            send_btn_left = gr.Button(value="Send to Left", variant="primary", scale=0)
            send_btn_right = gr.Button(value="Send to Right", variant="primary", scale=0)

        with gr.Row():
            with gr.Column(scale=1, min_width=50):
                leftvote_btn = gr.Button(value="👈  Left is better", visible=False, interactive=False)
            with gr.Column(scale=1, min_width=50):
                rightvote_btn = gr.Button(value="👉  Right is better", visible=False, interactive=False)
            with gr.Column(scale=1, min_width=50):
                tie_btn = gr.Button(value="🤝  Tie", visible=False, interactive=False)
            with gr.Column(scale=1, min_width=50):
                bothbad_btn = gr.Button(value="👎  Both are bad", visible=False, interactive=False)

        with gr.Row():
            clear_btn = gr.Button(value="🗑️  Clear history", interactive=True)
            regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=False)
            share_btn = gr.Button(value="📷  Share")

        with gr.Row():
            with gr.Accordion("Parameters", open=False, visible=False) as parameter_row:
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.1,
                    interactive=True,
                    label="Top P",
                )
                max_new_tokens = gr.Slider(
                    minimum=16,
                    maximum=2048,
                    value=512,
                    step=64,
                    interactive=True,
                    label="Max output tokens",
                )

        textbox.submit(
            add_text_multi,
            states + model_selectors + sandbox_states + [textbox],
            states + notebooks + sandbox_states + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        ).then(
            execute_jupyter_agent_multi,
            states + [temperature, top_p, max_new_tokens] + sandbox_states,
            states + notebooks + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        )

        send_btn.click(
            add_text_multi,
            states + model_selectors + sandbox_states + [textbox],
            states + notebooks + sandbox_states + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        ).then(
            execute_jupyter_agent_multi,
            states + [temperature, top_p, max_new_tokens] + sandbox_states,
            states + notebooks + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        )

        send_btn_left.click(
            add_text,
            [states[0], model_selectors[0], sandbox_states[0], textbox],
            [states[0], notebooks[0], sandbox_states[0], textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        ).then(
            execute_jupyter_agent_multi,
            [states[0], states[1], temperature, top_p, max_new_tokens, sandbox_states[0], sandbox_states[1]],
            states + notebooks + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        )

        send_btn_right.click(
            add_text,
            [states[1], model_selectors[1], sandbox_states[1], textbox],
            [states[1], notebooks[1], sandbox_states[1], textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        ).then(
            execute_jupyter_agent_multi,
            [states[0], states[1], temperature, top_p, max_new_tokens, sandbox_states[0], sandbox_states[1]],
            states + notebooks + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        )

        regenerate_btn.click(
            execute_jupyter_agent_multi,
            states + [temperature, top_p, max_new_tokens] + sandbox_states,
            states + notebooks + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        )

        clear_btn.click(
            clear_history,
            None,
            states + notebooks + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn] + sandbox_states,
        )

        for i in range(num_sides):
            model_selectors[i].change(
                clear_history,
                None,
                states + notebooks + [textbox] + [leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
            )

        leftvote_btn.click(
            leftvote_last_response,
            [states[0], states[1], model_selectors[0], model_selectors[1]],
            [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        )

        rightvote_btn.click(
            rightvote_last_response,
            [states[0], states[1], model_selectors[0], model_selectors[1]],
            [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        )

        tie_btn.click(
            tievote_last_response,
            [states[0], states[1], model_selectors[0], model_selectors[1]],
            [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        )

        bothbad_btn.click(
            bothbad_vote_last_response,
            [states[0], states[1], model_selectors[0], model_selectors[1]],
            [textbox, leftvote_btn, rightvote_btn, tie_btn, bothbad_btn],
        )

        share_btn.click(
            share_click,
            [states[0], states[1], model_selectors[0], model_selectors[1]],
            [textbox],
        )

    # Return all components in the order expected by the demo
    return states + model_selectors

def regenerate(state, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"regenerate. ip: {ip}")
    if state is None:
        return (None, None, "") + (no_change_btn,) * 8
    if not state.regen_support:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * 8
    state.conv.update_last_message(None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 8

def add_text(state, model_selector, sandbox_state, text, request: gr.Request):
    ip = get_ip(request)
    logger.info(f"add_text. ip: {ip}. len: {len(text)}")

    if state is None:
        state = State(model_selector)

    if len(text) <= 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "", None) + (no_change_btn,) * sandbox_state["btn_list_length"]

    all_conv_text = state.conv.get_prompt()
    all_conv_text = all_conv_text[-2000:] + "\nuser: " + text
    flagged = moderation_filter(all_conv_text, [state.model_name])
    if flagged:
        logger.info(f"violate moderation. ip: {ip}. text: {text}")
        text = MODERATION_MSG

    if (len(state.conv.messages) - state.conv.offset) // 2 >= CONVERSATION_TURN_LIMIT:
        logger.info(f"conversation turn limit. ip: {ip}. text: {text}")
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), CONVERSATION_LIMIT_MSG, None) + (
            no_change_btn,
        ) * sandbox_state["btn_list_length"]

    text = text[:INPUT_CHAR_LEN_LIMIT]  # Hard cut-off
    state.conv.append_message(state.conv.roles[0], text)
    state.conv.append_message(state.conv.roles[1], None)
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * sandbox_state["btn_list_length"] 