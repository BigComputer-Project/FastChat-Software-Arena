'''
Run generated code in a sandbox environment.
'''

from enum import StrEnum
from typing import Any, Generator, TypedDict
import gradio as gr
import re
import os
import base64
from e2b import Sandbox
from e2b_code_interpreter import Sandbox as CodeSandbox
from gradio_sandboxcomponent import SandboxComponent

E2B_API_KEY = os.environ.get("E2B_API_KEY")
'''
API key for the e2b API.
'''

class SandboxEnvironment(StrEnum):
    AUTO = 'auto'
    # Code Interpreter
    PYTHON_CODE_INTERPRETER = 'python_code_interpreter'
    JAVASCRIPT_CODE_INTERPRETER = 'javascript_code_interpreter'
    # Web UI Frameworks
    HTML = 'html'
    REACT = 'react'
    VUE = 'vue'
    GRADIO = 'gradio'
    STREAMLIT = 'streamlit'
    NICEGUI = 'nicegui'
    PYGAME = 'pygame'


SUPPORTED_SANDBOX_ENVIRONMENTS: list[str] = [
    env.value for env in SandboxEnvironment
]

WEB_UI_SANDBOX_ENVIRONMENTS = [
    SandboxEnvironment.REACT,
    SandboxEnvironment.VUE,
    SandboxEnvironment.GRADIO,
    SandboxEnvironment.STREAMLIT,
    SandboxEnvironment.NICEGUI,
    SandboxEnvironment.PYGAME,
]

VALID_GRADIO_CODE_LANGUAGES = [
    'python', 'c', 'cpp', 'markdown', 'json', 'html', 'css', 'javascript', 'jinja2', 'typescript', 'yaml', 'dockerfile', 'shell', 'r', 'sql',
    'sql-msSQL', 'sql-mySQL', 'sql-mariaDB', 'sql-sqlite', 'sql-cassandra', 'sql-plSQL', 'sql-hive', 'sql-pgSQL', 'sql-gql', 'sql-gpSQL', 'sql-sparkSQL', 
    'sql-esper'
]

RUN_CODE_BUTTON_HTML = "<button style='background-color: #4CAF50; border: none; color: white; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;'>Click to Run in Sandbox</button>"


GENERAL_SANDBOX_INSTRUCTION = """You are a software engineer expert.
You have the capability to generate code for execution in sandbox environments. When tasked with creating such code, ensure it is safe and effective for the provided sandbox environments. All code must be self-contained within a single file, as the sandbox can only execute single-file applications.

## Code Format
When generating code intended for sandbox execution, strictly adhere to the following format:
- Start with the trigger `>>> sandbox("{environment_name}")` followed by a code block in the specified language.
- Do not place the sandbox trigger inside ANY code block.
- Follow the trigger with a code block in the specified language as shown below:

>>> sandbox("{environment_name}")
```[language]
[your code here]
```

## Code Usage Guidelines
- Always trigger the sandbox environment by default for any executable code generation, unless responding to theoretical questions or conceptual explanations.
- Begin responses with relevant context, explanations, or thought processes before introducing code execution. The sandbox trigger should be placed naturally within your response.
- Use only one sandbox environment per response for clarity and focus.
- When updating existing code, provide a complete new sandbox with the full updated code. Avoid partial updates or omissions to ensure functionality remains intact.

## Environment Specifications
When using this sandbox environment, follow these specifications:
"""

DEFAULT_PYTHON_CODE_INTERPRETER_INSTRUCTION = """
Generate self-contained Python code for execution with the following capabilities and requirements:
- Use only the standard library or these pre-installed packages: aiohttp, beautifulsoup4, bokeh, gensim, imageio, joblib, librosa, matplotlib, nltk, numpy, opencv-python, openpyxl, pandas, plotly, pytest, python-docx, pytz, requests, scikit-image, scikit-learn, scipy, seaborn, soundfile, spacy, textblob, tornado, urllib3, xarray, xlrd, sympy
- Code should produce output through stdout/stderr for text data and leverage matplotlib/plotly for data visualization
"""

DEFAULT_JAVASCRIPT_CODE_INTERPRETER_INSTRUCTION = """
Create JavaScript code that meets these requirements:
- Generate self-content code suitable for a Node.js-style environment that doesn't rely on browser-specific features
- You can output in stdout, stderr, or render images, plots, and tables.
"""

DEFAULT_HTML_SANDBOX_INSTRUCTION = """
Generate HTML code for a single HTML file to be executed in a sandbox:
- The HTML file should be self-contained and not rely on external libraries or resources
- You can use inline CSS and JavaScript
"""

DEFAULT_REACT_SANDBOX_INSTRUCTION = """
Generate typescript for a single-file Next.js 13+ React component tsx file:
- Available libraries: nextjs@14.2.5, typescript, @types/node, @types/react, @types/react-dom, postcss, tailwindcss, shadcn
- Do not use external libs or import external files
"""

DEFAULT_VUE_SANDBOX_INSTRUCTION = """
Generate TypeScript for a single-file Vue.js 3+ component (SFC) in .vue format:
- Component must use TypeScript and be contained within a styled `<div>` element
- Avoid external component dependencies including <NuxtWelcome />
"""

DEFAULT_PYGAME_SANDBOX_INSTRUCTION = """
Create PyGame applications that follow these structural requirements:
- Main game loop must be implemented as an async function, e.g.:
```python
import asyncio
import pygame

async def main():
    global game_state
    while game_state:
        game_state(pygame.event.get())
        pygame.display.update()
        await asyncio.sleep(0) # it must be called on every frame

if __name__ == "__main__":
    asyncio.run(main())
```
"""

DEFAULT_GRADIO_SANDBOX_INSTRUCTION = """
Generate Python code for a single-file Gradio application using the Gradio library:
- Available libraries: gradio, pandas, numpy, matplotlib, requests, seaborn, plotly
- Do not use external libraries or import external files outside of the allowed list
"""

DEFAULT_NICEGUI_SANDBOX_INSTRUCTION = """
Create modern Python web interfaces using NiceGUI with these requirements:
- Generate single-file NiceGUI applications with complete functionality
"""

DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION = """
Generate Python code for a single-file Streamlit application using the Streamlit library:
- Available libraries: streamlit, pandas, numpy, matplotlib, requests, seaborn, plotly
- Do not use external libraries or import external files outside of the allowed list
- The app should automatically reload when changes are made. 
"""

AUTO_SANDBOX_INSTRUCTION = f"""You are a software engineer expert.
You have the capability to generate code for execution in sandbox environments. When tasked with creating such code, ensure it is safe and effective for the provided sandbox environments. All code must be self-contained within a single file, as the sandbox can only execute single-file applications.

## Code Format
When generating code intended for sandbox execution, strictly adhere to the following format:
- Start with the trigger `>>> sandbox("[environment_name]")` followed by a code block in the specified language.
- Do not place the sandbox trigger inside ANY code block.
- Follow the trigger with a code block in the specified language as shown below:

>>> sandbox("[environment_name]")
```[language]
[your code here]
```

## Code Usage Guidelines
- Always trigger the sandbox environment by default for any executable code generation, unless responding to theoretical questions or conceptual explanations.
- Begin responses with relevant context, explanations, or thought processes before introducing code execution. The sandbox trigger should be placed naturally within your response.
- Use only one sandbox environment per response for clarity and focus.
- When updating existing code, provide a complete new sandbox with the full updated code. Avoid partial updates or omissions to ensure functionality remains intact.

## Supported Environments

### Python Code Interpreter (`>>> sandbox("{SandboxEnvironment.PYTHON_CODE_INTERPRETER}")`)
{DEFAULT_PYTHON_CODE_INTERPRETER_INSTRUCTION.strip()}

### JavaScript Code Interpreter (`>>> sandbox("{SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER}")`)
{DEFAULT_JAVASCRIPT_CODE_INTERPRETER_INSTRUCTION.strip()}

### HTML (`>>> sandbox("{SandboxEnvironment.HTML}")`)
{DEFAULT_HTML_SANDBOX_INSTRUCTION.strip()}

### React (`>>> sandbox("{SandboxEnvironment.REACT}")`)
{DEFAULT_REACT_SANDBOX_INSTRUCTION.strip()}

### Vue (`>>> sandbox("{SandboxEnvironment.VUE}")`)
{DEFAULT_VUE_SANDBOX_INSTRUCTION.strip()}

### Gradio (`>>> sandbox("{SandboxEnvironment.GRADIO}")`)
{DEFAULT_GRADIO_SANDBOX_INSTRUCTION.strip()}

### Streamlit (`>>> sandbox("{SandboxEnvironment.STREAMLIT}")`)
{DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION.strip()}

### NiceGUI (`>>> sandbox("{SandboxEnvironment.NICEGUI}")`)
{DEFAULT_NICEGUI_SANDBOX_INSTRUCTION.strip()}

### PyGame (`>>> sandbox("{SandboxEnvironment.PYGAME}")`)
{DEFAULT_PYGAME_SANDBOX_INSTRUCTION.strip()}
""".strip()

DEFAULT_SANDBOX_INSTRUCTIONS: dict[SandboxEnvironment, str] = {
    SandboxEnvironment.AUTO: AUTO_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.PYTHON_CODE_INTERPRETER: GENERAL_SANDBOX_INSTRUCTION.format(environment_name=SandboxEnvironment.PYTHON_CODE_INTERPRETER) + DEFAULT_PYTHON_CODE_INTERPRETER_INSTRUCTION.strip(),
    SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER: GENERAL_SANDBOX_INSTRUCTION.format(environment_name=SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER) + DEFAULT_JAVASCRIPT_CODE_INTERPRETER_INSTRUCTION.strip(),
    SandboxEnvironment.HTML: GENERAL_SANDBOX_INSTRUCTION.format(environment_name=SandboxEnvironment.HTML) + DEFAULT_HTML_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.REACT: GENERAL_SANDBOX_INSTRUCTION.format(environment_name=SandboxEnvironment.REACT) + DEFAULT_REACT_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.VUE: GENERAL_SANDBOX_INSTRUCTION.format(environment_name=SandboxEnvironment.VUE) + DEFAULT_VUE_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.GRADIO: GENERAL_SANDBOX_INSTRUCTION.format(environment_name=SandboxEnvironment.GRADIO) + DEFAULT_GRADIO_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.STREAMLIT: GENERAL_SANDBOX_INSTRUCTION.format(environment_name=SandboxEnvironment.STREAMLIT) + DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.NICEGUI: GENERAL_SANDBOX_INSTRUCTION.format(environment_name=SandboxEnvironment.NICEGUI) + DEFAULT_NICEGUI_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.PYGAME: GENERAL_SANDBOX_INSTRUCTION.format(environment_name=SandboxEnvironment.PYGAME) + DEFAULT_PYGAME_SANDBOX_INSTRUCTION.strip(),
}

print(AUTO_SANDBOX_INSTRUCTION)
print("-" * 80)
print(DEFAULT_SANDBOX_INSTRUCTIONS[SandboxEnvironment.PYTHON_CODE_INTERPRETER])

type SandboxGradioSandboxComponents = tuple[
    gr.Markdown | Any,  # sandbox_output
    SandboxComponent | Any,  # sandbox_ui
    gr.Code | Any,  # sandbox_code
]
'''
Gradio components for the sandbox.
'''

class ChatbotSandboxState(TypedDict):
    '''
    Chatbot sandbox state in gr.state.
    '''
    enable_sandbox: bool
    '''
    Whether the code sandbox is enabled.
    '''
    sandbox_instruction: str | None
    '''
    The sandbox instruction to display.
    '''
    enabled_round: int
    '''
    The chat round after which the sandbox is enabled.
    '''
    sandbox_environment: SandboxEnvironment | None
    '''
    The sandbox environment to run the code.
    '''
    auto_selected_sandbox_environment: SandboxEnvironment | None
    '''
    The sandbox environment selected automatically.
    '''
    code_to_execute: str | None
    '''
    The code to execute in the sandbox.
    '''
    code_language: str | None
    '''
    The code language to execute in the sandbox.
    '''


def create_chatbot_sandbox_state() -> ChatbotSandboxState:
    '''
    Create a new chatbot sandbox state.
    '''
    return {
        "enable_sandbox": False,
        "sandbox_environment": None,
        "auto_selected_sandbox_environment": None,
        "sandbox_instruction": None,
        "code_to_execute": "",
        "code_language": None,
        "enabled_round": 0
    }


def update_sandbox_config_multi(
    enable_sandbox: bool,
    sandbox_environment: SandboxEnvironment,
    *states: ChatbotSandboxState
) -> list[ChatbotSandboxState]:
    '''
    Fn to update sandbox config.
    '''
    return [
        update_sandbox_config(enable_sandbox, sandbox_environment, state) 
        for state
        in states
    ]

def update_sandbox_config(
    enable_sandbox: bool,
    sandbox_environment: SandboxEnvironment,
    state: ChatbotSandboxState
) -> ChatbotSandboxState:
    '''
    Fn to update sandbox config for single model.
    '''
    state["enable_sandbox"] = enable_sandbox
    state["sandbox_environment"] = sandbox_environment
    state['sandbox_instruction'] = DEFAULT_SANDBOX_INSTRUCTIONS.get(sandbox_environment, None)
    return state


def update_visibility(visible):
    return [gr.update(visible=visible)] * 13


def update_visibility_for_single_model(visible: bool, component_cnt: int):
    return [gr.update(visible=visible)] * component_cnt


def extract_code_from_markdown(message: str, enable_auto_env: bool=False) -> tuple[str, str, SandboxEnvironment | None] | None:
    '''
    Extracts code from a markdown message.

    Returns:
        tuple[str, str, bool]: A tuple:
            1. code,
            2. code language, 
            3. sandbox environment if auto environment is enabled, otherwise None
    '''
    # Regular expression to match code in sandboxCode tags
    code_block_regex = r'>>> sandbox\("(?P<sandbox_env_name>[^"]+)"\)\s*```(?P<code_lang>\w+)?\n(?P<code>.*?)```'

    match = re.search(code_block_regex, message, re.DOTALL)
    if not match:
        # if no matching code block is found, return None
        return None

    sandbox_env_name = match.group('sandbox_env_name') or None
    code_lang = match.group('code_lang') or ''
    code = match.group('code').strip()

    if enable_auto_env and sandbox_env_name is None:
        # auto must come with a sandbox environment name
        return None

    if sandbox_env_name is not None:
        try :
            sandbox_env_name = SandboxEnvironment(sandbox_env_name)
        except ValueError:
            # if the sandbox environment name is not valid, return None
            return None

    return code, code_lang, sandbox_env_name


def render_result(result):
    if result.png:
        if isinstance(result.png, str):
            img_str = result.png
        else:
            img_str = base64.b64encode(result.png).decode()
        return f"![png image](data:image/png;base64,{img_str})"
    elif result.jpeg:
        if isinstance(result.jpeg, str):
            img_str = result.jpeg
        else:
            img_str = base64.b64encode(result.jpeg).decode()
        return f"![jpeg image](data:image/jpeg;base64,{img_str})"
    elif result.svg:
        if isinstance(result.svg, str):
            svg_data = result.svg
        else:
            svg_data = result.svg.decode()
        svg_base64 = base64.b64encode(svg_data.encode()).decode()
        return f"![svg image](data:image/svg+xml;base64,{svg_base64})"
    elif result.html:
        return result.html
    elif result.markdown:
        return f"```markdown\n{result.markdown}\n```"
    elif result.latex:
        return f"```latex\n{result.latex}\n```"
    elif result.json:
        return f"```json\n{result.json}\n```"
    elif result.javascript:
        return result.javascript  # Return raw JavaScript
    else:
        return str(result)


def run_code_interpreter(code: str, code_language: str | None) -> str:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.
    """
    sandbox = CodeSandbox()

    execution = sandbox.run_code(
        code=code,
        language=code_language
    )

    # collect stdout, stderr from sandbox
    stdout = "\n".join(execution.logs.stdout)
    stderr = "\n".join(execution.logs.stderr)
    output = ""
    if stdout:
        output += f"### Stdout:\n```\n{stdout}\n```\n\n"
    if stderr:
        output += f"### Stderr:\n```\n{stderr}\n```\n\n"

    results = []
    js_code = ""
    for result in execution.results:
        rendered_result = render_result(result)
        if result.javascript:
            # TODO: js_code are not used
            # js_code += rendered_result + "\n"
            print("JavaScript code:", rendered_result)
        else:
            results.append(rendered_result)
    if results:
        output += "\n### Results:\n" + "\n".join(results)

    return output


def run_react_sandbox(code: str) -> str:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.

    Returns:
        url for remote sandbox
    """
    sandbox = Sandbox(
        template="nextjs-developer",
        metadata={
            "template": "nextjs-developer"
        },
        api_key=E2B_API_KEY,
    )

    # set up the sandbox
    sandbox.files.make_dir('pages')
    file_path = "~/pages/index.tsx"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    # get the sandbox url
    sandbox_url = 'https://' + sandbox.get_host(3000)
    return sandbox_url


def run_vue_sandbox(code: str) -> str:
    """
    Executes the provided Vue code within a sandboxed environment and returns the output.

    Args:
        code (str): The Vue code to be executed.

    Returns:
        url for remote sandbox
    """
    sandbox = Sandbox(
        template="vue-developer",
        metadata={
            "template": "vue-developer"
        },
        api_key=E2B_API_KEY,
    )

    # Set up the sandbox
    file_path = "~/app.vue"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    # Get the sandbox URL
    sandbox_url = 'https://' + sandbox.get_host(3000)
    return sandbox_url


def run_pygame_sandbox(code: str) -> str:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.

    Returns:
        url for remote sandbox
    """
    sandbox = Sandbox(
        api_key=E2B_API_KEY,
    )

    sandbox.files.make_dir('mygame')
    file_path = "~/mygame/main.py"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    sandbox.commands.run("pip install pygame pygbag black",
                         timeout=60 * 3,
                         # on_stdout=lambda message: print(message),
                         on_stderr=lambda message: print(message),)

    # build the pygame code
    sandbox.commands.run(
        "pygbag --build ~/mygame",  # build game
        timeout=60 * 5,
        # on_stdout=lambda message: print(message),
        # on_stderr=lambda message: print(message),
    )

    process = sandbox.commands.run(
        "python -m http.server 3000", background=True)  # start http server

    # get game server url
    host = sandbox.get_host(3000)
    url = f"https://{host}"
    return url + '/mygame/build/web/'


def run_nicegui_sandbox(code: str) -> str:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.

    Returns:
        url for remote sandbox
    """
    sandbox = Sandbox(
        api_key=E2B_API_KEY,
    )

    # set up sandbox
    setup_commands = [
        "pip install --upgrade nicegui",
    ]
    for command in setup_commands:
        sandbox.commands.run(
            command,
            timeout=60 * 3,
            on_stdout=lambda message: print(message),
            on_stderr=lambda message: print(message),
        )

    # write code to file
    sandbox.files.make_dir('mynicegui')
    file_path = "~/mynicegui/main.py"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    process = sandbox.commands.run(
        "python ~/mynicegui/main.py", background=True)

    # get web gui url
    host = sandbox.get_host(port=8080)
    url = f"https://{host}"
    return url


def run_gradio_sandbox(code: str) -> str:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.

    Returns:
        url for remote sandbox
    """
    sandbox = Sandbox(
        template="gradio-developer",
        metadata={
            "template": "gradio-developer"
        },
        api_key=E2B_API_KEY,
    )

    # set up the sandbox
    file_path = "~/app.py"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    # get the sandbox url
    sandbox_url = 'https://' + sandbox.get_host(7860)
    return sandbox_url


def run_streamlit_sandbox(code: str) -> str:
    sandbox = Sandbox(api_key=E2B_API_KEY)

    setup_commands = ["pip install --upgrade streamlit"]

    for command in setup_commands:
        sandbox.commands.run(
            command,
            timeout=60 * 3,
            on_stdout=lambda message: print(message),
            on_stderr=lambda message: print(message),
        )

    sandbox.files.make_dir('mystreamlit')
    file_path = "~/mystreamlit/app.py"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    process = sandbox.commands.run(
        "streamlit run ~/mystreamlit/app.py --server.port 8501 --server.headless true",
        background=True
    )

    host = sandbox.get_host(port=8501)
    url = f"https://{host}"
    return url

def on_edit_code(
    state,
    sandbox_state: ChatbotSandboxState,
    sandbox_output: gr.Markdown,
    sandbox_ui: SandboxComponent,
    sandbox_code: str,
) -> Generator[tuple[Any, Any, Any], None, None]:
    '''
    Gradio Handler when code is edited.
    '''
    if sandbox_state['enable_sandbox'] is False:
        yield None, None, None
        return
    if len(sandbox_code.strip()) == 0 or sandbox_code == sandbox_state['code_to_execute']:
        yield gr.skip(), gr.skip(), gr.skip()
        return
    sandbox_state['code_to_execute'] = sandbox_code
    yield from on_run_code(state, sandbox_state, sandbox_output, sandbox_ui, sandbox_code)

def on_click_code_message_run(
    state,
    sandbox_state: ChatbotSandboxState,
    sandbox_output: gr.Markdown,
    sandbox_ui: SandboxComponent,
    sandbox_code: str,
    evt: gr.SelectData
) -> Generator[SandboxGradioSandboxComponents, None, None]:
    '''
    Gradio Handler when run code button in message is clicked. Update Sandbox components.
    '''
    if sandbox_state['enable_sandbox'] is False:
        yield None, None, None
        return
    if not evt.value.endswith(RUN_CODE_BUTTON_HTML):
        yield gr.skip(), gr.skip(), gr.skip()
        return

    message = evt.value.replace(RUN_CODE_BUTTON_HTML, "").strip()
    extract_result = extract_code_from_markdown(
        message=message,
        enable_auto_env=sandbox_state['sandbox_environment'] == SandboxEnvironment.AUTO
    )
    if extract_result is None:
        yield gr.skip(), gr.skip(), gr.skip()
        return
    code, code_language, env_selection = extract_result

    # validate whether code to execute has been updated.
    previous_code = sandbox_state['code_to_execute']
    if previous_code == code:
        yield gr.skip(), gr.skip(), gr.skip()
        return

    if code_language == 'tsx':
        code_language = 'typescript'
    code_language = code_language.lower() if code_language and code_language.lower(
        # ensure gradio supports the code language
    ) in VALID_GRADIO_CODE_LANGUAGES else None

    sandbox_state['code_to_execute'] = code
    sandbox_state['code_language'] = code_language
    if sandbox_state['sandbox_environment'] == SandboxEnvironment.AUTO:
        sandbox_state['auto_selected_sandbox_environment'] = env_selection
    yield from on_run_code(state, sandbox_state, sandbox_output, sandbox_ui, sandbox_code)

def on_run_code(
    state,
    sandbox_state: ChatbotSandboxState,
    sandbox_output: gr.Markdown,
    sandbox_ui: SandboxComponent,
    sandbox_code: str
) -> Generator[tuple[Any, Any, Any], None, None]:
    '''
    gradio fn when run code button is clicked. Update Sandbox components.
    '''
    if sandbox_state['enable_sandbox'] is False:
        yield None, None, None
        return

    # validate e2b api key
    if not E2B_API_KEY:
        raise ValueError("E2B_API_KEY is not set in env vars.")

    code, code_language = sandbox_state['code_to_execute'], sandbox_state['code_language']
    if code is None or code_language is None:
        yield None, None, None
        return

    if code_language == 'tsx':
        code_language = 'typescript'
    code_language = code_language.lower() if code_language and code_language.lower(
        # ensure gradio supports the code language
    ) in VALID_GRADIO_CODE_LANGUAGES else None

    # show loading
    yield (
        gr.Markdown(value="### Loading Sandbox", visible=True),
        gr.skip(),
        gr.Code(value=code, language=code_language, visible=True),
    )

    sandbox_env = sandbox_state['sandbox_environment'] if sandbox_state['sandbox_environment'] != SandboxEnvironment.AUTO else sandbox_state['auto_selected_sandbox_environment']

    match sandbox_env:
        case SandboxEnvironment.REACT:
            url = run_react_sandbox(code)
            yield (
                gr.Markdown(value="### Running Sandbox", visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.VUE:
            url = run_vue_sandbox(code)
            yield (
                gr.Markdown(value="### Running Sandbox", visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.PYGAME:
            url = run_pygame_sandbox(code)
            yield (
                gr.Markdown(value="### Running Sandbox", visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.GRADIO:
            url = run_gradio_sandbox(code)
            yield (
                gr.Markdown(value="### Running Sandbox", visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.STREAMLIT:
            url = run_streamlit_sandbox(code)
            yield (
                gr.Markdown(value="### Running Sandbox", visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.NICEGUI:
            url = run_nicegui_sandbox(code)
            yield (
                gr.Markdown(value="### Running Sandbox", visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.PYTHON_CODE_INTERPRETER:
            output = run_code_interpreter(
                code=code, code_language='python'
            )
            yield (
                gr.Markdown(value=output, sanitize_html=False, visible=True),
                SandboxComponent(
                    value=('', ''),
                    label="Example",
                    visible=False,
                    key="newsandbox",
                ),  # hide the sandbox component
                gr.skip()
            )
        case SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER:
            output = run_code_interpreter(
                code=code, code_language='javascript'
            )
            yield (
                gr.Markdown(value=output, visible=True),
                SandboxComponent(
                    value=('', ''),
                    label="Example",
                    visible=False,
                    key="newsandbox",
                ),  # hide the sandbox component
                gr.skip()
            )
        case _:
            raise ValueError(
                f"Unsupported sandbox environment: {sandbox_state['sandbox_environment']}")
