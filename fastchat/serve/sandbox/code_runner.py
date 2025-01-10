'''
Run generated code in a sandbox environment.
'''

from enum import StrEnum
from typing import Any, Generator, TypeAlias, TypedDict, Set
import gradio as gr
import re
import os
import base64
from e2b import Sandbox
from e2b_code_interpreter import Sandbox as CodeSandbox
from e2b.sandbox.commands.command_handle import CommandExitException
from gradio_sandboxcomponent import SandboxComponent
import ast
import subprocess
import json
from tempfile import NamedTemporaryFile
from tree_sitter import Language, Node, Parser
import tree_sitter_javascript
import tree_sitter_typescript
from pathlib import Path
import sys

E2B_API_KEY = os.environ.get("E2B_API_KEY")
'''
API key for the e2b API.
'''

class SandboxEnvironment(StrEnum):
    AUTO = 'Auto'
    # Code Interpreter
    PYTHON_CODE_INTERPRETER = 'Python Code Interpreter'
    JAVASCRIPT_CODE_INTERPRETER = 'Javascript Code Interpreter'
    # Web UI Frameworks
    HTML = 'HTML'
    REACT = 'React'
    VUE = 'Vue'
    GRADIO = 'Gradio'
    STREAMLIT = 'Streamlit'
    NICEGUI = 'NiceGUI'
    PYGAME = 'PyGame'


SUPPORTED_SANDBOX_ENVIRONMENTS: list[str] = [
    env.value for env in SandboxEnvironment
]

WEB_UI_SANDBOX_ENVIRONMENTS = [
    SandboxEnvironment.HTML,
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
'''
Languages that gradio code component can render.
'''

RUN_CODE_BUTTON_HTML = "<button style='background-color: #4CAF50; border: none; color: white; padding: 10px 24px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 12px;'>Click to Run in Sandbox</button>"
'''
Button in the chat to run the code in the sandbox.
'''

GENERAL_SANDBOX_INSTRUCTION = """\
You are an expert Software Engineer with strong UI/UX design skills. Your task is to generate self-contained, executable code for a single file that can run directly in a sandbox environment. 

Your code must be presented in markdown format:
```<language>
<code>
```

Your code must be written using one of these supported development frameworks and environments:
- React (JavaScript/TypeScript)
- Vue (JavaScript/TypeScript)
- HTML
- Gradio
- Streamlit
- PyGame
- Python Code Interpreter
- JavaScript Code Interpreter

All web framework code (React, Vue, HTML) must be directly rendered in a browser and immediately executable without additional setup. Python-based frameworks should be directly executable in a browser environment. The code to be executed in Code Interpreters must be plain Python or JavaScript programs that do not require any web UI frameworks and standard inputs from the user.


When choosing your development environment for the task:
- For web applications, if no environment is explicitly specified by the user, ALWAYS prefer React or Vue over plain HTML
- Choose the environment that best matches the task's requirements  - web frameworks for interactive applications, specialized Python frameworks for specific needs, and code interpreters for general-purpose code execution (e.g, computational tasks)

Before you begin writing any code, you must follow these fundamental rules:
- You are NOT allowed to start directly with a code block. Before writing code, ALWAYS think carefully step-by-step
- Your response must contain a clear explanation of the solution you are providing
- ALAWYS generate complete, self-contained code in a single file
- Each response must contain only one code block
- Each code block must be completely independent. If modifications are needed, the entire code block must be rewritten
- Make sure it can run by itself by using a default export
- Make sure the program is functional by creating state when needed and having no required props
- Make sure to include all necessary code in one file
- There is no additional files in the local file system, unless you create them inside the same program
- Do not touch project dependencies files like package.json, package-lock.json, requirements.txt, etc.

When developing with React or Vue components, follow these specific requirements:
- Use TypeScript or JavaScript as the language
- Use Tailwind classes for styling. DO NOT USE ARBITRARY VALUES (e.g. 'h-[600px]'). Make sure to use a consistent color palette
- If you use any imports from React like useState or useEffect, make sure to import them directly
- Use Tailwind margin and padding classes to style the components and ensure proper spacing
- DO NOT RETURN THE COMPONENT AT THE END OF THE FILE (e.g. `ReactDOM.render(</>, document.getElementById('root'));`)
- The recharts library is available to be imported, e.g. `import { LineChart, XAxis, ... } from "recharts"` & `<LineChart ...><XAxis dataKey="name"> ...`

For HTML development, ensure that:
- All HTML code must be self-contained in a single file
- Include any necessary CSS and JavaScript within the HTML file
- Ensure the code is directly executable in a browser environment

For Python development, you must follow these constraints:
- Make sure it does not require any user inputs
- Avoid using libraries that require desktop GUI interfaces, with the exceptions of `pygame`, `gradio`, and `streamlit` which are explicitly supported
- For PyGame applications, you have to write the main function as an async function like:
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

DEFAULT_PYTHON_CODE_INTERPRETER_INSTRUCTION = """
Generate self-contained Python code for execution in a code interpreter.
There are standard and pre-installed libraries: aiohttp, beautifulsoup4, bokeh, gensim, imageio, joblib, librosa, matplotlib, nltk, numpy, opencv-python, openpyxl, pandas, plotly, pytest, python-docx, pytz, requests, scikit-image, scikit-learn, scipy, seaborn, soundfile, spacy, textblob, tornado, urllib3, xarray, xlrd, sympy.
Output via stdout, stderr, or render images, plots, and tables.
"""

DEFAULT_JAVASCRIPT_CODE_INTERPRETER_INSTRUCTION = """
Generate JavaScript code suitable for execution in a code interpreter environment. This is not for web page apps.
Ensure the code is self-contained and does not rely on browser-specific APIs.
You can output in stdout, stderr, or render images, plots, and tables.
"""

DEFAULT_HTML_SANDBOX_INSTRUCTION = """
Generate HTML code for a single vanilla HTML file to be executed in a sandbox. You can add style and javascript.
"""

DEFAULT_REACT_SANDBOX_INSTRUCTION = """ Generate typescript for a single-file Next.js 13+ React component tsx file. Pre-installed libs: ["nextjs@14.2.5", "typescript", "@types/node", "@types/react", "@types/react-dom", "postcss", "tailwindcss", "shadcn"] """
'''
Default sandbox prompt instruction.
'''

DEFAULT_VUE_SANDBOX_INSTRUCTION = """ Generate TypeScript for a single-file Vue.js 3+ component (SFC) in .vue format. The component should be a simple custom page in a styled `<div>` element. Do not include <NuxtWelcome /> or reference any external components. Surround the code with ``` in markdown. Pre-installed libs: ["nextjs@14.2.5", "typescript", "@types/node", "@types/react", "@types/react-dom", "postcss", "tailwindcss", "shadcn"], """
'''
Default sandbox prompt instruction for vue.
'''

DEFAULT_PYGAME_SANDBOX_INSTRUCTION = (
'''
Generate a pygame code snippet for a single file.
Write pygame main method in async function like:
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
'''
)

DEFAULT_GRADIO_SANDBOX_INSTRUCTION = """
Generate Python code for a single-file Gradio application using the Gradio library.
"""

DEFAULT_NICEGUI_SANDBOX_INSTRUCTION = """
Generate a Python NiceGUI code snippet for a single file.
"""

DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION = """
Generate Python code for a single-file Streamlit application using the Streamlit library.
The app should automatically reload when changes are made. 
"""

AUTO_SANDBOX_INSTRUCTION = (
"""
You are an expert Software Engineer. Generate code for a single file to be executed in a sandbox. Do not import external files. You can output information if needed. 

The code should be in the markdown format:
```<language>
<code>
```

You can choose from the following sandbox environments:
"""
+ 'Sandbox Environment Name: ' + SandboxEnvironment.PYTHON_CODE_INTERPRETER + '\n' + DEFAULT_PYTHON_CODE_INTERPRETER_INSTRUCTION.strip() + '\n------\n'
+ 'Sandbox Environment Name: ' + SandboxEnvironment.REACT + '\n' + DEFAULT_REACT_SANDBOX_INSTRUCTION.strip() + '\n------\n'
+ 'Sandbox Environment Name: ' + SandboxEnvironment.VUE + '\n' + DEFAULT_VUE_SANDBOX_INSTRUCTION.strip() + '\n------\n'
+ 'Sandbox Environment Name: ' + SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER + '\n' + DEFAULT_JAVASCRIPT_CODE_INTERPRETER_INSTRUCTION.strip() + '\n------\n'
+ 'Sandbox Environment Name: ' + SandboxEnvironment.HTML + '\n' + DEFAULT_HTML_SANDBOX_INSTRUCTION.strip() + '\n------\n'
+ 'Sandbox Environment Name: ' + SandboxEnvironment.GRADIO + '\n' + DEFAULT_GRADIO_SANDBOX_INSTRUCTION.strip() + '\n------\n'
+ 'Sandbox Environment Name: ' + SandboxEnvironment.STREAMLIT + '\n' + DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION.strip() + '\n------\n'
+ 'Sandbox Environment Name: ' + SandboxEnvironment.NICEGUI + '\n' + DEFAULT_NICEGUI_SANDBOX_INSTRUCTION.strip() + '\n------\n'
+ 'Sandbox Environment Name: ' + SandboxEnvironment.PYGAME + '\n' + DEFAULT_PYGAME_SANDBOX_INSTRUCTION.strip() + '\n------\n'
)

DEFAULT_SANDBOX_INSTRUCTIONS: dict[SandboxEnvironment, str] = {
    SandboxEnvironment.AUTO: GENERAL_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.PYTHON_CODE_INTERPRETER: GENERAL_SANDBOX_INSTRUCTION + DEFAULT_PYTHON_CODE_INTERPRETER_INSTRUCTION.strip(),
    SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER: GENERAL_SANDBOX_INSTRUCTION + DEFAULT_JAVASCRIPT_CODE_INTERPRETER_INSTRUCTION.strip(),
    SandboxEnvironment.HTML: GENERAL_SANDBOX_INSTRUCTION + DEFAULT_HTML_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.REACT: GENERAL_SANDBOX_INSTRUCTION + DEFAULT_REACT_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.VUE: GENERAL_SANDBOX_INSTRUCTION + DEFAULT_VUE_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.GRADIO: GENERAL_SANDBOX_INSTRUCTION + DEFAULT_GRADIO_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.STREAMLIT: GENERAL_SANDBOX_INSTRUCTION + DEFAULT_STREAMLIT_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.NICEGUI: GENERAL_SANDBOX_INSTRUCTION + DEFAULT_NICEGUI_SANDBOX_INSTRUCTION.strip(),
    SandboxEnvironment.PYGAME: GENERAL_SANDBOX_INSTRUCTION + DEFAULT_PYGAME_SANDBOX_INSTRUCTION.strip(),
}


SandboxGradioSandboxComponents: TypeAlias =  tuple[
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
    code_dependencies: tuple[list[str], list[str]]
    '''
    The code dependencies for the sandbox (python, npm).
    '''
    btn_list_length: int | None


def create_chatbot_sandbox_state(btn_list_length: int) -> ChatbotSandboxState:
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
        "code_dependencies": ([], []),
        "enabled_round": 0,
        "btn_list_length": btn_list_length
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
    return [gr.update(visible=visible)] *12


def update_visibility_for_single_model(visible: bool, component_cnt: int):
    return [gr.update(visible=visible)] * component_cnt


def extract_python_imports(code: str) -> list[str]:
    '''
    Extract Python package imports using AST parsing.
    Returns a list of top-level package names.
    '''
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    packages: Set[str] = set()
    
    for node in ast.walk(tree):
        try:
            if isinstance(node, ast.Import):
                for name in node.names:
                    # Get the top-level package name from any dotted path
                    # e.g., 'foo.bar.baz' -> 'foo'
                    if name.name:  # Ensure there's a name
                        packages.add(name.name.split('.')[0])
                        
            elif isinstance(node, ast.ImportFrom):
                # Skip relative imports (those starting with dots)
                if node.level == 0 and node.module:
                    # Get the top-level package name
                    # e.g., from foo.bar import baz -> 'foo'
                    packages.add(node.module.split('.')[0])
                    
            # Also check for common dynamic import patterns
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'importlib':
                    # Handle importlib.import_module('package')
                    if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                        packages.add(node.args[0].s.split('.')[0])
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    # Handle __import__('package') and importlib.import_module('package')
                    if node.func.value.id == 'importlib' and node.func.attr == 'import_module':
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                            packages.add(node.args[0].s.split('.')[0])
                    elif node.func.attr == '__import__':
                        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                            packages.add(node.args[0].s.split('.')[0])
        except Exception as e:
            print(f"Error processing node {type(node)}: {e}")
            continue
    
    # Filter out standard library modules using sys.stdlib_module_names
    std_libs = set(sys.stdlib_module_names)
    
    return list(packages - std_libs)

def extract_js_imports(code: str) -> list[str]:
    '''
    Extract npm package imports using Tree-sitter for robust parsing.
    Handles both JavaScript and TypeScript code, including Vue SFC.
    Returns a list of package names.
    '''
    try:
        # For Vue SFC, extract the script section first
        script_match = re.search(r'<script.*?>(.*?)</script>', code, re.DOTALL)
        if script_match:
            code = script_match.group(1).strip()

        # Initialize parsers with language modules
        ts_parser = Parser(Language(tree_sitter_typescript.language_tsx()))
        js_parser = Parser(Language(tree_sitter_javascript.language()))
        
        # Try parsing as TypeScript first, then JavaScript
        code_bytes = bytes(code, "utf8")
        try:
            tree = ts_parser.parse(code_bytes)
        except Exception as e:
            print(f"TypeScript parsing failed: {e}")
            try:
                tree = js_parser.parse(code_bytes)
            except Exception as e:
                print(f"JavaScript parsing failed: {e}")
                tree = None

        if tree is None:
            raise Exception("Both TypeScript and JavaScript parsing failed")
        
        packages: Set[str] = set()
        
        def extract_package_name(node: Node) -> str | None:
            '''Extract package name from string literal or template string'''
            if node.type in ['string', 'string_fragment']:
                # Handle regular string literals
                pkg_path = code[node.start_byte:node.end_byte].strip('"\'')
                if not pkg_path.startswith('.'):
                    # Handle scoped packages differently
                    if pkg_path.startswith('@'):
                        parts = pkg_path.split('/')
                        if len(parts) >= 2:
                            return '/'.join(parts[:2])  # Return @scope/package
                    return pkg_path.split('/')[0]  # Return just the package name for non-scoped packages
            elif node.type == 'template_string':
                # Handle template literals
                content = ''
                has_template_var = False
                for child in node.children:
                    if child.type == 'string_fragment':
                        content += code[child.start_byte:child.end_byte]
                    elif child.type == 'template_substitution':
                        has_template_var = True
                        continue
                
                if not content or content.startswith('.'):
                    return None

                if has_template_var:
                    if content.endswith('-literal'):
                        return 'package-template-literal'
                    return None

                if content.startswith('@'):
                    parts = content.split('/')
                    if len(parts) >= 2:
                        return '/'.join(parts[:2])
                return content.split('/')[0]
            return None
        
        def visit_node(node: Node) -> None:
            if node.type == 'import_statement':
                # Handle ES6 imports
                string_node = node.child_by_field_name('source')
                if string_node:
                    pkg_name = extract_package_name(string_node)
                    if pkg_name:
                        packages.add(pkg_name)
                        
            elif node.type == 'export_statement':
                # Handle re-exports
                source = node.child_by_field_name('source')
                if source:
                    pkg_name = extract_package_name(source)
                    if pkg_name:
                        packages.add(pkg_name)
                        
            elif node.type == 'call_expression':
                # Handle require calls and dynamic imports
                func_node = node.child_by_field_name('function')
                if func_node and func_node.text:
                    func_name = func_node.text.decode('utf8')
                    if func_name in ['require', 'import']:
                        args = node.child_by_field_name('arguments')
                        if args and args.named_children:
                            arg = args.named_children[0]
                            pkg_name = extract_package_name(arg)
                            if pkg_name:
                                packages.add(pkg_name)
            
            # Recursively visit children
            for child in node.children:
                visit_node(child)
        
        visit_node(tree.root_node)
        return list(packages)
        
    except Exception as e:
        print(f"Tree-sitter parsing failed: {e}")
        # Fallback to basic regex parsing if tree-sitter fails
        packages: Set[str] = set()
        
        # First try to extract script section for Vue SFC
        script_match = re.search(r'<script.*?>(.*?)</script>', code, re.DOTALL)
        if script_match:
            code = script_match.group(1).strip()
        
        # Look for imports
        import_patterns = [
            r'(?:import|require)\s*\(\s*[\'"](@?[\w-]+(?:/[\w-]+)*)[\'"]',  # dynamic imports
            r'(?:import|from)\s+[\'"](@?[\w-]+(?:/[\w-]+)*)[\'"]',  # static imports
            r'require\s*\(\s*[\'"](@?[\w-]+(?:/[\w-]+)*)[\'"]',  # require statements
        ]
        
        for pattern in import_patterns:
            matches = re.finditer(pattern, code)
            for match in matches:
                pkg_name = match.group(1)
                if not pkg_name.startswith('.'):
                    if pkg_name.startswith('@'):
                        parts = pkg_name.split('/')
                        if len(parts) >= 2:
                            packages.add('/'.join(parts[:2]))
                    else:
                        packages.add(pkg_name.split('/')[0])
        
        return list(packages)

def determine_python_environment(code: str, imports: list[str]) -> SandboxEnvironment | None:
    '''
    Determine Python sandbox environment based on imports and AST analysis.
    '''
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Check for specific framework usage patterns
            if isinstance(node, ast.Name) and node.id == 'gr':
                return SandboxEnvironment.GRADIO
            elif isinstance(node, ast.Name) and node.id == 'st':
                return SandboxEnvironment.STREAMLIT
    except SyntaxError:
        pass

    # Check imports for framework detection
    if 'pygame' in imports:
        return SandboxEnvironment.PYGAME
    elif 'gradio' in imports:
        return SandboxEnvironment.GRADIO
    elif 'streamlit' in imports:
        return SandboxEnvironment.STREAMLIT
    elif 'nicegui' in imports:
        return SandboxEnvironment.NICEGUI
    
    return SandboxEnvironment.PYTHON_CODE_INTERPRETER

def determine_js_environment(code: str, imports: list[str]) -> SandboxEnvironment | None:
    '''
    Determine JavaScript/TypeScript sandbox environment based on imports and AST analysis.
    '''
    try:
        # Initialize parser
        ts_parser = Parser(Language(tree_sitter_typescript.language_tsx()))
        
        # Parse the code
        tree = ts_parser.parse(bytes(code, "utf8"))
        
        def has_framework_patterns(node: Node) -> bool:
            # Check for React patterns
            if node.type in ['jsx_element', 'jsx_self_closing_element']:
                return True
            # Check for Vue template
            elif node.type == 'template_element':
                return True
            return False
        
        # Check for framework-specific patterns in the AST
        cursor = tree.walk()
        reached_end = False
        while not reached_end:
            if has_framework_patterns(cursor.node):
                if cursor.node.type.startswith('jsx'):
                    return SandboxEnvironment.REACT
                elif cursor.node.type == 'template_element':
                    return SandboxEnvironment.VUE
            
            reached_end = not cursor.goto_next_sibling()
            if reached_end and cursor.goto_parent():
                reached_end = not cursor.goto_next_sibling()
    
    except Exception:
        pass
    
    # Check imports for framework detection
    react_packages = {'react', '@react', 'next', '@next'}
    vue_packages = {'vue', '@vue', 'nuxt', '@nuxt'}
    
    if any(pkg in react_packages for pkg in imports):
        return SandboxEnvironment.REACT
    elif any(pkg in vue_packages for pkg in imports):
        return SandboxEnvironment.VUE
    
    return SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER


def detect_js_ts_code_lang(code: str) -> str:
    '''
    Detect whether code is JavaScript or TypeScript using Tree-sitter AST parsing.
    Handles Vue SFC, React, and regular JS/TS files.
    
    Args:
        code (str): The code to analyze
        
    Returns:
        str: 'typescript' if TypeScript patterns are found, 'javascript' otherwise
    '''
    # Quick check for explicit TypeScript in Vue SFC
    if '<script lang="ts">' in code or '<script lang="typescript">' in code:
        return 'typescript'

    try:
        # Initialize TypeScript parser
        ts_parser = Parser(Language(tree_sitter_typescript.language_tsx()))
        
        # Parse the code
        tree = ts_parser.parse(bytes(code, "utf8"))
        
        def has_typescript_patterns(node: Node) -> bool:
            # Check for TypeScript-specific syntax
            if node.type in {
                'type_annotation',           # Type annotations
                'type_alias_declaration',    # type Foo = ...
                'interface_declaration',     # interface Foo
                'enum_declaration',          # enum Foo
                'implements_clause',         # implements Interface
                'type_parameter',            # Generic type parameters
                'type_assertion',            # Type assertions
                'type_predicate',           # Type predicates in functions
                'type_arguments',           # Generic type arguments
                'readonly_type',            # readonly keyword
                'mapped_type',              # Mapped types
                'conditional_type',         # Conditional types
                'union_type',               # Union types
                'intersection_type',        # Intersection types
                'tuple_type',              # Tuple types
                'optional_parameter',       # Optional parameters
                'decorator',                # Decorators
                'ambient_declaration',      # Ambient declarations
                'declare_statement',        # declare keyword
                'accessibility_modifier',   # private/protected/public
            }:
                return True
                
            # Check for type annotations in variable declarations
            if node.type == 'variable_declarator':
                for child in node.children:
                    if child.type == 'type_annotation':
                        return True
            
            # Check for return type annotations in functions
            if node.type in {'function_declaration', 'method_definition', 'arrow_function'}:
                for child in node.children:
                    if child.type == 'type_annotation':
                        return True
            
            return False

        # Walk the AST to find TypeScript patterns
        cursor = tree.walk()
        
        def visit_node() -> bool:
            if has_typescript_patterns(cursor.node):
                return True
                
            # Check children
            if cursor.goto_first_child():
                while True:
                    if visit_node():
                        return True
                    if not cursor.goto_next_sibling():
                        break
                cursor.goto_parent()
            
            return False

        if visit_node():
            return 'typescript'

    except Exception as e:
        print(f"Tree-sitter parsing error: {e}")
        # Fallback to basic checks if parsing fails
        pass

    return 'javascript'


def extract_code_from_markdown(message: str, enable_auto_env: bool=False) -> tuple[str, str, tuple[list[str], list[str]], SandboxEnvironment | None] | None:
    '''
    Extracts code from a markdown message by parsing code blocks directly.
    Determines sandbox environment based on code content and frameworks used.

    Returns:
        tuple[str, str, tuple[list[str], list[str]], SandboxEnvironment | None]: A tuple:
            1. code - the longest code block found
            2. code language
            3. sandbox python and npm dependencies (extracted using static analysis)
            4. sandbox environment determined from code content
    '''
    # Find all code blocks
    code_block_regex = r'```(?P<code_lang>\w+)?\n(?P<code>.*?)```'
    matches = list(re.finditer(code_block_regex, message, re.DOTALL))
    
    if not matches:
        return None
        
    # Find the longest code block
    longest_match = max(matches, key=lambda m: len(m.group('code')))
    code = longest_match.group('code').strip()
    code_lang = (longest_match.group('code_lang') or '').lower()
    
    # Skip empty code blocks
    if not code:
        return None

    # Extract package dependencies using static analysis
    python_packages: list[str] = []
    npm_packages: list[str] = []
    
    # Determine sandbox environment based on language and imports
    if code_lang in ['py', 'python', 'pygame', 'pygame-sdl2', 'pygame-sdl2-ce', 'gradio', 'streamlit', 'nicegui']:
        python_packages = extract_python_imports(code)
        sandbox_env_name = determine_python_environment(code, python_packages)
    elif code_lang in ['vue', 'vue3', 'vue2']:
        npm_packages = extract_js_imports(code)
        sandbox_env_name = SandboxEnvironment.VUE
        code_lang = detect_js_ts_code_lang(code)
    elif code_lang in ['html','xhtml', 'xml'] or ('<!DOCTYPE html>' in code or '<html' in code):
        sandbox_env_name = SandboxEnvironment.HTML
        code_lang = 'html'
    elif code_lang in ['react', 'reactjs', 'react-native', 'react-hook-form']:
        npm_packages = extract_js_imports(code)
        sandbox_env_name = SandboxEnvironment.REACT
        code_lang = detect_js_ts_code_lang(code)
    elif code_lang in ['js', 'javascript',  'jsx', 'coffeescript', 'ecmascript', 'javascript', 'js', 'node']:
        code_lang = 'javascript'
        npm_packages = extract_js_imports(code)
        sandbox_env_name = determine_js_environment(code, npm_packages)
    elif code_lang in ['ts', 'typescript']:
        code_lang = 'typescript'
        npm_packages = extract_js_imports(code)
        sandbox_env_name = determine_js_environment(code, npm_packages)
    else:
        sandbox_env_name = None

    if enable_auto_env and sandbox_env_name is None:
        return None

    return code, code_lang, (python_packages, npm_packages), sandbox_env_name


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

def install_pip_dependencies(sandbox: Sandbox, dependencies: list[str]):
    '''
    Install pip dependencies in the sandbox.
    '''
    if not dependencies:
        return
        
    def log_output(message):
        print(f"pip: {message}")
        
    sandbox.commands.run(
        f"uv pip install --system {' '.join(dependencies)}",
        timeout=60 * 3,
        on_stdout=log_output,
        on_stderr=log_output,
    )


def install_npm_dependencies(sandbox: Sandbox, dependencies: list[str]):
    '''
    Install npm dependencies in the sandbox.
    '''
    if not dependencies:
        return
    sandbox.commands.run(
        f"npm install {' '.join(dependencies)}",
        timeout=60 * 3,
        on_stdout=lambda message: print(message),
        on_stderr=lambda message: print(message),
    )


def run_code_interpreter(code: str, code_language: str | None, code_dependencies: tuple[list[str], list[str]]) -> str:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.
    """
    sandbox = CodeSandbox(
        api_key=E2B_API_KEY,
    )

    sandbox.commands.run("pip install uv",
                         timeout=60 * 3,
                         on_stderr=lambda message: print(message),)
    
    python_dependencies, npm_dependencies = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)
    install_npm_dependencies(sandbox, npm_dependencies)

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


def run_html_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> str:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.

    Returns:
        url for remote sandbox
    """
    stderr = ""
    sandbox = Sandbox(
        api_key=E2B_API_KEY,
    )

    python_dependencies, npm_dependencies = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)
    install_npm_dependencies(sandbox, npm_dependencies)

    sandbox.files.make_dir('myhtml')
    file_path = "~/myhtml/main.html"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    try:
        process = sandbox.commands.run(
            "python -m http.server 3000", background=True)  # start http server
    except CommandExitException as e:
        print(f"Error starting http server: {e}")
        stderr = "\n".join(e.stderr)
    
    # get game server url
    host = sandbox.get_host(3000)
    url = f"https://{host}"
    return url + '/myhtml/main.html', stderr


def run_react_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> str:
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

    python_dependencies, npm_dependencies = code_dependencies
    
    # Stream logs for Python dependencies
    if python_dependencies:
        print("Installing Python dependencies...")
        install_pip_dependencies(sandbox, python_dependencies)
        print("Python dependencies installed.")
    
    # Stream logs for NPM dependencies
    if npm_dependencies:
        print("Installing NPM dependencies...")
        def log_output(message):
            print(f"npm: {message}")
        
        sandbox.commands.run(
            f"npm install {' '.join(npm_dependencies)}",
            timeout=60 * 3,
            on_stdout=log_output,
            on_stderr=log_output,
        )
        print("NPM dependencies installed.")

    # set up the sandbox
    print("Setting up sandbox directory structure...")
    sandbox.files.make_dir('pages')
    file_path = "~/pages/index.tsx"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)
    print("Code files written successfully.")

    # get the sandbox url
    print("Starting development server...")
    sandbox_url = 'https://' + sandbox.get_host(3000)
    print(f"Sandbox URL ready: {sandbox_url}")
    
    return sandbox_url


def run_vue_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> str:
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

    python_dependencies, npm_dependencies = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)
    install_npm_dependencies(sandbox, npm_dependencies)

    # Get the sandbox URL
    sandbox_url = 'https://' + sandbox.get_host(3000)
    return sandbox_url


def run_pygame_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> str:
    """
    Executes the provided code within a sandboxed environment and returns the output.

    Args:
        code (str): The code to be executed.

    Returns:
        url for remote sandbox
    """
    stderr = ""
    sandbox = Sandbox(
        api_key=E2B_API_KEY,
    )

    sandbox.files.make_dir('mygame')
    file_path = "~/mygame/main.py"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)
    sandbox.commands.run("pip install uv",
                         timeout=60 * 3,
                         # on_stdout=lambda message: print(message),
                         on_stderr=lambda message: print(message),)
    sandbox.commands.run("uv pip install --system pygame pygbag black",
                         timeout=60 * 3,
                         # on_stdout=lambda message: print(message),
                         on_stderr=lambda message: print(message),)
    
    python_dependencies, npm_dependencies = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)
    install_npm_dependencies(sandbox, npm_dependencies)

    # build the pygame code
    try:
        sandbox.commands.run(
            "pygbag --build ~/mygame",  # build game
            timeout=60 * 5,
            # on_stdout=lambda message: print(message),
            # on_stderr=lambda message: print(message),
        )
    except CommandExitException as e:
        print(f"Error building pygame code: {e}")
        stderr = "\n".join(e.stderr)
    
    try:
        process = sandbox.commands.run(
            "python -m http.server 3000", background=True)  # start http server
    except CommandExitException as e:
        print(f"Error starting http server: {e}")
        stderr = "\n".join(e.stderr)

    # get game server url
    host = sandbox.get_host(3000)
    url = f"https://{host}"
    return url + '/mygame/build/web/', stderr


def run_nicegui_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> str:
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
        "uv pip install --system --upgrade nicegui",
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

    python_dependencies, npm_dependencies = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)
    install_npm_dependencies(sandbox, npm_dependencies)

    process = sandbox.commands.run(
        "python ~/mynicegui/main.py", background=True)

    # get web gui url
    host = sandbox.get_host(port=8080)
    url = f"https://{host}"
    return url


def run_gradio_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> str:
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

    sandbox.commands.run("pip install uv", timeout=60 * 3, on_stdout=lambda message: print(message), on_stderr=lambda message: print(message))
    # set up the sandbox
    file_path = "~/app.py"
    sandbox.files.write(path=file_path, data=code, request_timeout=60)

    python_dependencies, npm_dependencies = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)
    install_npm_dependencies(sandbox, npm_dependencies)

    # get the sandbox url
    sandbox_url = 'https://' + sandbox.get_host(7860)
    return sandbox_url


def run_streamlit_sandbox(code: str, code_dependencies: tuple[list[str], list[str]]) -> str:
    sandbox = Sandbox(api_key=E2B_API_KEY)

    setup_commands = ["pip install uv", "uv pip install --system streamlit"]

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

    python_dependencies, npm_dependencies = code_dependencies
    install_pip_dependencies(sandbox, python_dependencies)
    install_npm_dependencies(sandbox, npm_dependencies)

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
    Gradio Handler when code is edited manually by users.
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
    code, code_language, code_dependencies, env_selection = extract_result

    if sandbox_state['code_to_execute'] == code and sandbox_state['code_language'] == code_language and sandbox_state['code_dependencies'] == code_dependencies:
        # skip if no changes
        yield gr.skip(), gr.skip(), gr.skip()
        return

    if code_language == 'tsx':
        code_language = 'typescript'
    code_language = code_language.lower() if code_language and code_language.lower(
        # ensure gradio supports the code language
    ) in VALID_GRADIO_CODE_LANGUAGES else None

    sandbox_state['code_to_execute'] = code
    sandbox_state['code_language'] = code_language
    sandbox_state['code_dependencies'] = code_dependencies
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

    # Initialize output with loading message
    output_text = "### Sandbox Execution Log\n\n"
    yield (
        gr.Markdown(value=output_text + "🔄 Initializing sandbox environment...", visible=True),
        SandboxComponent(visible=False),
        gr.Code(value=code, language=code_language, visible=True),
    )

    sandbox_env = sandbox_state['sandbox_environment'] if sandbox_state['sandbox_environment'] != SandboxEnvironment.AUTO else sandbox_state['auto_selected_sandbox_environment']
    code_dependencies = sandbox_state['code_dependencies']

    def update_output(message: str):
        nonlocal output_text
        output_text += f"\n{message}"
        return (
            gr.Markdown(value=output_text, visible=True, sanitize_html=False),
            gr.skip(),
            gr.skip(),
        )

    match sandbox_env:
        case SandboxEnvironment.HTML:
            yield update_output("🔄 Setting up HTML sandbox...")
            url, stderr = run_html_sandbox(code=code, code_dependencies=code_dependencies)
            if stderr:
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ HTML sandbox ready!")
                yield (
                    gr.Markdown(value=output_text, visible=True),
                    SandboxComponent(
                        value=(url, code),
                        label="Example",
                        visible=True,
                        key="newsandbox",
                    ),
                    gr.skip(),
                )
        case SandboxEnvironment.REACT:
            yield update_output("🔄 Setting up React sandbox...")
            yield update_output("⚙️ Installing dependencies...")
            url = run_react_sandbox(code=code, code_dependencies=code_dependencies)
            yield update_output("✅ React sandbox ready!")
            yield (
                gr.Markdown(value=output_text, visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.VUE:
            yield update_output("🔄 Setting up Vue sandbox...")
            yield update_output("⚙️ Installing dependencies...")
            url = run_vue_sandbox(code=code, code_dependencies=code_dependencies)
            yield update_output("✅ Vue sandbox ready!")
            yield (
                gr.Markdown(value=output_text, visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.PYGAME:
            yield update_output("🔄 Setting up PyGame sandbox...")
            yield update_output("⚙️ Installing PyGame dependencies...")
            url, stderr = run_pygame_sandbox(code=code, code_dependencies=code_dependencies)
            if stderr:
                yield update_output(f"### Stderr:\n```\n{stderr}\n```\n\n")
            else:
                yield update_output("✅ PyGame sandbox ready!")
                yield (
                    gr.Markdown(value=output_text, visible=True),
                    SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.GRADIO:
            yield update_output("🔄 Setting up Gradio sandbox...")
            yield update_output("⚙️ Installing Gradio dependencies...")
            url = run_gradio_sandbox(code=code, code_dependencies=code_dependencies)
            yield update_output("✅ Gradio sandbox ready!")
            yield (
                gr.Markdown(value=output_text, visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.STREAMLIT:
            yield update_output("🔄 Setting up Streamlit sandbox...")
            yield update_output("⚙️ Installing Streamlit dependencies...")
            url = run_streamlit_sandbox(code=code, code_dependencies=code_dependencies)
            yield update_output("✅ Streamlit sandbox ready!")
            yield (
                gr.Markdown(value=output_text, visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.NICEGUI:
            yield update_output("🔄 Setting up NiceGUI sandbox...")
            yield update_output("⚙️ Installing NiceGUI dependencies...")
            url = run_nicegui_sandbox(code=code, code_dependencies=code_dependencies)
            yield update_output("✅ NiceGUI sandbox ready!")
            yield (
                gr.Markdown(value=output_text, visible=True),
                SandboxComponent(
                    value=(url, code),
                    label="Example",
                    visible=True,
                    key="newsandbox",
                ),
                gr.skip(),
            )
        case SandboxEnvironment.PYTHON_CODE_INTERPRETER:
            yield update_output("🔄 Running Python Code Interpreter...")
            output = run_code_interpreter(
                code=code, code_language='python', code_dependencies=code_dependencies
            )
            yield update_output("✅ Code execution complete!")
            yield (
                gr.Markdown(value=output_text + "\n\n" + output, sanitize_html=False, visible=True),
                SandboxComponent(
                    value=('', ''),
                    label="Example",
                    visible=False,
                    key="newsandbox",
                ),
                gr.skip()
            )
        case SandboxEnvironment.JAVASCRIPT_CODE_INTERPRETER:
            yield update_output("🔄 Running JavaScript Code Interpreter...")
            output = run_code_interpreter(
                code=code, code_language='javascript', code_dependencies=code_dependencies
            )
            yield update_output("✅ Code execution complete!")
            yield (
                gr.Markdown(value=output_text + "\n\n" + output, visible=True),
                SandboxComponent(
                    value=('', ''),
                    label="Example",
                    visible=False,
                    key="newsandbox",
                ),
                gr.skip()
            )
        case _:
            raise ValueError(
                f"Unsupported sandbox environment: {sandbox_state['sandbox_environment']}")
