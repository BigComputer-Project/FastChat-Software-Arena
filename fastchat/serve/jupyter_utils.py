"""Jupyter notebook utilities."""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
import os
import json
from traitlets.config import Config

# Define header message and templates
header_message = """<p align="center">
  <img src="https://huggingface.co/spaces/lvwerra/jupyter-agent/resolve/main/jupyter-agent.png" />
</p>
<p style="text-align:center;">Let a LLM agent write and execute code inside a notebook!</p>"""

system_template = """\
<details>
  <summary style="display: flex; align-items: center;">
    <div class="alert alert-block alert-info" style="margin: 0; width: 100%;">
      <b>System: <span class="arrow">▶</span></b>
    </div>
  </summary>
  <div class="alert alert-block alert-info">
    {}
  </div>
</details>
<style>
details > summary .arrow {{
  display: inline-block;
  transition: transform 0.2s;
}}
details[open] > summary .arrow {{
  transform: rotate(90deg);
}}
</style>
"""

user_template = """<div class="alert alert-block alert-success">
<b>User:</b> {}
</div>
"""

bad_html_bad = """input[type="file"] {
  display: block;
}"""

config = Config()
html_exporter = HTMLExporter(config=config, template_name="classic")

def create_base_notebook(messages):
    """Create a base Jupyter notebook with initial messages."""
    base_notebook = {
        "metadata": {
            "kernel_info": {"name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 0,
        "cells": []
    }
    base_notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": header_message
            })

    if len(messages)==0:
        base_notebook["cells"].append({
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "source": "",
                            "outputs": []
                        })

    # code_cell_counter = 0
    
    for message in messages:
        print(message)
        if message["role"] == "system":
            text = system_template.format(message["content"].replace('\n', '<br>'))
            base_notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": text
                })
        elif message["role"] == "user":
            text = user_template.format(message["content"].replace('\n', '<br>'))
            base_notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": text
                })

        elif message["role"] == "assistant" and "tool_calls" in message:
            base_notebook["cells"].append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": message["content"],
                "outputs": []
            })

        elif message["role"] == "ipython":
            # code_cell_counter +=1
            base_notebook["cells"][-1]["outputs"] = message["nbformat"]
            # base_notebook["cells"][-1]["execution_count"] = code_cell_counter

        elif message["role"] == "assistant" and "tool_calls" not in message:
            base_notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": message["content"]
            })
            
        else:
            raise ValueError(message)
        
    return base_notebook

def update_notebook_display(notebook_data):
    """Convert notebook to HTML for display."""
    # try:
    notebook = nbformat.from_dict(notebook_data)
    notebook_body, _ = html_exporter.from_notebook_node(notebook)
    
    # Add custom CSS for better notebook display
    custom_css = """
    <style>
        /* Base styles */
        body {
            color: #333333 !important;
            background-color: #ffffff !important;
        }
        
        /* Make notebook fit the container */
        .jp-Notebook {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: hidden !important;
            background-color: #ffffff !important;
            color: #333333 !important;
        }
        
        /* Adjust cell widths */
        .jp-Cell {
            width: 100% !important;
            max-width: 100% !important;
            margin: 10px 0 !important;
            border: 1px solid #e0e0e0 !important;
            border-radius: 4px !important;
            background-color: #ffffff !important;
            color: #333333 !important;
        }
        
        /* Code cell styling */
        .jp-CodeCell {
            background-color: #f8f8f8 !important;
        }
        
        .jp-CodeCell-input {
            overflow-x: auto !important;
            background-color: #f8f8f8 !important;
            padding: 10px !important;
            font-family: monospace !important;
            font-size: 14px !important;
            line-height: 1.4 !important;
            color: #333333 !important;
        }
        
        .jp-CodeCell pre {
            color: #333333 !important;
        }
        
        /* Markdown cell styling */
        .jp-MarkdownCell {
            padding: 10px !important;
            background-color: #ffffff !important;
            color: #333333 !important;
        }
        
        .jp-MarkdownCell p {
            color: #333333 !important;
        }
        
        /* Improve output display */
        .jp-OutputArea {
            padding: 10px !important;
            background-color: #ffffff !important;
            border-top: 1px solid #e0e0e0 !important;
            color: #333333 !important;
        }
        
        .jp-OutputArea-output {
            overflow-x: auto !important;
            max-width: 100% !important;
            padding: 5px !important;
            font-family: monospace !important;
            font-size: 14px !important;
            line-height: 1.4 !important;
            color: #333333 !important;
        }
        
        /* Output stream styling */
        .jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stdout"] {
            background-color: #f8f8f8 !important;
            border-radius: 4px !important;
            padding: 8px !important;
            color: #333333 !important;
        }
        
        .jp-OutputArea-output[data-mime-type="application/vnd.jupyter.stderr"] {
            background-color: #fff0f0 !important;
            color: #cc0000 !important;
            border-radius: 4px !important;
            padding: 8px !important;
        }
        
        /* Adjust image outputs */
        .jp-RenderedImage {
            max-width: 100% !important;
            height: auto !important;
        }
        
        /* Make text wrap properly */
        .jp-RenderedText {
            overflow-x: auto !important;
            white-space: pre-wrap !important;
            color: #333333 !important;
        }
        
        .notebook-container {
            width: 100% !important;
            max-width: 100% !important;
            padding: 1rem !important;
            box-sizing: border-box !important;
            overflow-x: hidden !important;
            background-color: #ffffff !important;
            box-shadow: 0 0 10px rgba(0,0,0,0.1) !important;
            color: #333333 !important;
        }

        /* Error message styling */
        .notebook-error {
            background-color: #fee !important;
            border: 1px solid #fcc !important;
            border-radius: 4px !important;
            padding: 1rem !important;
            margin: 1rem 0 !important;
            color: #c00 !important;
            font-family: monospace !important;
        }
        
        /* Alert styling */
        .alert {
            padding: 15px !important;
            margin-bottom: 20px !important;
            border: 1px solid transparent !important;
            border-radius: 4px !important;
        }
        
        .alert-info {
            color: #004085 !important;
            background-color: #cce5ff !important;
            border-color: #b8daff !important;
        }
        
        .alert-success {
            color: #155724 !important;
            background-color: #d4edda !important;
            border-color: #c3e6cb !important;
        }
        
        /* Code syntax highlighting */
        .highlight {
            background-color: #f8f8f8 !important;
            padding: 10px !important;
            border-radius: 4px !important;
            color: #333333 !important;
        }
        
        .highlight pre {
            margin: 0 !important;
            color: #333333 !important;
        }

        /* Additional text color overrides */
        p, span, div, h1, h2, h3, h4, h5, h6 {
            color: #333333 !important;
        }

        code {
            color: #333333 !important;
            background-color: #f8f8f8 !important;
        }

        pre {
            color: #333333 !important;
            background-color: #f8f8f8 !important;
        }
    </style>
    """
    
    # Wrap the notebook in a container div
    notebook_html = f"""
    <div class="notebook-container">
        {custom_css}
        {notebook_body}
    </div>
    """
    return notebook_html
    # except Exception as e:
    #     error_html = f"""
    #     <div class="notebook-error">
    #         {custom_css}
    #         Error rendering notebook: {str(e)}
    #     </div>
    #     """
    #     return error_html
