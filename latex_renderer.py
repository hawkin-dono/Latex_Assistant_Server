import os
import tempfile
import subprocess
import shutil
import uuid
import base64
from typing import Dict
import html
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def render_latex_to_html(latex_content: str) -> Dict[str, str]:
    """
    Render a complete LaTeX document to HTML using server-side processing
    
    Args:
        latex_content: The full LaTeX document content
        
    Returns:
        Dict containing HTML representation of the rendered LaTeX
    """
    try:
        # Create a unique temporary directory
        temp_dir = tempfile.mkdtemp()
        unique_id = str(uuid.uuid4())
        input_file = os.path.join(temp_dir, f"document_{unique_id}.tex")
        
        try:
            # Write the LaTeX content to a file
            with open(input_file, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            
            logger.info(f"Created temporary LaTeX file: {input_file}")
            
            # Run pdflatex to generate PDF
            logger.info("Running pdflatex...")
            pdf_process = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-output-directory", temp_dir, input_file],
                capture_output=True,
                text=True,
                check=False
            )
            
            pdf_file = os.path.join(temp_dir, f"document_{unique_id}.pdf")
            
            # Check if PDF was created
            if not os.path.exists(pdf_file):
                logger.error("PDF generation failed")
                logger.error(f"pdflatex stdout: {pdf_process.stdout}")
                logger.error(f"pdflatex stderr: {pdf_process.stderr}")
                
                # If PDF generation failed, fall back to MathJax rendering
                return fallback_rendering(latex_content, error_msg="LaTeX compilation failed")
            
            logger.info(f"PDF generated successfully: {pdf_file}")
            
            # Try to convert PDF to HTML using pdf2htmlEX if available
            try:
                logger.info("Converting PDF to HTML using pdf2htmlEX...")
                output_html = os.path.join(temp_dir, f"document_{unique_id}.html")
                
                html_process = subprocess.run(
                    ["pdf2htmlEX", "--dest-dir", temp_dir, pdf_file],
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if os.path.exists(output_html):
                    # Read the HTML file
                    with open(output_html, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    
                    logger.info("Successfully converted PDF to HTML")
                    return {"html": html_content, "type": "pdf2html"}
                else:
                    # pdf2htmlEX failed or isn't installed
                    logger.warning("pdf2htmlEX conversion failed, embedding PDF")
            except FileNotFoundError:
                # pdf2htmlEX not installed
                logger.warning("pdf2htmlEX not found, embedding PDF")
            
            # Fallback: Embed the PDF directly
            with open(pdf_file, 'rb') as f:
                pdf_data = f.read()
            
            pdf_base64 = base64.b64encode(pdf_data).decode('utf-8')
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>LaTeX Document</title>
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        height: 100vh;
                    }}
                    .pdf-container {{
                        width: 100%;
                        height: 100%;
                    }}
                </style>
            </head>
            <body>
                <div class="pdf-container">
                    <embed src="data:application/pdf;base64,{pdf_base64}" type="application/pdf" width="100%" height="100%" />
                </div>
            </body>
            </html>
            """
            
            logger.info("Returning PDF embed HTML")
            return {"html": html_content, "type": "pdf_embed"}
            
        finally:
            # Clean up the temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
            
    except Exception as e:
        logger.exception("Error in LaTeX rendering process")
        # If server-side processing fails completely, fall back to MathJax rendering
        return fallback_rendering(latex_content, error_msg=str(e))

def fallback_rendering(latex_content: str, error_msg: str = "") -> Dict[str, str]:
    """
    Fallback rendering using MathJax when server-side processing fails
    """
    logger.info("Using fallback MathJax rendering")
    # Escape HTML entities in the LaTeX content for safe display
    escaped_latex = html.escape(latex_content)
    
    # Basic extraction of document structure for better rendering
    title_match = re.search(r'\\title\{(.*?)\}', latex_content)
    title = title_match.group(1) if title_match else "LaTeX Document"
    
    author_match = re.search(r'\\author\{(.*?)\}', latex_content)
    author = author_match.group(1) if author_match else ""
    
    # Create a more document-like structure
    document_body = process_latex_content(latex_content)
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>{html.escape(title)}</title>
        <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body {{{{
                font-family: 'Computer Modern', Georgia, serif;
                line-height: 1.5;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #FFFFFF;
                color: #333333;
            }}}}
            .document {{{{
                background-color: #FFFFFF;
                padding: 40px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}}}
            .title {{{{
                font-size: 2em;
                text-align: center;
                margin-bottom: 0.5em;
            }}}}
            .author {{{{
                text-align: center;
                margin-bottom: 2em;
            }}}}
            h1, h2, h3, h4, h5, h6 {{{{
                font-weight: bold;
                margin-top: 1.5em;
                margin-bottom: 0.5em;
            }}}}
            h1 {{{{ font-size: 1.7em; }}}}
            h2 {{{{ font-size: 1.5em; }}}}
            p {{{{ margin-bottom: 1em; }}}}
            .equation {{{{
                margin: 1em 0;
                text-align: center;
            }}}}
            .exercise {{{{
                margin: 1.5em 0;
                padding: 1em;
                border-left: 4px solid #4CAF50;
                background-color: #f9f9f9;
            }}}}
            .solution {{{{
                margin: 1.5em 0;
                padding: 1em;
                border-left: 4px solid #2196F3;
                background-color: #f9f9f9;
            }}}}
            pre.latex-code {{{{
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                white-space: pre-wrap;
                overflow-x: auto;
                font-family: monospace;
                font-size: 14px;
                margin-top: 20px;
            }}}}
            .source-toggle {{{{
                background-color: #eeeeee;
                border: none;
                padding: 5px 10px;
                cursor: pointer;
                border-radius: 3px;
            }}}}
            .error-message {{{{
                color: #D32F2F;
                background-color: #FFEBEE;
                padding: 10px;
                margin: 20px 0;
                border-left: 4px solid #D32F2F;
            }}}}
        </style>
        <script>
            function toggleSource() {{{{
                var source = document.getElementById('latex-source');
                if (source.style.display === 'none') {{{{
                    source.style.display = 'block';
                }}}} else {{{{
                    source.style.display = 'none';
                }}}}
            }}}}
            
            window.MathJax = {{{{
                tex: {{{{
                    inlineMath: [['$', '$']],
                    displayMath: [['$$', '$$']],
                    processEscapes: true,
                    processEnvironments: true,
                    packages: ['base', 'ams', 'noerrors', 'noundefined']
                }}}},
                options: {{{{
                    ignoreHtmlClass: 'latex-source'
                }}}}
            }}}};
        </script>
    </head>
    <body>
        <div class="document">
            <div class="title">{html.escape(title)}</div>
            <div class="author">{html.escape(author)}</div>
            
            {f'<div class="error-message">Server-side rendering failed: {error_msg}</div>' if error_msg else ''}
            
            <div id="rendered-content">
                {document_body}
            </div>
            
            <hr style="margin-top: 30px;">
            <button class="source-toggle" onclick="toggleSource()">Show/Hide LaTeX Source</button>
            <pre id="latex-source" class="latex-code" style="display: none;">{escaped_latex}</pre>
        </div>
    </body>
    </html>
    """
    
    return {"html": html_content, "type": "fallback"}

def process_latex_content(latex_content: str) -> str:
    """
    Process LaTeX content to convert common environments and commands to HTML
    This is a basic implementation that handles some common LaTeX elements
    """
    # Extract content between \begin{document} and \end{document}
    document_match = re.search(r'\\begin\{document\}(.*?)\\end\{document\}', latex_content, re.DOTALL)
    if document_match:
        content = document_match.group(1)
    else:
        content = latex_content
    
    # Skip to after \maketitle if it exists
    maketitle_match = re.search(r'\\maketitle(.*?)$', content, re.DOTALL)
    if maketitle_match:
        content = maketitle_match.group(1)
    
    # Process sections
    content = re.sub(r'\\section\{(.*?)\}', r'<h1>\1</h1>', content)
    content = re.sub(r'\\subsection\{(.*?)\}', r'<h2>\1</h2>', content)
    
    # Process environments
    content = process_environments(content, 'exercise', 'Exercise')
    content = process_environments(content, 'solution', 'Solution')
    content = process_environments(content, 'statement', 'Statement')
    
    # Process basic formatting
    content = re.sub(r'\\textbf\{(.*?)\}', r'<strong>\1</strong>', content)
    content = re.sub(r'\\textit\{(.*?)\}', r'<em>\1</em>', content)
    content = re.sub(r'\\newline', r'<br>', content)
    
    # Replace \\ with line breaks
    content = re.sub(r'\\\\', r'<br>', content)
    
    # Handle empty lines as paragraph breaks
    content = re.sub(r'\n\s*\n', r'</p><p>', content)
    
    # Wrap in paragraph tags if not already wrapped
    if not content.startswith('<p>'):
        content = '<p>' + content + '</p>'
    
    return content

def process_environments(content: str, env_name: str, display_name: str) -> str:
    """
    Process LaTeX environments and convert them to HTML divs
    """
    pattern = fr'\\begin\{{{env_name}\}}\{{(.*?)\}}(.*?)\\end\{{{env_name}\}}'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    # Keep track of replacements to make
    replacements = []
    
    for match in matches:
        number = match.group(1)
        env_content = match.group(2)
        replacement = f'<div class="{env_name}"><strong>{display_name} {number}.</strong>{env_content}</div>'
        replacements.append((match.group(0), replacement))
    
    # Apply replacements
    for old, new in replacements:
        content = content.replace(old, new)
    
    return content 