import os
import subprocess

def clean_tmp_files(dir_path, base_name):
    # Ensure dir_path is a valid directory before proceeding
    if not dir_path or not os.path.isdir(dir_path):
        print(f"Warning: clean_tmp_files received invalid or non-existent directory path: '{dir_path}'")
        return
    
    print(f"Cleaning files starting with '{base_name}' in directory '{os.path.abspath(dir_path)}'")
    for file_name in os.listdir(dir_path):
        if file_name.startswith(base_name): 
            file_to_remove = os.path.join(dir_path, file_name)
            try:
                if os.path.isfile(file_to_remove):
                    os.remove(file_to_remove)
                    print(f"Removed: {file_to_remove}")
                # else: # Optionally log if it's a directory or symlink etc.
                #     print(f"Skipped (not a file): {file_to_remove}")
            except OSError as e:
                print(f"Error removing file {file_to_remove}: {e}")

def render_latex_to_html(file_path):
    if not os.path.exists(file_path):
        return {"error": f"Input LaTeX file not found: {file_path}", "html": "", "css": ""}

    output_dir = "tmp_res"
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    
    make4ht_executable = "C:/texlive/2025/bin/windows/make4ht.exe"
    
    # Use mathjax to avoid generating math images, which speeds up conversion significantly
    # fastmathjax: turns off post-processing of math in MathJax mode for even faster compilation
    # NoFonts: don't use original font style (faster processing)
    command = [make4ht_executable, file_path, "mathjax,fastmathjax,NoFonts", "-d", output_dir]

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False, 
            encoding='utf-8'
        )
        
    except Exception as e:
        error_msg = f"An unexpected error occurred during LaTeX to HTML conversion: {str(e)}"
        print(error_msg)

    
    html_file_name = f"{base_name}.html"
    html_file_path = os.path.join(output_dir, html_file_name)

    css_file_name = f"{base_name}.css" # make4ht có thể tạo các tên file CSS khác nhau
    css_file_path = os.path.join(output_dir, css_file_name)
    
    html_content = ""
    css_content = ""

    if os.path.exists(html_file_path):
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    else:
        return {"error": f"make4ht ran, but output HTML file '{html_file_name}' not found in '{output_dir}'"}

    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            css_content = f.read()
    else:
        print(f"Info: CSS file '{css_file_name}' not found in '{output_dir}'. CSS might be inlined or not present.")

    clean_tmp_files(output_dir, base_name) 
    clean_tmp_files(".", base_name) # Using "." for current directory

    return {"html": html_content, "css": css_content}
    

if __name__ == "__main__":
    file_path = "tmp_data/try.tex"
    print(render_latex_to_html(file_path))
