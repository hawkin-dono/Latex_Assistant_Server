# LaTeX Server

A FastAPI-based server that provides LaTeX processing capabilities, including image-to-LaTeX conversion and LaTeX-to-HTML rendering.

## Features

- Convert images containing mathematical expressions to LaTeX code
- Render LaTeX documents to HTML
- Knowledge graph-based reasoning chatbot
- RESTful API interface
- Support for Together AI integration

## Prerequisites

Before running the server, you need to install:

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. LaTeX and make4ht Installation

You need a TeX distribution such as TeX Live or MiKTeX. It must include `tex4ht` system, Noto fonts and `texlua` script. All modern distributions include it.

#### For Unix systems:
Run these commands:
```bash
make
make install
```

`make4ht` is installed to `/usr/local/bin` directory by default. The directory can be changed by passing it's location to the `BIN_DIR` variable:

```bash
make install BIN_DIR=~/.local/bin/
```

#### For Windows:
See a [guide by Volker Gottwald](https://d800fotos.wordpress.com/2015/01/19/create-e-books-from-latex-tex-files-ebook-aus-latex-tex-dateien-erstellen/) on how to install `make4ht` and `tex4ebook`. 

Create a batch file for `make4ht` somewhere in the `path`:
```batch
texlua "C:\full\path\to\make4ht" %*
```

You can find directories in the path with:
```batch
path
```
command, or you can create new directory and [add it to the path](http://stackoverflow.com/questions/9546324/adding-directory-to-path-environment-variable-in-windows).

Note for `MiKTeX` users: you may need to create `texmf` directory first. See [this answer on TeX.sx](http://tex.stackexchange.com/questions/69483/create-a-local-texmf-tree-in-miktex).

## Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/hawkin-dono/Latex_Assistant_Server.git
cd latex-server
```

2. Create a `.env` file in the project root with your configuration:
```env
TOGETHER_API_KEY=your_api_key_here
TOGETHER_MODEL_NAME=your_model_name_here
```

## Running the Server

Start the server using uvicorn:

```bash
uvicorn api:app --reload
```

The server will be available at `http://localhost:8000`

## API Endpoints

### 1. Image to LaTeX Conversion
- **Endpoint**: `/image_to_latex`
- **Method**: POST
- **Input**: JSON with format:
  ```json
  {
    "image": "Base64 encoded image string"
  }
  ```
- **Output**: JSON with format:
  ```json
  {
    "latex": "Generated LaTeX code"
  }
  ```
- **Error Response**:
  ```json
  {
    "error": "Error message"
  }
  ```

### 2. LaTeX Rendering
- **Endpoint**: `/render_latex`
- **Method**: POST
- **Input**: JSON with format:
  ```json
  {
    "latex": "Your LaTeX content string"
  }
  ```
- **Output**: JSON with format:
  ```json
  {
    "html": "Rendered HTML content",
    "css": "Associated CSS styles"
  }
  ```
- **Error Response**: 
  ```json
  {
    "error": "Error message",
    "saved_latex_path": "Path to temporary LaTeX file (if applicable)"
  }
  ```

### 3. Chatbot Interface
- **Endpoint**: `/chatbot`
- **Method**: POST
- **Input**: JSON with format:
  ```json
  {
    "message": "Your message to the chatbot"
  }
  ```
- **Output**: JSON with format:
  ```json
  {
    "response": "Model's response"
  }
  ```
- **Error Response**:
  ```json
  {
    "error": "Error message"
  }
  ```

### 4. Dynamic Reasoning
- **Endpoint**: `/dynamic_reasoning`
- **Method**: POST
- **Input**: JSON with format:
  ```json
  {
    "message": "Your message for reasoning"
  }
  ```
- **Output**: JSON with format:
  ```json
  {
    "response": "Reasoned response from the model"
  }
  ```
- **Error Response**:
  ```json
  {
    "error": "Error message"
  }
  ```

## Project Structure

- `api.py`: Main FastAPI application and endpoints
- `latex_handler.py`: LaTeX processing utilities
- `image_to_latex_model.py`: Image to LaTeX conversion model
- `dynamic_reasoning_solver.py`: Knowledge graph reasoning system
- `config.py`: Configuration management
- `together_api_model.py`: Together AI integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

Copyright (c) 2024 Latex Assistant Server

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
