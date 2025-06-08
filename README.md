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

## Models

### 1. Image-to-LaTeX Model

The Image-to-LaTeX model converts mathematical expressions in images to LaTeX code. It combines MobileNetV3 for image feature extraction with a Transformer decoder for LaTeX generation.

#### Performance Metrics

##### Normal Test Results
- **BLEU**: 0.8003 ± 0.0144  
- **NED**: 0.8826 ± 0.0099  
- **Accuracy**: 0.5805 ± 0.0244  
- **Samples**: 1000  
- **Time**: 1835.31s  

##### Handwritten Test Results
- **BLEU**: 0.6860 ± 0.0134  
- **NED**: 0.8181 ± 0.0096  
- **Accuracy**: 0.4489 ± 0.0223  
- **Samples**: 1000  
- **Time**: 1764.92s  

#### Dataset
The model is trained on 3.4 million image-text pairs, including:
- Handwritten mathematical expressions (200,330 examples)
- Printed mathematical expressions (3,237,250 examples)

Dataset available at: [27GB Fusion Image-to-LaTeX Dataset](https://huggingface.co/datasets/hoang-quoc-trung/fusion-image-to-latex-datasets)

##### Data Sources
- **Printed expressions**: Im2latex-100k, I2L-140K Normalized, Im2latex-90k Normalized, Im2latex-170k, Im2latex-230k, latexformulas datasets
- **Handwritten expressions**: CROHME dataset, Aida Calculus Math Handwriting Recognition Dataset, Handwritten Mathematical Expression Convert LaTeX

#### References
1. [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. [Image-to-Markup Generation with Coarse-to-Fine Attention](https://arxiv.org/abs/1609.04938v2)
4. [Mobile-net-V3](https://arxiv.org/pdf/1905.02244)

### 2. Knowledge Graph-based Reasoning Chatbot

The reasoning chatbot uses a sophisticated Knowledge Graph (KG) architecture for understanding and answering queries. The system combines vector search, graph traversal, and LLM-based reasoning.

#### System Architecture

1. **Data Processing & KG Construction**
   - Text splitting into semantic chunks
   - Entity and relation extraction using LLM
   - Knowledge Graph construction with NetworkX
   - Vector embeddings and FAISS indexing
   - Document store management

2. **Reasoning Components**
   - **Query Understanding**: Analyzes user questions
   - **Tool Selection & Execution**:
     - Vector database search
     - Knowledge Graph querying
   - **Iterative Reasoning**: Uses ReAct pattern (Reasoning + Action)
   - **Response Generation**: LLM-based answer synthesis

3. **LLM Integration**
   - Supports both local models and API-based LLMs
   - Uses Together AI for enhanced performance
   - Customizable system prompts

#### Features
- Dynamic reasoning process
- Multi-step inference
- Context-aware responses
- Transparent reasoning display
- Configurable parameters (top_k, max_steps)

For more detailed information about the reasoning chatbot, visit [KAG_VNUBot](https://github.com/hoanghelloworld/KAG_VNUBot).

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
