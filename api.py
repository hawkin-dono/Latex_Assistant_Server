# LaTeX rendering requires:
# 1. TeX Live or MiKTeX: https://tug.org/texlive/ or https://miktex.org/
# 2. pdf2htmlEX (optional but recommended): https://github.com/pdf2htmlEX/pdf2htmlEX
#
# For Ubuntu/Debian: 
#   sudo apt-get install texlive-full
#   sudo apt-get install pdf2htmlex
#
# For Windows:
#   Install MiKTeX from https://miktex.org/download
#   Install pdf2htmlEX from https://github.com/pdf2htmlEX/pdf2htmlEX/releases
#
# For macOS:
#   brew install --cask mactex
#   brew install pdf2htmlex

from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO
from PIL import Image, ImageEnhance
import json
import os
import sys
import math
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

from image_to_latex_model import Im2LatexModel, load_vocab, decode_prediction
from latex_renderer import render_latex_to_html

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((150, 700)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.where(x > 0.5, 1.0, 0.0)),
    ])
    
    image = image.convert('L')
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = transform(image)
    
    if torch.mean(image) > 0.5:
        image = 1 - image
        
    return image

try:
    vocab_path = "model/tokenizer.json"
    vocab = load_vocab(vocab_path)
    reverse_vocab = {str(idx): word for word, idx in vocab.items()}
        
    model = Im2LatexModel(
            embed_size=256,
            vocab_size=len(vocab),
            num_layers=6,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1
        )
        
    checkpoint_path = "model/best_model.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print("Model loaded successfully")
    model.eval()
    
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.post("/image_to_latex")
async def image_to_latex(data: Dict[str, str] = Body(...)):
    base64_image = data.get("image")
    
    if not base64_image:
        return JSONResponse(content={"error": "No image data provided"}, status_code=400)
    
    # print(f"Received base64 image data: {base64_image[:100]}...")
    
    try:
        if model is None:
            return JSONResponse(content={"latex": "done", "message": "Model not loaded - this is a placeholder response"})
            
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))
        
        tensor = preprocess_image(image)
        
        start_token = vocab.get("<START>", 1)
        end_token = vocab.get("<END>", 2)
        tokens = model.generate(tensor, start_token, end_token)
        
        latex = decode_prediction(tokens, reverse_vocab)
        
        return JSONResponse(content={"latex": latex})
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(content={"latex": latex, "error": str(e)})

@app.post("/render_latex")
async def render_latex(data: Dict[str, str] = Body(...)):
    """
    Render LaTeX content to HTML
    
    The endpoint expects a JSON with the following format:
    {
        "latex": "\\documentclass{article}\\begin{document}Hello World\\end{document}"
    }
    
    Returns HTML content of the rendered LaTeX
    """
    latex_content = data.get("latex")
    
    if not latex_content:
        return JSONResponse(content={"error": "No LaTeX content provided"}, status_code=400)
    
    try:
        # If the latex content doesn't have document class and begin/end document,
        # wrap it in a basic document structure
        if "\\documentclass" not in latex_content:
            latex_content = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{amssymb}}
\\begin{{document}}
{latex_content}
\\end{{document}}
"""
        
        result = render_latex_to_html(latex_content)
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Error rendering LaTeX: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
