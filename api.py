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
import torch
from io import BytesIO
from PIL import Image, ImageEnhance
import os
from torchvision import transforms
from together_api_model import get_together_response 
from dynamic_reasoning_solver import DynamicKAGSolver

from image_to_latex_model import Im2LatexModel, load_vocab, decode_prediction
from latex_handler import render_latex_to_html

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khai báo biến global solver
solver = None

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
    
    
def get_latex_tmp_path():
    # files = os.listdir("tmp_res")
    # max_cnt = 0
    # for file in files:
    #     try:
    #         file_name = file.split(".")[0]
    #         cnt = int(file_name.split("_")[1])
    #         max_cnt = max(max_cnt, cnt)
    #     except:
    #         pass
    # return os.path.join("tmp_res", f"tmp_{max_cnt + 1}.tex")
    return "tmp_res/tmp.tex"
    

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
    Render LaTeX content to HTML.
    Input JSON: {"latex": "LaTeX content string"}
    Output JSON: {"html": "HTML string", "css": "CSS string (optional)"} or {"error": "message"}
    """
    print(f"Received data: {data}")
    latex_content = data.get("latex")

    if not latex_content:
        return JSONResponse(
            content={"error": "No 'latex' field found in the request body."},
            status_code=400
        )

    tmp_file_path = get_latex_tmp_path()
    print(f"tmp_file_path: {tmp_file_path}")

    try:
        # Save the LaTeX content to a temporary file
        with open(tmp_file_path, "w", encoding="utf-8") as f:
            f.write(latex_content)
        
        
        rendered_output = render_latex_to_html(tmp_file_path)
        
        if isinstance(rendered_output, dict) and "html" in rendered_output:
             return JSONResponse(content=rendered_output)
        else:
            print(f"Warning: render_latex_to_html returned an unexpected format: {type(rendered_output)}")
            return JSONResponse(content={"html": str(rendered_output), "css": ""})

    except FileNotFoundError:
        print(f"Error: Temporary LaTeX file {tmp_file_path} not found after writing.")
        return JSONResponse(
            content={"error": "Failed to create or access the temporary LaTeX file."},
            status_code=500
        )
    except Exception as e:
        # Ghi log exception để debug
        print(f"Error processing LaTeX content for {tmp_file_path}: {e}")
        error_message = f"Failed to render LaTeX content: {str(e)}"
        return JSONResponse(
            content={"error": error_message, "saved_latex_path": tmp_file_path},
            status_code=500 # Internal Server Error
        )
    
@app.post("/chatbot")
def chatbot(data: Dict[str, str] = Body(...)):
    """
    Chat with the model.
    Input JSON: {"message": "User message"}
    Output JSON: {"response": "Model response"}
    """
    message = data.get("message")
    if not message:
        return JSONResponse(content={"error": "No message provided"}, status_code=400)
    
    response = get_together_response(message)
    return JSONResponse(content={"response": response})

@app.post("/dynamic_reasoning")
def dynamic_reasoning(data: Dict[str, str] = Body(...)):
    """
    Dynamic reasoning with the model.
    Input JSON: {"message": "User message"}
    Output JSON: {"response": "Model response"}
    """
    
    message = data.get("message")
    if not message:
        return JSONResponse(content={"error": "No message provided"}, status_code=400)
    
    global solver
    
    if solver is None:
        solver = DynamicKAGSolver()
    response = solver.solve(message)
    return JSONResponse(content={"response": response})
        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)