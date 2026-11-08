from fastapi import FastAPI
from enum import Enum
from model.tiny_gpt2 import tinyGpt2
from pydantic import BaseModel,ConfigDict
import uvicorn
import time
from fastapi.responses import StreamingResponse
from schema import promptFormat
from src.engine.tokenizer import tokenizer
import torch 


app = FastAPI()
model=tinyGpt2()
model.loadWeights()
model.model.eval()
        

class modelName(str,Enum):
    gpt2_tiny="gpt2_tiny"
    
@app.get("/model_info/{model_name}")
async def model_info(model_name):
    if model_name==modelName.gpt2_tiny:
        conf_file=tinyGpt2().config
        return{           
            "model_dimension": conf_file.d_model,
            "num_heads": conf_file.n_heads,
            "num_layers": conf_file.num_layers,
            "vocab_size": conf_file.vocab_size,
            "context_length": conf_file.context_length,            
        }
        
""" 
Receive str prompt  
"""
    
@app.post("/generate",response_class=StreamingResponse)
async def send_prompt(
    prompt:promptFormat
    ):
    body=prompt.prompt #type str
    assert isinstance(body,str),f"Expected type `str`, got {type(body)}"
    tokenized_body=tokenizer().tokenize(text=body)
    assert isinstance(tokenized_body,torch.Tensor),f"Expected type `torch.Tensor`, got {type(tokenized_body)}"
    model_output=model.GenerateResult(input_prompt=tokenized_body) #type torch.Tensor
    response=tokenizer().TokenToWord(model_output)[0]
    assert isinstance(response,str),f"Expected type `str`, got {type(response)}"
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)