from fastapi import FastAPI
from enum import Enum
from model.tiny_gpt2 import tinyGpt2
from pydantic import BaseModel,ConfigDict
from src.engine.inference_engine import inferenceEngine
import uvicorn
import time
from fastapi.responses import StreamingResponse
from schema import promptFormat,Problem
from src.engine.tokenizer import tokenizer
import torch 


app = FastAPI()


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
    model_name=prompt.model_name
    assert isinstance(body,str),f"Expected type `str`, got {type(body)}"
    assert isinstance(model_name,str),f"Expected type `str`, got {type(body)}"
    
    if len(body.strip())==0 or len(body.strip())>200:
        response=Problem(error_message="Prompt shouldn't be empty or having a length > 200 characters",status_code=415)
        return response.error_message,response.status_code
    
    if len(body.strip())==0 or len(body.strip())>50:
        response=Problem(error_message="Model name shouldn't be empty or having a length > 50 characters",status_code=415)
        return response.error_message,response.status_code

    """TOKENIZATION"""
    tokenized_body=tokenizer().tokenize(text=body)
    assert isinstance(tokenized_body,torch.Tensor),f"Expected type `torch.Tensor`, got {type(tokenized_body)}"
    """CALL TO MODEL"""
    #TO DO await model call or enginecore call
    inference_engine=inferenceEngine(tokenized_body=tokenized_body,model_name=model_name)
    model_output=inference_engine.generate() #type torch.Tensor
    
    """DETOKENIZATION"""
    
    response=tokenizer().TokenToWord(model_output)[0]
    
    assert isinstance(response,str),f"Expected type `str`, got {type(response)}"
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)