
from fastapi import FastAPI
from enum import Enum
from model.tiny_gpt2 import tinyGpt2
from pydantic import BaseModel,ConfigDict
import uvicorn
import time
from fastapi.responses import StreamingResponse
from schema import promptFormat,Problem
from src.engine.tokenizer import tokenizer
import torch 



class inferenceEngine():
    def __init__(self,tokenized_body,model_name) -> None:
        self.tokenized_body=tokenized_body
        self.model_name=model_name
    
    def generate(self,model_name="gpt2_tiny"): #providing a default model to call
        assert model_name is not None,"No model name provided, please specify"
        model_response=""
        if model_name=="gpt2_tiny":
            model=tinyGpt2()
            model.loadWeights()
            model.model.eval()
            model_response=model.GenerateResult(input_prompt=self.tokenized_body)
        
        return model_response
            
        
        
        
    