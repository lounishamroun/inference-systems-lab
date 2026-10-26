from fastapi import FastAPI
from enum import Enum
from inference_lab.model.tiny_gpt2 import tinyGpt2
from pydantic import BaseModel,ConfigDict
import uvicorn
import time
from fastapi.responses import StreamingResponse
from schema import promptFormat

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
    assert isinstance(body,str)
    model_output=model.GenerateResult(input_prompt=body)
    for word in model_output:
        yield(word)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)