from fastapi import FastAPI
from enum import Enum
import src.inference_lab.model.tiny_gpt2 as tiny_gpt2 
from pydantic import BaseModel,ConfigDict
import uvicorn

app = FastAPI()

class promptFormat(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt:str 
    
    def __getitem__(self):
        return self.prompt
        

class modelName(str,Enum):
    gpt2_tiny="gpt2_tiny"
    
@app.get("/model_info/{model_name}")
async def model_info(model_name):
    if model_name==modelName.gpt2_tiny:
        conf_file=tiny_gpt2.tinyGpt2().config
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
    
@app.post("/generate")
async def send_prompt(
    prompt:promptFormat
    ):
    body=prompt.prompt
    model_output=tiny_gpt2.tinyGpt2().GenerateResult(prompt=body)
    return model_output

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)