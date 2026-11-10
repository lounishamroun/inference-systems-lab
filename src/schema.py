from typing import Any

from pydantic import BaseModel,ConfigDict,Field


class Problem(BaseModel):
    error_message:str 
    status_code:int

class promptFormat(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt:str
    model_name:str = Field(default="gpt2_tiny")
    
    def __getitem__(self):
        return self.prompt
        