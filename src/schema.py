from typing import Any

from pydantic import BaseModel,ConfigDict


class Problem(BaseModel):
    def __init__(self,error_message,status_code):
        self.error_message:str=error_message 
        self.status_code:int=status_code

class promptFormat(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt:str
    
    def __getitem__(self):
        return self.prompt
        