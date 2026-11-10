from typing import Any

from pydantic import BaseModel,ConfigDict


class Problem(BaseModel):
    error_message:str 
    status_code:int

class promptFormat(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt:str
    
    def __getitem__(self):
        return self.prompt
        