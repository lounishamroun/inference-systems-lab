from pydantic import BaseModel,ConfigDict


class promptFormat(BaseModel):
    model_config = ConfigDict(strict=True)
    prompt:str
    
    def __getitem__(self):
        return self.prompt
        