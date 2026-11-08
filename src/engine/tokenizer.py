
from transformers import AutoTokenizer

class tokenizer():
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
                            "openai-community/gpt2"
                        )
    
    def tokenize(self,text):
        
        ids = self.tokenizer(
            text,
            return_tensors="pt",
        )["input_ids"]
        return ids
    
    def TokenToWord(self,token_id):
        word=self.tokenizer.decode(token_id)
        return word
