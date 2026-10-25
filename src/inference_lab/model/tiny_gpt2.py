from tiny_transformer.block import TinyModel
from tiny_transformer.transfer_model_param import GPT2WeightLoader
from tiny_transformer.config import GPT2CustomConfig
from tiny_transformer.generation import greedy_generate
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch



class tinyGpt2():
    def __init__(self) -> None:
        self.device="cuda:0" if torch.cuda.is_available() else "cpu"
        self.config=GPT2CustomConfig()
        self.model=TinyModel(config=self.config)
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

    def loadWeights(self):
        self.reference_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        GPT2WeightLoader(reference_model=self.reference_model,custom_model=self.model,single_block=None)
    
    @torch.inference_mode()
    def GenerateResult(self,input_prompt,max_new_tokens=10):
       
        self.model.to(device=self.device)
        
        
        generated_ids =  self.tokenize(input_prompt)
        for _ in range(max_new_tokens):

            logits = self.model(generated_ids)
            next_token_logits = logits[:, -1, :] #corresponding to the latest sequence position
            
            highest_correlation_token = torch.argmax(
                next_token_logits,
                dim=1,
                keepdim=True,
            )
            
            next_token_logits=next_token_logits[next_token_logits!=highest_correlation_token]
            
            next_token_id = torch.argmax(
                            next_token_logits,
                            dim=-1,
                            keepdim=True,
                        )
            
            generated_ids = torch.cat(
                [generated_ids, next_token_id],
                dim=1,
            )
            
            yield self.TokenToWord(next_token_id)
        
