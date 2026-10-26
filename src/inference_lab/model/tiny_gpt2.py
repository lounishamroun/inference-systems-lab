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
        GPT2WeightLoader(reference_model=self.reference_model,custom_model=self.model,single_block=None).transfer_all()
        print('weights transferred')
    
    @torch.inference_mode()
    def GenerateResult(self,input_prompt,max_new_tokens=10):
       
        self.model.to(device=self.device)
        
        
        generated_ids=self.tokenize(input_prompt)
        generated_ids=generated_ids.to(self.device)
        
        for _ in range(max_new_tokens):
            
            #model output 
            # shape => [B,T,vocab_size]
            logits = self.model(generated_ids) 
            
            #corresponding to the latest sequence position
            # shape => [B,1,vocab_size]
            next_token_logits = logits[:, -1, :] 
            
            
            argmax_vocab_index = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )
            
            next_token_id=argmax_vocab_index

            #generated_ids.shape => [B=1, T] | next_token_id.shape => [B=1,1] 
            generated_ids = torch.cat(
                [generated_ids, next_token_id],
                dim=-1,
            )
            
        return self.TokenToWord(generated_ids)
        
