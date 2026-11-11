from tiny_transformer.block import TinyModel
from tiny_transformer.transfer_model_param import GPT2WeightLoader
from tiny_transformer.config import GPT2CustomConfig
from tiny_transformer.generation import greedy_generate
from transformers import AutoModelForCausalLM
import torch



class tinyGpt2():
    def __init__(self) -> None:
        self.device="cuda:0" if torch.cuda.is_available() else "cpu"
        self.config=GPT2CustomConfig()
        self.model=TinyModel(config=self.config)

    def loadWeights(self):
        self.reference_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        GPT2WeightLoader(reference_model=self.reference_model,custom_model=self.model,single_block=None).transfer_all()
        print('weights transferred')
    
    @torch.inference_mode()
    def GenerateResult(self,input_prompt,n_turns=10,debug_mode=False):
       
        self.model.to(device=self.device)
        assert isinstance(input_prompt,torch.Tensor),f'Got type: {type(input_prompt)}, expected type: torch.Tensor'
        generated_ids=input_prompt
        generated_ids=generated_ids.to(self.device)
        
        #model output 
        # shape => [B,T,vocab_size]
        init_logits = self.model(generated_ids) 
        
        for _ in range(n_turns):
            
            #corresponding to the latest sequence position
            # shape => [B,1,vocab_size]
            next_token_logits = init_logits[:, -1, :] 
            
            
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
            
            init_logits=self.model(generated_ids) 
        
        if debug_mode==True:
            return init_logits
            
        return generated_ids
