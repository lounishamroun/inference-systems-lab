import pytest
import json
from tiny_transformer.transfer_model_param import GPT2WeightLoader
from tiny_transformer.block import TinyModel
from model.tiny_gpt2 import tinyGpt2
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="session")
def reference_model(device):
    reference_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    reference_model = reference_model.to(device)
    reference_model.eval()
    return reference_model

@pytest.fixture(scope="session")
def custom_config():
    conf=tinyGpt2().config
    return conf

@pytest.fixture(scope="session")
def custom_model():
    custom_model=tinyGpt2()
    custom_model.loadWeights()
    custom_model.model.eval()
    return custom_model
    
@pytest.fixture(scope="session")
def json_prompt():
    payload={"prompt": "My favorite italian food is"}
    json_payload=json.dumps(payload)
    return json_payload

@pytest.fixture(scope="session")
def raw_prompt():
    raw_prompt="My favorite italian food is"
    return raw_prompt

@pytest.fixture(scope="session")
def tokenized_prompt(raw_prompt,device):
    tokenizer = AutoTokenizer.from_pretrained(
        "openai-community/gpt2"
    )
    ids = tokenizer(
        raw_prompt,
        return_tensors="pt",
    )["input_ids"]
    
    return ids.to(device)

@pytest.fixture(scope="session")
def n_turns():
    n_turns=6
    return n_turns