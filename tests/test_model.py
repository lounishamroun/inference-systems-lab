import pytest
from model.tiny_gpt2 import tinyGpt2
import torch


@torch.inference_mode()
def test_non_null(raw_prompt,custom_model,n_turns):
    assert raw_prompt and raw_prompt is not None
    model_output=custom_model.GenerateResult(input_prompt=raw_prompt,n_turns=n_turns)
    assert model_output is not None


def test_matching_ref_model(custom_model,reference_model,raw_prompt,n_turns,tokenized_prompt):
    assert raw_prompt and raw_prompt is not None

    expected = reference_model(
        input_ids=tokenized_prompt,
        use_cache=False,
    ).logits
    

    actual =custom_model.GenerateResult(input_prompt=raw_prompt,n_turns=n_turns,debug_mode=True)

    diff = (actual - expected).abs()

    print("max abs diff:", diff.max().item())
    print("mean abs diff:", diff.mean().item())

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )
    
