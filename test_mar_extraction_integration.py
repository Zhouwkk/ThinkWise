#!/usr/bin/env python3
"""
Test MAR extraction integration in training pipeline.
This script simulates the data flow without running full training.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from verl.protocol import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.config import ActorConfig

def test_mar_extraction():
    print("="*70)
    print("Testing MAR Extraction Integration")
    print("="*70)
    
    # 1. Load model
    model_path = "/data/zhouwenkang/models/Qwen2.5-VL-3B-Instruct"
    print(f"\n1. Loading model from {model_path}...")
    
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation='eager'  # Required for attention extraction
    )
    model.eval()
    print("  Model loaded successfully")
    
    # 2. Create actor with MAR extraction enabled
    print("\n2. Creating DataParallelPPOActor with extract_mar=True...")
    config = ActorConfig()
    config.extract_mar = True
    config.padding_free = False  # Use standard mode for testing
    config.micro_batch_size_per_device_for_experience = 1
    
    actor = DataParallelPPOActor(config=config, actor_module=model)
    print("  Actor created successfully")
    
    # 3. Prepare test data
    print("\n3. Preparing test data...")
    from PIL import Image
    import numpy as np
    
    # Create a dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Create a test prompt
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "What is 2+2?"}
        ]},
        {"role": "assistant", "content": "Let me think about this. 2+2 equals 4."}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=[dummy_image], padding=True, return_tensors="pt")
    
    # Move to CUDA
    inputs_cuda = {k: v.to('cuda') for k, v in inputs.items()}
    input_ids = inputs_cuda['input_ids']
    attention_mask = inputs_cuda['attention_mask']
    position_ids = torch.arange(input_ids.shape[1], device='cuda').unsqueeze(0)
    
    # Create response tensor (simulate rollout output)
    response_length = 10
    responses = input_ids[:, -response_length:]
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Response length: {response_length}")
    
    # Check for vision tokens
    vision_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    vision_mask = (input_ids[0] == vision_token_id)
    num_vision_tokens = vision_mask.sum().item()
    print(f"  Vision token ID: {vision_token_id}")
    print(f"  Number of vision tokens: {num_vision_tokens}")
    
    # 4. Test _forward_micro_batch
    print("\n4. Testing _forward_micro_batch...")
    micro_batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
    }
    
    with torch.no_grad():
        result = actor._forward_micro_batch(micro_batch, temperature=1.0)
    
    if isinstance(result, tuple):
        log_probs, mar_value = result
        print(f"  ✓ Returned tuple (log_probs, mar_value)")
        print(f"  Log probs shape: {log_probs.shape}")
        print(f"  MAR value: {mar_value}")
        
        if mar_value is not None and mar_value > 0:
            print(f"  ✓ MAR extraction successful!")
        else:
            print(f"  ✗ MAR value is None or zero")
    else:
        print(f"  ✗ Expected tuple, got {type(result)}")
        return False
    
    # 5. Test compute_log_prob (full pipeline)
    print("\n5. Testing compute_log_prob (full pipeline)...")
    data = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
        },
        meta_info={"temperature": 1.0}
    )
    
    with torch.no_grad():
        output = actor.compute_log_prob(data)
    
    if isinstance(output, tuple):
        log_probs, mar_list = output
        print(f"  ✓ Returned tuple (log_probs, mar_list)")
        print(f"  Log probs shape: {log_probs.shape}")
        print(f"  MAR list length: {len(mar_list)}")
        print(f"  MAR values: {mar_list}")
        
        if mar_list and all(m >= 0 for m in mar_list):
            print(f"  ✓ MAR list extraction successful!")
        else:
            print(f"  ✗ MAR list is empty or contains invalid values")
    else:
        print(f"  ✗ Expected tuple, got {type(output)}")
        return False
    
    print("\n" + "="*70)
    print("✓ All tests passed! MAR extraction is working correctly.")
    print("="*70)
    return True


if __name__ == "__main__":
    import os
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    
    success = test_mar_extraction()
    exit(0 if success else 1)
