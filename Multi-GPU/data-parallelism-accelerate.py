import torch
from accelerate import PartialState
from diffusers import StableDiffusion3Pipeline

pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16,
        use_safetensors=True
)

distributed_state = PartialState()
pipeline.to(distributed_state.device)

with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")