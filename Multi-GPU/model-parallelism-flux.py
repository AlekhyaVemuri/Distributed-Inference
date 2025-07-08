import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev",
        device_map="balanced",
        # max_memory={0: "32GB", 1: "32GB"},
        torch_dtype=torch.bfloat16)

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("xpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")