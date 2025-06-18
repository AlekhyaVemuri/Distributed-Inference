import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16",
    device_map="balanced",
    max_memory={0: "32GB", 1: "32GB", 2: "32GB", 3: "32GB", 4: "32GB", 5: "32GB", 6: "32GB", 7: "32GB"},
)
image = load_image(
     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd-docstring-example.jpeg"
)
image = image.resize((1024, 576))

frames = pipe(image, decode_chunk_size=2, generator=torch.Generator("xpu").manual_seed(0), num_frames=25).frames[0]
# frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
export_to_video(frames, "generated.mp4", fps=7)