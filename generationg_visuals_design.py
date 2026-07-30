#YOU DO NEED A GPU-ENABLED SYSTEM FOR IMAGE GENERATION USING DIFFUSERS. 
# IF YOU ARE USING A CPU-ONLY SYSTEM, YOU WILL GET AN ERROR.
#make sure you have the needed libraries installed
#pip install torch diffusers transformers accelerate safetensors
import torch
from diffusers import StableDiffusionPipeline

# this will download the model weights the first time you run it
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float16
    )
pipe = pipe.to("cuda")

#1. style: minimalist vector art, clean lines, futuristic feel
#2. Colour palette: A professional and modern mix of deep blue (#0A2342)
#3. Mood: Innovative, clean, intelligent, and approachable
#4. Composistion: Often centered, with clear subjects and sometimes negative space for text overlays.

subject = "a smart robot brain processing data"
prompt_template = (
    f"Minimalist vector art design of {subject} with a "
    "futuristic feel, clean lines, and a professional color "
    "palette dominated by deep blue (#0A2342). "
    "The mood should convey innovation, intelligence, and approachability. "
    "The composition should be centered, with clear subjects and ample "
    "negative space for potential text overlays."
)

#using a negative prompt to avoid unwanted elements
negative_prompt = (
    "cluttered backgrounds, dark colors, chaotic compositions, "
    "unprofessional elements, overly complex designs"
)

image = pipe(
    prompt=prompt_template,
    negative_prompt=negative_prompt,
    num_inference_steps=50,
    guidance_scale=8.0,
).images[0]

image.save("minimalist_robot_brain.png")


