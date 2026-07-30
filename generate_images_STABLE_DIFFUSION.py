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
pipe = pipe.to("cuda") #move the model to GPU for faster inference

#The text prompt
prompt = "a cinematic photograph of a lone astronaut discovering a glowing alien artifact on a distant planet, cinematic lighting, 8k, ultra-detailed, trending on artstation"
image = pipe(prompt).images[0]
#save the image
image.save("astronaut_artifact.png")
