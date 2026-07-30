#make sure you have the openai library installed
#pip install openai
from openai import OpenAI
import os
# GOOD PRACTICE
# Set your OpenAI API key as an environment variable
# export OPENAI_API_KEY="your_api_key_here"
# Create an OpenAI client
client = OpenAI() # the client will automatically use the OPENAI_API_KEY environment variable

prompt = "An oil painting of a futuristic cityscape at sunset, in the style of Monet, with flying cars leaving light trails"

try:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        n=1,
        quality="high"
    )
    # Get the generated image URL
    image_url = response.data[0].url
    print(f"Generated image URL: {image_url}")
except Exception as e:
    print(f"Error generating image: {e}")

    
