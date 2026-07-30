import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

#Load the processor and model from the huggingface hub
# The processor preapares the data for the model (image preprocessing, tokenization, etc.)
# The model is the actual pre-trained network that will answer our question

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

print("Processor and model loaded successfully.")

#LEt's use an image from the web
image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
#Download the image and open it using PIL
raw_image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")

#define the question we want to ask about the image
question = "What is in the background in the image?"

#The processor combines the image and question into a format suitable for the model
inputs = processor(raw_image, question, return_tensors="pt")

print("Inputs prepared for the model.")

#Feed the inputs to the model to get the answer
outputs = model.generate(**inputs, max_new_tokens=20)

print("Model has generated an output.")
answer = processor.decode(outputs[0], skip_special_tokens=True)
print("\n-----------\n")
print("Question:", question)
print("Answer:", answer)
print("\n-----------\n")