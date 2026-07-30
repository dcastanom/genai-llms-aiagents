from transformers import pipeline
# Initialize the text generation pipeline
# This is the easiest way to get started with a pretrained model
generator = pipeline('text-generation', model='distilgpt2')
# define the prompt
prompt = "The AI agent quickly analyzed the data stream and its core decision was to"
output = generator(
    prompt, 
    max_length=100, #stop at 100 tokens
    num_return_sequences=1, #generate one sequence
    do_sample=False, #disable sampling for greedy generation
    temperature=0.75, #set the temperature for sampling. control randomness of predictions. Lower values make the output more deterministic.
    top_k=50, #consider only the top 50 tokens likely to be generated at each step
    top_p=0.95 #consider only the top 95% probability mass , the nucleus sampling technique. This means that the model will only consider 
    #the most likely tokens that make up 95% of the probability distribution, which can help to reduce the risk of generating 
    # low-probability or nonsensical tokens.
    )
# Print prompt and the generated text
print("Prompt:")
print(prompt)
print("Generated text:")
print(output[0]['generated_text'])
