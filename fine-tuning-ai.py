import torch    
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
#The model we want to fin-tune
model_name = "gpt2"

#load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)

#set the padding token if it's not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# a prompt to test the model before fine-tuning
prompt = "The movie started with a captivating scene where that"

#tokenizze the prompt
inputs = tokenizer(prompt, return_tensors="pt")

#generate a completion using the model
# we are moving the model and inputs to the GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
inputs = inputs.to(device)

#generate text using the model

generate_ids = model.generate(inputs["input_ids"], max_length=50)
response = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
print("Model response before fine-tuning:", response)

# TRAINING SETUP
#load the IMDB dataset
dataset = load_dataset("stanfordnlp/imdb", split="train")

#filter for only positive reviews
positive_reviews = dataset.filter(lambda x: x["label"] == 1)

#to make this demo run quickly, let's just use a small subset of the data
small_subset = positive_reviews.select(range(500))

# we need to format our examples into a single text string for the SFTTrainer

def format_review(example):
    # for thise simple task, the text itself is our training data
    return {"text": f"Review: {example['text'] + " TL;DR: Positive review."}"}

formatted_dataset = small_subset.map(format_review)

print("Sample formatted review:", formatted_dataset[0]["text"])

#CONFIGURE AND APPLY LORA
from peft import LoraConfig, get_peft_model
#create lora configuration
lora_config = LoraConfig(
    r=8,  # rank of the low-rank matrices
    lora_alpha=16,  # scaling factor for the low-rank matrices
    target_modules=["c_attn"],  # target modules to apply LoRA to
    lora_dropout=0.1,  # dropout rate for LoRA layers
    bias="none",  # no bias in LoRA layers
    task_type="CAUSAL_LM"  # task type for causal language modeling
)

#wrap the base model with the PEFT model using LoRA
peft_model = get_peft_model(model, lora_config)

#let's see how many parameters we are actually training!
peft_model.print_trainable_parameters()

#SETUP THE TRAINING ARGUMENTS
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

#some safety properties
peft_model.config.use_cache = False
tokenizer.padding_side = "right"

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    peft_model.config.pad_token_id = peft_model.config.eos_token_id


#tokenize the entire dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        truncation=True, 
        padding="max_length",
        max_length=512
    )

tokenized_dataset = formatted_dataset.map(
    tokenize_function, 
    batched=True,
    remove_columns=formatted_dataset.column_names)


# the DataCollator handles batching and creating labels for us
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False #we are doing causal language modeling, not masked language modeling
)

#define the training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-imdb-finetune",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=50,
    fp16=True, #use mixed precission for speedup
    remove_unused_columns=False, #we need to keep all columns for the data collator
)

# create the Trainer instance
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# and now we can start training!
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete!")

#RERUN THE TEST PROMPT
#generate a completion using the fine-tuned model
print("Generating response after fine-tuning...")

#the trainer wraps the model, so we can use it directly
fine_tuned_model = trainer.model

#generate text using the fine-tuned model
generate_ids = fine_tuned_model.generate(inputs["input_ids"], max_length=50)
response = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
print("Model response after fine-tuning:", response)