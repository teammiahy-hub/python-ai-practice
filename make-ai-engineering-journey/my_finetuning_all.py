import torch


from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import pandas as pd

# Step1: DATA PREPARATION + STEP2 CONCATENATION FORMATTING DONE IN THIS STEP ALSO
print("DATA PREPARATION")
training_examples = [
    {
        "instruction": "Explain what RAG is",
        "output": "RAG (Retrieval Augmented Generation) is a technique that enhances LLM responses by first retrieving relevant documents from a knowledge base, then using that context to generate accurate, grounded answers."
    },
    {
        "instruction": "What is LoRA?", 
        "output": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adds small trainable matrices to existing model layers instead of updating all parameters, dramatically reducing memory requirements."
    },
    # Add 50-100 examples for a real project
]

def format_example(example):
        return f"""### Instruction:
                {example['instruction']}

                ### Response:
                {example['output']}"""

df = pd.DataFrame(training_examples)
df["text"] = df.apply(format_example,axis=1)
dataset = Dataset.from_pandas(df)

# STEP3 TOKENIZE THE DATA SET
model_name = "microsoft/phi-2"  # Fits on T4 GPU
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# STEP 4 LOAD THE MODEL FOR TRAINING
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
)
model = prepare_model_for_kbit_training(model)

# STEP5 CALL LORA FOR FINETUNING
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
print("This is what LoRA for: Adding trainable matrices to the model to train it finely")
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Training {100*trainable/total:.2f}% of parameters")

# FINAL STEP: TRAIN
training_args = SFTConfig(
    output_dir="./pythia-3gpp-lora",

    num_train_epochs=3,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=1e-4,

    logging_steps=1,

    save_strategy="epoch",

    report_to="none",

    use_cpu=True,

    bf16=False,

    fp16=False,
)
trainer = SFTTrainer(
        model = model,
        train_dataset = dataset,
        processing_class=tokenizer,
        args=training_args,
)

print("Start training...")
trainer.train()

# Optionally
# model.push_to_hub("teammiahy-hub/phi2-lora-ai-tutor")
# tokenizer.push_to_hub("teammiahy-hub/phi2-lora-ai-tutor")
# print("Model published to HuggingFace Hub!")