from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
import json
import pandas as pd

def prepare_training_data():
    # Load instruction pairs
    with open('data/instruction_pairs.json', 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Convert to dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    return dataset

def main():
    # Load model and tokenizer
    model_name = "google/gemma-2b"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir="./gemma-afrikaans",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch"
    )
    
    # Load and prepare dataset
    dataset = prepare_training_data()
    
    # Start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    trainer.train()
    
    # Save the model
    trainer.save_model("./gemma-afrikaans-final")

if __name__ == "__main__":
    main()