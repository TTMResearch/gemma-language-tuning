from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "google/gemma-2b"
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

class GemmaAfrikaans:
    def __init__(self, config: ModelConfig):
        self.config = config
        
        # Quantization config for 4-bit training
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=self.quant_config,
            device_map="auto"
        )
        
        # Set up LoRA configuration
        self.lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules or ["q_proj", "v_proj"],
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=config.task_type
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        
    def save_model(self, output_dir: str):
        """Save the model and tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)