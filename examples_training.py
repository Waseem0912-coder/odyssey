#!/usr/bin/env python3
"""
Example: Training with GUI-Odyssey DPO Pairs using Hugging Face TRL

This script demonstrates how to use the generated odyssey_dpo_pairs.jsonl
for training with the DPOTrainer from the TRL library.

Model: Google Gemma 3N 4B (Instruct)
  - Efficient 4B parameter instruction-following model
  - Optimal for consumer GPUs (8GB VRAM with quantization)
  - Fast inference with maintained quality

Note: This is a template example showing the integration points.
Actual training requires:
  - NVIDIA GPU with at least 8GB VRAM (16GB recommended)
  - TRL, PEFT, and BitsAndBytes libraries
  - 5-10 minutes per epoch on typical hardware
"""

from datasets import load_dataset
from dataclasses import dataclass

@dataclass
class DPOTrainerConfig:
    """Configuration for DPO training with GUI-Odyssey data using Gemma 3N 4B."""
    
    # Dataset
    dpo_dataset_path: str = "odyssey_dpo_pairs.jsonl"
    split_ratio: float = 0.9  # 90% train, 10% eval
    
    # Model (Gemma 3N 4B Instruct)
    base_model_id: str = "google/gemma-3-4b-instruct"
    ref_model_id: str = None  # Auto-copy base_model if None
    use_4bit_quantization: bool = True  # Essential for 4B model on consumer GPUs
    
    # Training hyperparameters (optimized for 4B model)
    output_dir: str = "odyssey_dpo_checkpoints_gemma3n"
    num_train_epochs: int = 3  # Increased for smaller model
    per_device_train_batch_size: int = 2  # Reduced for 4B model (8GB VRAM)
    per_device_eval_batch_size: int = 4
    learning_rate: float = 1e-4  # Lower LR for smaller model
    beta: float = 0.1  # DPO β parameter (controls divergence from reference)
    max_prompt_length: int = 512
    max_completion_length: int = 128
    
    # Optimization
    gradient_accumulation_steps: int = 4  # Increased to compensate for lower batch size
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # LoRA configuration (for efficient fine-tuning)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = None  # Will be set to Gemma-specific modules
    
    # Logging
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 50


def load_dpo_dataset(dataset_path: str, split_ratio: float = 0.9):
    """
    Load and prepare DPO dataset from JSONL file.
    
    Args:
        dataset_path: Path to odyssey_dpo_pairs.jsonl
        split_ratio: Fraction of data to use for training (rest for evaluation)
    
    Returns:
        dict: {"train": Dataset, "eval": Dataset}
    """
    # Load from JSONL
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=1-split_ratio)
    
    return {
        "train": split_dataset["train"],
        "eval": split_dataset["test"]
    }


def format_dpo_dataset(dataset):
    """
    Format dataset for DPOTrainer compatibility.
    
    DPOTrainer expects:
    - prompt: str (instruction/context)
    - chosen: str (preferred response)
    - rejected: str (dispreferred response)
    
    Our dataset already has this format!
    """
    def format_example(example):
        # Extract image token if present
        prompt = example["prompt"]
        
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    
    return dataset.map(format_example, remove_columns=["prompt", "chosen", "rejected"])


def train_dpo_example(config: DPOTrainerConfig):
    """
    Example training script using DPOTrainer with Gemma 3N 4B.
    
    To run this, install required packages:
      uv pip install trl peft bitsandbytes
    
    Requires:
      - NVIDIA GPU with at least 8GB VRAM
      - CUDA 12.1+ (for BitsAndBytes quantization)
    
    Then call:
      python examples_training.py
    """
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import DPOTrainer, DPOConfig
    except ImportError:
        print("Error: Required packages not installed. Install with:")
        print("  uv pip install trl peft bitsandbytes")
        return
    
    # ========== Load Dataset ==========
    print(f"Loading DPO dataset from {config.dpo_dataset_path}...")
    datasets = load_dpo_dataset(
        config.dpo_dataset_path,
        split_ratio=config.split_ratio
    )
    
    train_dataset = format_dpo_dataset(datasets["train"])
    eval_dataset = format_dpo_dataset(datasets["eval"])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(eval_dataset)}")
    
    # ========== Load Models ==========
    print(f"Loading base model: {config.base_model_id}")
    
    # 4-bit quantization for efficient memory usage on consumer GPUs
    if config.use_4bit_quantization:
        print("  Using 4-bit quantization for memory efficiency...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2",  # Faster attention for Gemma
        )
        # Prepare for training with quantization
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    
    if config.ref_model_id:
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.ref_model_id,
            torch_dtype="auto",
            device_map="auto",
        )
    else:
        ref_model = None  # DPOTrainer will auto-copy
    
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ========== LoRA Configuration (Essential for 4B model) ==========
    # LoRA dramatically reduces memory and computational requirements
    # Gemma-3 has: o_proj, up_proj, down_proj, q_proj, v_proj, k_proj, gate_proj
    target_modules = [
        "q_proj",      # Query projection
        "v_proj",      # Value projection
        "k_proj",      # Key projection
        "o_proj",      # Output projection
        "gate_proj",   # Gating (specific to Gemma architecture)
        "up_proj",     # Intermediate layer up-projection
        "down_proj",   # Intermediate layer down-projection
    ]
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=["lm_head"],  # Save output layer separately
    )
    
    print(f"Applying LoRA (r={config.lora_rank}, α={config.lora_alpha})...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # ========== DPO Training Configuration ==========
    training_args = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        beta=config.beta,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    
    # ========== Initialize Trainer ==========
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # ========== Train ==========
    print("Starting DPO training...")
    trainer.train()
    
    # ========== Save ==========
    trainer.model.save_pretrained(f"{config.output_dir}/final_model")
    tokenizer.save_pretrained(f"{config.output_dir}/final_model")
    
    print(f"Training complete. Model saved to {config.output_dir}/final_model")


if __name__ == "__main__":
    # Example: Print expected dataset format
    print("=" * 70)
    print("GUI-Odyssey DPO Training Example")
    print("=" * 70)
    
    # Load and inspect dataset
    try:
        datasets = load_dpo_dataset("odyssey_dpo_pairs.jsonl", split_ratio=0.9)
        
        print(f"\n✓ Dataset loaded successfully")
        print(f"  Train samples: {len(datasets['train'])}")
        print(f"  Eval samples: {len(datasets['eval'])}")
        
        # Show example
        example = datasets["train"][0]
        print(f"\nExample DPO pair:")
        print(f"  Prompt: {example['prompt'][:80]}...")
        print(f"  Chosen: {example['chosen']}")
        print(f"  Rejected: {example['rejected']}")
        
        print("\n" + "=" * 70)
        print("To train with Gemma 3N 4B using TRL:")
        print("  1. Install: uv pip install trl peft bitsandbytes")
        print("  2. Ensure NVIDIA GPU with 8GB+ VRAM and CUDA 12.1+")
        print("  3. Uncomment train_dpo_example() call below")
        print("  4. Review DPOTrainerConfig (optimized for 4B model)")
        print("  5. Run: python examples_training.py")
        print("\nEstimated training time:")
        print("  - 1 epoch: ~3-5 minutes (with 10k pairs)")
        print("  - 3 epochs: ~10-15 minutes (default config)")
        print("=" * 70)
        
    except FileNotFoundError:
        print("✗ Dataset file not found. Run:")
        print("  uv run python src/odyssey_dpo_generator.py")
        print("  to generate odyssey_dpo_pairs.jsonl first.")
    
    # Uncomment below to start training (requires TRL + BitsAndBytes installation)
    # config = DPOTrainerConfig()
    # train_dpo_example(config)
    
    # Quick reference: Available models
    print("\nAvailable alternatives (same config optimizations apply):")
    print("  - google/gemma-3-4b-instruct (recommended - used in this example)")
    print("  - google/gemma-2-2b-it")
    print("  - meta-llama/Llama-2-7b-chat-hf")
    print("  - TinyLlama/TinyLlama-1.1B-Chat-v1.0")
