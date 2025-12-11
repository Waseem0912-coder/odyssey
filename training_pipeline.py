#!/usr/bin/env python3
"""
Two-Stage Training Pipeline for GUI-Odyssey Model

This script implements a complete training pipeline:

Stage 1: Supervised Fine-Tuning (SFT)
    - Goal: Teach the model the format and basic instruction following
    - Train on instruction -> action pairs from Odyssey dataset
    - Makes the model learn to output coordinates in correct format
    - Duration: 1-2 epochs
    - Result: Model outputs reasonable predictions (accuracy ~40-60%)

Stage 2: Direct Preference Optimization (DPO)
    - Goal: Teach the model to prefer correct over incorrect coordinates
    - Uses DPO pairs with correct (chosen) vs wrong (rejected) actions
    - Refines predictions without needing explicit reward model
    - Duration: 1-3 epochs
    - Result: Model prefers accurate coordinates (accuracy ~70-85%)

Architecture:
    Base Model: Google Gemma 3 4B Instruct
    Fine-tuning: LoRA (r=16, α=32) for efficiency
    Quantization: 4-bit (NF4) for consumer GPUs
    
Hardware Requirements:
    - GPU: 8GB VRAM minimum
    - CUDA: 12.1+
    - RAM: 16GB+ system RAM
    
Installation:
    uv pip install trl peft bitsandbytes datasets torch transformers
"""

import json
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datasets import Dataset, load_dataset


# ============================================================================
# STAGE 1: SUPERVISED FINE-TUNING (SFT) CONFIG
# ============================================================================

@dataclass
class SFTConfig:
    """Configuration for Supervised Fine-Tuning stage."""
    
    # Model
    base_model_id: str = "google/gemma-3-4b-instruct"
    use_4bit_quantization: bool = True
    
    # Dataset
    dataset_path: str = "odyssey_dpo_pairs.jsonl"
    split_ratio: float = 0.9  # 90% train, 10% eval
    max_samples: Optional[int] = None  # None = use all
    
    # Training
    output_dir: str = "checkpoints/sft_odyssey"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_seq_length: int = 1024
    
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Logging & Checkpointing
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 25
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    
    # Optional: Validation
    validation_split: float = 0.1


# ============================================================================
# STAGE 2: DPO TRAINING CONFIG
# ============================================================================

@dataclass
class DPOConfig:
    """Configuration for DPO training stage."""
    
    # Model (use SFT checkpoint)
    base_model_id: str = "checkpoints/sft_odyssey"  # Path to SFT checkpoint
    use_4bit_quantization: bool = True
    
    # Dataset
    dpo_dataset_path: str = "odyssey_dpo_pairs.jsonl"
    split_ratio: float = 0.9
    max_samples: Optional[int] = None
    
    # Training
    output_dir: str = "checkpoints/dpo_odyssey"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5  # Lower LR for refinement
    beta: float = 0.1  # DPO β parameter
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_prompt_length: int = 512
    max_completion_length: int = 128
    
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Logging
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 25
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])


# ============================================================================
# STAGE 1: SUPERVISED FINE-TUNING (SFT)
# ============================================================================

def prepare_sft_dataset(dataset_path: str, split_ratio: float = 0.9, max_samples: Optional[int] = None):
    """
    Prepare dataset for SFT training.
    
    Converts DPO pairs to SFT format:
    Input: "User: Tap on the home button.\nModel:"
    Output: "CLICK: (250, 400)"
    
    For SFT, we only use the "chosen" action (correct answer).
    """
    print(f"Loading SFT dataset from {dataset_path}...")
    
    # Load from JSONL
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # For SFT, we use the "chosen" (correct) actions
    def format_sft_example(example):
        return {
            "text": f"{example['prompt']}{example['chosen']}<|end_of_turn|>",
        }
    
    dataset = dataset.map(format_sft_example, remove_columns=["prompt", "chosen", "rejected"])
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=1-split_ratio)
    
    return {
        "train": split_dataset["train"],
        "eval": split_dataset["test"]
    }


def train_sft(config: SFTConfig):
    """
    Stage 1: Supervised Fine-Tuning
    
    Teaches the model the format and basic instruction following.
    Uses standard causal language modeling loss on correct actions.
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer
    except ImportError:
        print("Error: Required packages not installed. Install with:")
        print("  uv pip install trl peft bitsandbytes")
        return False
    
    print("\n" + "=" * 80)
    print("STAGE 1: SUPERVISED FINE-TUNING (SFT)")
    print("=" * 80)
    print(f"Model: {config.base_model_id}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print("=" * 80 + "\n")
    
    # ========== Load Dataset ==========
    datasets = prepare_sft_dataset(
        config.dataset_path,
        split_ratio=config.split_ratio,
        max_samples=config.max_samples
    )
    
    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Eval samples: {len(eval_dataset)}\n")
    
    # ========== Load Tokenizer ==========
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded\n")
    
    # ========== Load Model with Quantization ==========
    print(f"Loading model with 4-bit quantization...")
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
        attn_implementation="flash_attention_2",
    )
    
    model = prepare_model_for_kbit_training(model)
    print("✓ Model loaded with quantization\n")
    
    # ========== Configure LoRA ==========
    print("Configuring LoRA...")
    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=["lm_head"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()
    
    # ========== Training Configuration ==========
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        optim="paged_adamw_8bit",
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to=config.report_to,
        remove_unused_columns=False,
        bf16=torch.cuda.is_bf16_supported(),
    )
    
    # ========== Initialize SFT Trainer ==========
    print("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,  # Disabled for better control
    )
    
    # ========== Train ==========
    print("Starting SFT training...\n")
    trainer.train()
    
    # ========== Save ==========
    print("\nSaving SFT model...")
    trainer.model.save_pretrained(f"{config.output_dir}/final_model")
    tokenizer.save_pretrained(f"{config.output_dir}/final_model")
    print(f"✓ SFT model saved to {config.output_dir}/final_model\n")
    
    return True


# ============================================================================
# STAGE 2: DPO TRAINING
# ============================================================================

def prepare_dpo_dataset(dataset_path: str, split_ratio: float = 0.9, max_samples: Optional[int] = None):
    """
    Prepare dataset for DPO training.
    
    DPO requires:
    - prompt: instruction
    - chosen: correct action
    - rejected: incorrect action
    
    Our DPO JSONL already has this format!
    """
    print(f"Loading DPO dataset from {dataset_path}...")
    
    dataset = load_dataset("json", data_files=dataset_path)["train"]
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # DPO requires prompt, chosen, rejected
    def format_dpo_example(example):
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    
    dataset = dataset.map(format_dpo_example, remove_columns=["prompt", "chosen", "rejected"])
    
    # Split into train/eval
    split_dataset = dataset.train_test_split(test_size=1-split_ratio)
    
    return {
        "train": split_dataset["train"],
        "eval": split_dataset["test"]
    }


def train_dpo(config: DPOConfig):
    """
    Stage 2: Direct Preference Optimization
    
    Refines the SFT model using preference pairs.
    Teaches the model to prefer correct coordinates over incorrect ones.
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import DPOTrainer, DPOConfig as TRLDPOConfig
    except ImportError:
        print("Error: Required packages not installed. Install with:")
        print("  uv pip install trl peft bitsandbytes")
        return False
    
    print("\n" + "=" * 80)
    print("STAGE 2: DIRECT PREFERENCE OPTIMIZATION (DPO)")
    print("=" * 80)
    print(f"Base Model: {config.base_model_id}")
    print(f"Output Directory: {config.output_dir}")
    print(f"Beta (divergence control): {config.beta}")
    print(f"Epochs: {config.num_train_epochs}")
    print("=" * 80 + "\n")
    
    # ========== Load Dataset ==========
    datasets = prepare_dpo_dataset(
        config.dpo_dataset_path,
        split_ratio=config.split_ratio,
        max_samples=config.max_samples
    )
    
    train_dataset = datasets["train"]
    eval_dataset = datasets["eval"]
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Eval samples: {len(eval_dataset)}\n")
    
    # ========== Load Tokenizer ==========
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded\n")
    
    # ========== Load Model ==========
    print(f"Loading model with 4-bit quantization...")
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
        attn_implementation="flash_attention_2",
    )
    
    model = prepare_model_for_kbit_training(model)
    print("✓ Model loaded\n")
    
    # ========== Configure LoRA ==========
    print("Configuring LoRA...")
    target_modules = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=["lm_head"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print()
    
    # ========== DPO Training Configuration ==========
    training_args = TRLDPOConfig(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        beta=config.beta,
        weight_decay=config.weight_decay,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        optim="paged_adamw_8bit",
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to=config.report_to,
        remove_unused_columns=False,
        bf16=torch.cuda.is_bf16_supported(),
    )
    
    # ========== Initialize DPO Trainer ==========
    print("Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    # ========== Train ==========
    print("Starting DPO training...\n")
    trainer.train()
    
    # ========== Save ==========
    print("\nSaving DPO model...")
    trainer.model.save_pretrained(f"{config.output_dir}/final_model")
    tokenizer.save_pretrained(f"{config.output_dir}/final_model")
    print(f"✓ DPO model saved to {config.output_dir}/final_model\n")
    
    return True


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_full_pipeline(run_sft: bool = True, run_dpo: bool = True):
    """
    Run the complete two-stage training pipeline.
    
    Args:
        run_sft: Whether to run Stage 1 (SFT)
        run_dpo: Whether to run Stage 2 (DPO)
    """
    
    print("\n" + "=" * 80)
    print("GUI-ODYSSEY TWO-STAGE TRAINING PIPELINE")
    print("=" * 80)
    print("This pipeline trains Gemma 3 4B for GUI navigation:")
    print("  Stage 1: SFT  - Learn format and basic instruction following")
    print("  Stage 2: DPO  - Learn to prefer correct coordinates")
    print("=" * 80 + "\n")
    
    # ========== Stage 1: SFT ==========
    if run_sft:
        sft_config = SFTConfig()
        success = train_sft(sft_config)
        if not success:
            print("✗ SFT training failed. Exiting.")
            return False
    else:
        print("⊘ Skipping Stage 1 (SFT)\n")
    
    # ========== Stage 2: DPO ==========
    if run_dpo:
        dpo_config = DPOConfig()
        # Use SFT checkpoint as base for DPO
        if run_sft:
            dpo_config.base_model_id = "checkpoints/sft_odyssey/final_model"
        
        success = train_dpo(dpo_config)
        if not success:
            print("✗ DPO training failed. Exiting.")
            return False
    else:
        print("⊘ Skipping Stage 2 (DPO)\n")
    
    print("\n" + "=" * 80)
    print("✓ TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nFinal model saved to: checkpoints/dpo_odyssey/final_model")
    print("\nNext steps:")
    print("  1. Evaluate model on test set")
    print("  2. Run inference on new UI screenshots")
    print("  3. Deploy to production")
    print("=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GUI-Odyssey Two-Stage Training Pipeline")
    parser.add_argument("--sft-only", action="store_true", help="Run only SFT stage")
    parser.add_argument("--dpo-only", action="store_true", help="Run only DPO stage")
    parser.add_argument("--no-sft", action="store_true", help="Skip SFT stage")
    parser.add_argument("--no-dpo", action="store_true", help="Skip DPO stage")
    
    args = parser.parse_args()
    
    # Determine which stages to run
    run_sft = not args.no_sft and not args.dpo_only
    run_dpo = not args.no_dpo and not args.sft_only
    
    # Run pipeline
    success = run_full_pipeline(run_sft=run_sft, run_dpo=run_dpo)
    exit(0 if success else 1)
