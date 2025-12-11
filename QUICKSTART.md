# Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
cd odyssey_dpo_generator
uv sync
```

### 2. Generate DPO Dataset

```bash
uv run python src/odyssey_dpo_generator.py
```

This creates `odyssey_dpo_pairs.jsonl` with synthetic training data (~20 pairs for testing).

**Output:**
```
============================================================
DPO Dataset Generation Complete
============================================================
Output file: odyssey_dpo_pairs.jsonl
Valid pairs: 20
...
```

### 3. Inspect Dataset

```bash
head -3 odyssey_dpo_pairs.jsonl | python -m json.tool
```

**Output:**
```json
{
  "prompt": "<image>\nUser: Open the Settings app.\nModel:",
  "chosen": "CLICK: (250, 400)",
  "rejected": "CLICK: (243, 391)"
}
```

### 4. Load with Hugging Face Datasets

```python
from datasets import load_dataset

dpo_dataset = load_dataset("json", data_files="odyssey_dpo_pairs.jsonl")
print(f"Loaded {len(dpo_dataset['train'])} examples")
```

## Using Real GUI-Odyssey Data

Currently, the dataset on HF Hub requires special handling. To use real data once available:

### Option A: Update HF_DATASET_ID (When Available in Standard Format)

```python
# In src/odyssey_dpo_generator.py, change config:
config.HF_DATASET_ID = "OpenGVLab/GUI-Odyssey"  # When converted to Parquet/JSON
```

### Option B: Use GitHub Raw Annotations

```bash
# Clone the repo
git clone https://github.com/OpenGVLab/GUI-Odyssey.git

# Modify the script to read from local annotations:
# Change parse_odyssey_entry() to read from JSON files in annotations/ folder
```

## Customizing the Generator

### Change Noise Level (Spatial Rejection Sensitivity)

```python
config.NOISE_STD_DEV = 20.0  # Less jitter (2%)
# or
config.NOISE_STD_DEV = 100.0  # More jitter (10%)
```

### Generate More Pairs from Same Base Data

```python
config.SAMPLE_SIZE = None  # Use entire dataset (default)
# To process in batches:
config.SAMPLE_SIZE = 1000  # Process 1000 base examples = 2000 DPO pairs
```

### Change Output File

```python
config.OUTPUT_FILE = "my_dpo_dataset.jsonl"
```

## Training Integration

### With Hugging Face TRL

```bash
# Install TRL (optional dependency)
uv pip install trl peft

# View example training code
cat examples_training.py

# Run example (inspect dataset without training)
uv run python examples_training.py
```

### With Custom Training Loop

```python
from datasets import load_dataset
import json

# Load pairs
dataset = load_dataset("json", data_files="odyssey_dpo_pairs.jsonl")

# Iterate over examples
for example in dataset["train"]:
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]
    
    # Your DPO loss computation here
    # loss = dpo_loss(model(prompt), chosen, rejected)
```

## Troubleshooting

### ✗ Module Not Found: `datasets`

```bash
uv sync
```

### ✗ File Not Found: `odyssey_dpo_pairs.jsonl`

Run the generator first:
```bash
uv run python src/odyssey_dpo_generator.py
```

### ✗ Cannot Load HF Dataset

This is expected - the dataset is not yet available in a standard format. The script falls back to synthetic data for testing. Once the dataset is converted to Parquet/JSON format, update `HF_DATASET_ID` in the config.

## File Structure

```
odyssey_dpo_generator/
├── src/
│   └── odyssey_dpo_generator.py    # Main script
├── examples_training.py             # DPO training example
├── pyproject.toml                   # UV project config
├── README.md                        # Full documentation
└── QUICKSTART.md                    # This file
```

## Next Steps

1. **Generate more data** – Adjust `config.SAMPLE_SIZE` to 10000+ for real training
2. **Integrate with TRL** – Use `examples_training.py` as starting point
3. **Fine-tune a model** – See [Gemma Fine-Tuning Guide](https://huggingface.co/blog/gemma-fine-tuning)
4. **Deploy** – Package trained model for mobile GUI navigation

## Reference Commands

```bash
# Generate dataset
uv run python src/odyssey_dpo_generator.py

# Load in Python
python -c "from datasets import load_dataset; ds = load_dataset('json', data_files='odyssey_dpo_pairs.jsonl'); print(len(ds['train']))"

# View first pair
python -c "import json; f = open('odyssey_dpo_pairs.jsonl'); print(json.dumps(json.loads(f.readline()), indent=2))"

# Count pairs
wc -l odyssey_dpo_pairs.jsonl

# Sample 100 pairs for testing
head -100 odyssey_dpo_pairs.jsonl > sample_dpo_pairs.jsonl
```

## Performance Tips

- **Faster generation**: Use `SAMPLE_SIZE` to limit examples
- **Faster training**: Reduce `NOISE_STD_DEV` for less computation
- **Better quality**: Use real Odyssey data (when available) instead of synthetic
- **Reproducible**: Set `SEED` in config for deterministic noise

## Questions?

- Check `README.md` for full documentation
- See `src/odyssey_dpo_generator.py` for implementation details
- View `examples_training.py` for integration patterns
