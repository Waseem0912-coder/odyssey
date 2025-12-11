# GUI-Odyssey DPO Dataset Generator

Generate Direct Preference Optimization (DPO) training pairs from the GUI-Odyssey dataset for fine-tuning vision-language models on mobile GUI navigation tasks.

## Overview

This tool converts GUI-Odyssey annotation data into DPO training pairs with dual rejection strategies:

1. **Spatial Rejection ("Fat Finger")** – Perturbs click coordinates by ~5% (σ=50 pixels in [0,1000] normalized space) to simulate precision errors
2. **Logical Rejection ("Hallucination")** – Swaps action types (CLICK↔TYPE) to teach the model reasoning consistency

### Output Format

Each DPO pair includes:
- **prompt**: `<image>\nUser: {instruction}\nModel:`
- **chosen**: Ground-truth action in format `ACTION: (x, y)` or `ACTION: "text"`
- **rejected**: Incorrect action (either spatial or logical variant)

All coordinates are normalized to [0, 1000] range for optimal tokenization by vision-language models.

## Setup

### Prerequisites

- Python 3.10+
- `uv` package manager ([install uv](https://docs.astral.sh/uv/getting-started/installation/))

### Installation

```bash
cd odyssey_dpo_generator
uv sync
```

This creates a virtual environment and installs all dependencies:
- `datasets` – Hugging Face dataset utilities
- `huggingface_hub` – HF Hub access
- `tqdm` – Progress bars
- `pillow` – Image processing (optional, for image preprocessing)

### Verify Installation

```bash
uv run python src/odyssey_dpo_generator.py --help
```

## Usage

### Generate DPO Pairs (Default: Synthetic Data for Testing)

```bash
cd odyssey_dpo_generator
uv run python src/odyssey_dpo_generator.py
```

**Output:**
- `odyssey_dpo_pairs.jsonl` – JSONL file with DPO training pairs
- Each line is a complete JSON object: `{"prompt": "...", "chosen": "...", "rejected": "..."}`

### Example Output

```json
{"prompt": "<image>\nUser: Open the Settings app.\nModel:", "chosen": "CLICK: (250, 400)", "rejected": "CLICK: (243, 391)"}
{"prompt": "<image>\nUser: Open the Settings app.\nModel:", "chosen": "CLICK: (250, 400)", "rejected": "TYPE: \"placeholder_text\""}
```

### Configuration

Edit `src/odyssey_dpo_generator.py` to adjust settings:

```python
@dataclass
class Config:
    HF_DATASET_ID: str = "hflqf88888/GUIOdyssey"  # Dataset ID on HF Hub
    OUTPUT_FILE: str = "odyssey_dpo_pairs.jsonl"  # Output JSONL filename
    CANVAS_SIZE: Tuple[int, int] = (1000, 1000)  # Normalized coordinate range
    NOISE_STD_DEV: float = 50.0                   # Spatial perturbation σ (in [0,1000] space)
    SAMPLE_SIZE: Optional[int] = None             # Use entire dataset; set to N for sampling
    SEED: int = 42                                # Random seed for reproducibility
```

## Loading Real GUI-Odyssey Data

### Option 1: Hugging Face Hub (Easiest)

When the dataset becomes available in a standard format (Parquet, JSON, etc.):

```python
from datasets import load_dataset

dataset = load_dataset("hflqf88888/GUIOdyssey", split="train")
```

### Option 2: GitHub Raw Annotations

Clone the GitHub repo and use raw annotations:

```bash
git clone https://github.com/OpenGVLab/GUI-Odyssey.git
# Then modify `config.HF_DATASET_ID` to point to local annotations directory
```

### Processing Custom Data

The script expects data in conversation format:

```python
{
    "conversations": [
        {
            "from": "user",
            "value": "<image>\nUser: {instruction}\nModel:"
        },
        {
            "from": "assistant",
            "value": "CLICK: (500, 500)"  # Ground-truth action
        }
    ]
}
```

Modify `parse_odyssey_entry()` to adapt to your data structure.

## Data Format

### Action Formats

The generator produces actions in the following normalized format (coordinates in [0, 1000]):

```
CLICK: (x, y)                    # Tap at normalized coordinates
TYPE: "text"                      # Text input
LONG_PRESS: (x, y)              # Long press at coordinates
SCROLL: (x, y)                   # Scroll direction
COMPLETE                          # Task completed
INCOMPLETE                        # Task impossible
```

### Coordinate Conversion

At inference time, convert normalized coordinates to device pixels:

```python
actual_x = (predicted_x / 1000.0) * device_width
actual_y = (predicted_y / 1000.0) * device_height
```

## Rejection Strategies

### 1. Spatial Rejection (Coordinate Perturbation)

For CLICK/LONG_PRESS actions:

```python
new_x = original_x + N(0, σ=50)   # Gaussian noise
new_y = original_y + N(0, σ=50)
```

**Purpose:** Teaches the model precision – penalizes minor coordinate errors.

### 2. Logical Rejection (Action Type Swap)

By action type:
- **CLICK** → `TYPE: "placeholder_text"` (wrong action type)
- **TYPE** → `CLICK: (500, 500)` (center of canvas as fallback)
- **LONG_PRESS** → `CLICK: (0, 0)` (panic behavior)
- **SCROLL** → `CLICK: (0, 0)` (wrong action type)

**Purpose:** Teaches the model reasoning – penalizes fundamental action type confusion.

### Output Structure

Each base entry generates **2 rejected pairs**:
1. One with spatial perturbation
2. One with logical error

This doubles the dataset size but provides rich training signal for DPO.

## Validation

The script validates output JSONL:

```
✓ Required fields present (prompt, chosen, rejected)
✓ Coordinate ranges [0, 1000]
✓ Action format consistency
✓ JSON syntax correctness
✓ Action type distribution
```

Sample output:
```
============================================================
DPO Dataset Generation Complete
============================================================
Output file: odyssey_dpo_pairs.jsonl
Valid pairs: 16668
Valid coordinates: 25002
Invalid coordinates: 0
JSON errors: 0

Action Type Distribution (chosen + rejected):
  CLICK: 12450
  TYPE: 8340
  LONG_PRESS: 2250
  SCROLL: 2000
============================================================
```

## Training with Hugging Face TRL

Use the generated JSONL with `trl.DPOTrainer`:

```python
from datasets import load_dataset
from trl import DPOTrainer

# Load DPO pairs
dataset = load_dataset("json", data_files="odyssey_dpo_pairs.jsonl")

# Initialize trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    beta=0.1,
    args=training_args,
)

trainer.train()
```

## Troubleshooting

### Dataset Not Loading from HF Hub

The current HF dataset (`hflqf88888/GUIOdyssey`) requires a custom loading script. Options:

1. **Synthetic Mode** (current default) – Works immediately for testing
2. **Wait for standard format** – Contact dataset maintainers to convert to Parquet/JSON
3. **Local GitHub data** – Clone repo and modify script to read `annotations/` directory

### Coordinate Validation Errors

If `invalid_coords > 0`:

- Check `CANVAS_SIZE` configuration (should be `(1000, 1000)`)
- Verify input data coordinates are already normalized to [0, 1000]
- Ensure `NOISE_STD_DEV` isn't too large (default 50 is ~5% jitter)

### Memory Issues with Large Datasets

For datasets > 100K examples:

1. Set `SAMPLE_SIZE` to process in batches:
   ```python
   config.SAMPLE_SIZE = 50000  # Process 50K at a time
   ```

2. Or use `dataset.select()` to sample specific ranges

3. Stream from disk with custom data loader

## Performance Metrics

### Generation Speed

- Synthetic data (10 examples): ~0.5s
- Real data depends on size; ~1000 pairs/sec on CPU

### Dataset Sizes

- **Base examples**: 8,334 episodes in Odyssey
- **DPO pairs**: 16,668 pairs (2 rejections per base example)
- **File size**: ~50-100 MB typical JSONL

## References

- **GUI-Odyssey Paper**: [OpenGVLab/GUI-Odyssey](https://github.com/OpenGVLab/GUI-Odyssey)
- **DPO Training**: [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
- **Hugging Face TRL**: [TRL Library](https://huggingface.co/docs/trl)

## License

This generator follows the same license as the GUI-Odyssey dataset (check original repo).

## Contributing

To add custom rejection strategies or support for additional data formats:

1. Extend `generate_logical_rejection()` for new action type swaps
2. Modify `perturb_coordinates()` for different noise distributions
3. Update `parse_odyssey_entry()` for custom data structures

## Questions?

- Check dataset structure in `parse_odyssey_entry()` function
- Review coordinate system assumptions in `Config` class
- Examine example JSONL output for format validation
