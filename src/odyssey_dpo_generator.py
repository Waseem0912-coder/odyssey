"""
GUI-Odyssey DPO Dataset Generator

Fetches preprocessed GUI-Odyssey data from Hugging Face Hub and generates
DPO (Direct Preference Optimization) training pairs with dual rejection strategies:
1. Spatial rejection ("Fat Finger") - Coordinate perturbation with Gaussian noise
2. Logical rejection ("Hallucination") - Action type swaps and default behaviors

All coordinates are maintained in normalized [0, 1000] range.
"""

import json
import random
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

from datasets import load_dataset
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration for DPO dataset generation."""
    HF_DATASET_ID: str = "hflqf88888/GUIOdyssey"
    OUTPUT_FILE: str = "odyssey_dpo_pairs.jsonl"
    CANVAS_SIZE: Tuple[int, int] = (1000, 1000)  # Normalized coordinates
    NOISE_STD_DEV: float = 50.0  # Standard deviation in [0, 1000] space (~5% jitter)
    SAMPLE_SIZE: Optional[int] = None  # None = use entire dataset; int = sample N examples
    SEED: int = 42


config = Config()
random.seed(config.SEED)


# ============================================================================
# Coordinate Perturbation ("Fat Finger" - Spatial Rejection)
# ============================================================================

def perturb_coordinates(
    x: float,
    y: float,
    width: float = 1000.0,
    height: float = 1000.0,
    noise_scale: float = 50.0,
) -> Tuple[int, int]:
    """
    Generates a 'Hard Negative' by shifting coordinates slightly.
    Simulates the model missing the target by a small margin.

    Args:
        x, y: Original coordinates in [0, 1000] range
        width, height: Canvas dimensions (default 1000x1000)
        noise_scale: Standard deviation for Gaussian noise

    Returns:
        Perturbed (x, y) coordinates clipped to valid range
    """
    # Add Gaussian noise
    new_x = x + random.gauss(0, noise_scale)
    new_y = y + random.gauss(0, noise_scale)

    # Clip to canvas boundaries
    new_x = max(0, min(width, new_x))
    new_y = max(0, min(height, new_y))

    # Convert to integers
    new_x = int(round(new_x))
    new_y = int(round(new_y))

    # Ensure we didn't accidentally land back on the exact same pixel
    if new_x == int(x) and new_y == int(y):
        new_x = (int(x) + 10) % int(width)

    return new_x, new_y


# ============================================================================
# Action Formatting
# ============================================================================

def format_action_string(
    action_type: str,
    coords: Optional[Tuple[int, int]] = None,
    text: Optional[str] = None,
) -> str:
    """
    Formats action into string format matching Odyssey's decoded action format.
    Maintains normalized [0, 1000] coordinate system.

    Args:
        action_type: One of CLICK, TYPE, SCROLL, LONG_PRESS, COMPLETE, INCOMPLETE
        coords: (x, y) tuple in [0, 1000] range
        text: Text string for TYPE actions

    Returns:
        Formatted action string
    """
    if action_type == "CLICK":
        if coords:
            return f"CLICK: ({coords[0]}, {coords[1]})"
        return "CLICK: (0, 0)"
    elif action_type == "LONG_PRESS":
        if coords:
            return f"LONG_PRESS: ({coords[0]}, {coords[1]})"
        return "LONG_PRESS: (0, 0)"
    elif action_type == "TYPE":
        if text:
            return f'TYPE: "{text}"'
        return 'TYPE: ""'
    elif action_type == "SCROLL":
        if coords:
            return f"SCROLL: ({coords[0]}, {coords[1]})"
        return "SCROLL: (0, 0)"
    elif action_type == "COMPLETE":
        return "COMPLETE"
    elif action_type == "INCOMPLETE":
        return "INCOMPLETE"
    else:
        return "STOP"


# ============================================================================
# Rejection Generation Strategies
# ============================================================================

def generate_spatial_rejection(
    action_type: str,
    coords: Optional[Tuple[int, int]] = None,
    text: Optional[str] = None,
) -> str:
    """
    Spatial Rejection Strategy ("Fat Finger"):
    Perturb coordinates slightly for CLICK/LONG_PRESS actions.
    For TYPE, return the chosen action unchanged (spatial doesn't apply).
    """
    if action_type in ["CLICK", "LONG_PRESS"] and coords:
        bad_x, bad_y = perturb_coordinates(
            coords[0], coords[1], noise_scale=config.NOISE_STD_DEV
        )
        return format_action_string(action_type, coords=(bad_x, bad_y))
    else:
        # For non-coordinate actions, spatial rejection doesn't apply
        # Use a reasonable fallback
        return format_action_string(action_type, coords=coords, text=text)


def generate_logical_rejection(
    action_type: str,
    coords: Optional[Tuple[int, int]] = None,
    text: Optional[str] = None,
) -> str:
    """
    Logical Rejection Strategy ("Hallucination"):
    Swap action types or use default incorrect behaviors.

    - CLICK -> TYPE with wrong text
    - TYPE -> CLICK at center (500, 500)
    - LONG_PRESS -> CLICK at (0, 0) (default panic)
    - SCROLL -> CLICK at (0, 0)
    """
    if action_type == "CLICK":
        # Model hallucinates and types instead
        return format_action_string("TYPE", text="placeholder_text")
    elif action_type == "TYPE":
        # Model hallucinates and clicks at center instead
        return format_action_string("CLICK", coords=(500, 500))
    elif action_type == "LONG_PRESS":
        # Model panics and clicks at origin
        return format_action_string("CLICK", coords=(0, 0))
    elif action_type == "SCROLL":
        # Model forgets scroll, tries to click
        return format_action_string("CLICK", coords=(0, 0))
    else:
        # Fallback: default incorrect action
        return format_action_string("CLICK", coords=(0, 0))


# ============================================================================
# Dataset Loading and Parsing
# ============================================================================

def parse_odyssey_entry(entry: Dict) -> Optional[Dict]:
    """
    Parse a GUI-Odyssey entry from Hugging Face dataset.
    Extract instruction, action type, coordinates, and text.

    Returns:
        Dict with keys: instruction, action_type, coords, text
        None if entry cannot be parsed
    """
    try:
        # Expect HF dataset to have conversation format
        conversations = entry.get("conversations", [])
        if not conversations or len(conversations) < 2:
            return None

        # Extract instruction from user message
        user_msg = conversations[0].get("value", "")
        assistant_msg = conversations[1].get("value", "")

        if not user_msg or not assistant_msg:
            return None

        instruction = user_msg.strip()
        action_str = assistant_msg.strip()

        # Parse action string to extract type and parameters
        action_type, coords, text = parse_action_string(action_str)

        if action_type is None:
            return None

        return {
            "instruction": instruction,
            "action_type": action_type,
            "coords": coords,
            "text": text,
            "raw_action": action_str,
        }

    except Exception as e:
        print(f"Error parsing entry: {e}")
        return None


def parse_action_string(action_str: str) -> Tuple[Optional[str], Optional[Tuple[int, int]], Optional[str]]:
    """
    Parse action string to extract type, coordinates, and text.

    Handles formats:
    - "CLICK: (x, y)"
    - "TYPE: \"text\""
    - "LONG_PRESS: (x, y)"
    - "SCROLL: (x, y)"
    - "COMPLETE" / "INCOMPLETE"
    """
    action_str = action_str.strip()

    try:
        if action_str.startswith("CLICK:"):
            coords_str = action_str.replace("CLICK:", "").strip().strip("()")
            x, y = map(float, coords_str.split(","))
            return "CLICK", (int(x), int(y)), None

        elif action_str.startswith("LONG_PRESS:"):
            coords_str = action_str.replace("LONG_PRESS:", "").strip().strip("()")
            x, y = map(float, coords_str.split(","))
            return "LONG_PRESS", (int(x), int(y)), None

        elif action_str.startswith("TYPE:"):
            text = action_str.replace('TYPE:', "").strip().strip('"\'')
            return "TYPE", None, text

        elif action_str.startswith("SCROLL:"):
            coords_str = action_str.replace("SCROLL:", "").strip().strip("()")
            x, y = map(float, coords_str.split(","))
            return "SCROLL", (int(x), int(y)), None

        elif action_str.startswith("COMPLETE"):
            return "COMPLETE", None, None

        elif action_str.startswith("INCOMPLETE"):
            return "INCOMPLETE", None, None

        else:
            return None, None, None

    except Exception:
        return None, None, None


# ============================================================================
# DPO Pair Generation
# ============================================================================

def generate_dpo_pairs(dataset):
    """
    Generate DPO pairs from GUI-Odyssey dataset.

    For each valid entry, creates 2 rejected responses:
    1. Spatial rejection (coordinate perturbation)
    2. Logical rejection (action type swap)

    Yields:
        Dict with keys: prompt, chosen, rejected, metadata
    """
    stats = defaultdict(int)
    valid_count = 0

    for idx, entry in enumerate(tqdm(dataset, desc="Processing entries")):
        parsed = parse_odyssey_entry(entry)

        if parsed is None:
            stats["skipped_invalid"] += 1
            continue

        action_type = parsed["action_type"]
        coords = parsed["coords"]
        text = parsed["text"]
        instruction = parsed["instruction"]

        # Format chosen (correct) action
        chosen = format_action_string(action_type, coords=coords, text=text)

        # Create prompt with image placeholder
        prompt = f"<image>\nUser: {instruction}\nModel:"

        # Generate spatial rejection (Fat Finger)
        spatial_rejected = generate_spatial_rejection(
            action_type, coords=coords, text=text
        )
        yield {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": spatial_rejected,
            "metadata": {
                "action_type": action_type,
                "rejection_strategy": "spatial",
                "source": "odyssey",
            },
        }
        stats["pairs_spatial"] += 1

        # Generate logical rejection (Hallucination)
        logical_rejected = generate_logical_rejection(
            action_type, coords=coords, text=text
        )
        yield {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": logical_rejected,
            "metadata": {
                "action_type": action_type,
                "rejection_strategy": "logical",
                "source": "odyssey",
            },
        }
        stats["pairs_logical"] += 1
        valid_count += 1

        # Optional: stop after N samples for testing
        if config.SAMPLE_SIZE and valid_count >= config.SAMPLE_SIZE:
            break

    return stats


# ============================================================================
# Output Writing
# ============================================================================

def write_jsonl(output_file: str, pairs_generator):
    """
    Write DPO pairs to JSONL file.
    One JSON object per line.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pair in pairs_generator:
            # Remove metadata from output if not needed for training
            output_pair = {
                "prompt": pair["prompt"],
                "chosen": pair["chosen"],
                "rejected": pair["rejected"],
            }
            f.write(json.dumps(output_pair) + "\n")

    return output_path


# ============================================================================
# Validation
# ============================================================================

def validate_coordinates(x: int, y: int, canvas_size: Tuple[int, int] = (1000, 1000)) -> bool:
    """Check if coordinates are within valid range [0, canvas_size]."""
    return 0 <= x <= canvas_size[0] and 0 <= y <= canvas_size[1]


def validate_jsonl(output_file: str) -> Dict[str, int]:
    """
    Validate JSONL output:
    - Check all required fields present
    - Verify coordinate ranges
    - Count action types
    """
    stats = defaultdict(int)

    with open(output_file, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)

                # Check required fields
                if "prompt" in obj and "chosen" in obj and "rejected" in obj:
                    stats["valid_pairs"] += 1

                    # Extract and validate coordinates from action strings
                    for action_str in [obj["chosen"], obj["rejected"]]:
                        if "CLICK:" in action_str or "LONG_PRESS:" in action_str:
                            try:
                                coords_str = action_str.split("(")[1].split(")")[0]
                                x, y = map(int, coords_str.split(","))
                                if validate_coordinates(x, y):
                                    stats["valid_coords"] += 1
                                else:
                                    stats["invalid_coords"] += 1
                            except:
                                pass

                        # Count action types
                        if "CLICK:" in action_str:
                            stats["action_click"] += 1
                        elif "TYPE:" in action_str:
                            stats["action_type"] += 1
                        elif "LONG_PRESS:" in action_str:
                            stats["action_long_press"] += 1
                        elif "SCROLL:" in action_str:
                            stats["action_scroll"] += 1
                else:
                    stats["invalid_pairs"] += 1
            except json.JSONDecodeError:
                stats["json_errors"] += 1

    return stats


# ============================================================================
# Main Generation Pipeline
# ============================================================================

def main():
    """Main generation pipeline."""
    print(f"Loading GUI-Odyssey dataset from Hugging Face: {config.HF_DATASET_ID}")

    try:
        # Try direct instantiation for raw dataset structure
        dataset = load_dataset(config.HF_DATASET_ID, data_files="data.json")
        if isinstance(dataset, dict):
            dataset = list(dataset.values())[0]
        print(f"Loaded dataset with {len(dataset)} examples")

        # Optional: limit to sample size for testing
        if config.SAMPLE_SIZE:
            dataset = dataset.select(range(min(config.SAMPLE_SIZE, len(dataset))))
            print(f"Using sample of {len(dataset)} examples")

    except Exception as e:
        print(f"Error loading with data_files: {e}")
        print(f"Attempting to load with split specification...")
        try:
            # Try loading different splits
            for split_name in ["train", "validation", "test"]:
                try:
                    dataset = load_dataset(config.HF_DATASET_ID, split=split_name, trust_remote_code=True)
                    print(f"Successfully loaded '{split_name}' split with {len(dataset)} examples")
                    break
                except Exception:
                    continue
            else:
                # If no split works, try loading without split
                dataset = load_dataset(config.HF_DATASET_ID, trust_remote_code=True)
                if isinstance(dataset, dict):
                    dataset = list(dataset.values())[0]
                print(f"Loaded dataset with {len(dataset)} examples")

            # Optional: limit to sample size for testing
            if config.SAMPLE_SIZE:
                dataset = dataset.select(range(min(config.SAMPLE_SIZE, len(dataset))))
                print(f"Using sample of {len(dataset)} examples")

        except Exception as e2:
            print(f"Failed to load dataset: {e2}")
            print(f"\nCannot load preprocessed data. Will create synthetic DPO examples instead.")
            # Create minimal example for testing
            dataset = create_synthetic_examples(10)
            print(f"Created {len(dataset)} synthetic examples for testing")

    # Generate DPO pairs
    print("\nGenerating DPO pairs...")
    pairs_gen = generate_dpo_pairs(dataset)
    
    # Write to JSONL
    print(f"\nWriting to {config.OUTPUT_FILE}...")
    output_path = write_jsonl(config.OUTPUT_FILE, pairs_gen)

    # Validate output
    print("\nValidating output JSONL...")
    validation_stats = validate_jsonl(config.OUTPUT_FILE)

    # Print summary statistics
    print("\n" + "="*60)
    print("DPO Dataset Generation Complete")
    print("="*60)
    print(f"Output file: {output_path}")
    print(f"Valid pairs: {validation_stats['valid_pairs']}")
    print(f"Valid coordinates: {validation_stats['valid_coords']}")
    print(f"Invalid coordinates: {validation_stats['invalid_coords']}")
    print(f"JSON errors: {validation_stats['json_errors']}")
    print("\nAction Type Distribution (chosen + rejected):")
    print(f"  CLICK: {validation_stats['action_click']}")
    print(f"  TYPE: {validation_stats['action_type']}")
    print(f"  LONG_PRESS: {validation_stats['action_long_press']}")
    print(f"  SCROLL: {validation_stats['action_scroll']}")
    print("="*60)


def create_synthetic_examples(n_examples: int) -> List[Dict]:
    """Create synthetic examples for testing when HF data unavailable."""
    examples = []
    
    actions = [
        {"type": "CLICK", "x": 250, "y": 400},
        {"type": "CLICK", "x": 500, "y": 600},
        {"type": "TYPE", "text": "search query"},
        {"type": "CLICK", "x": 100, "y": 200},
        {"type": "SCROLL", "x": 500, "y": 500},
        {"type": "LONG_PRESS", "x": 300, "y": 350},
    ]
    
    instructions = [
        "Open the Settings app.",
        "Type 'hello' in the search bar.",
        "Tap on the home button.",
        "Scroll down to see more options.",
        "Long press to open context menu.",
        "Find and click the search icon.",
    ]
    
    for i in range(n_examples):
        action = actions[i % len(actions)]
        instruction = instructions[i % len(instructions)]
        
        example = {
            "conversations": [
                {
                    "from": "user",
                    "value": f"<image>\nUser: {instruction}\nModel:"
                },
                {
                    "from": "assistant",
                    "value": format_action_string(
                        action["type"],
                        coords=(action.get("x"), action.get("y")) if "x" in action else None,
                        text=action.get("text")
                    )
                }
            ]
        }
        examples.append(example)
    
    return examples


if __name__ == "__main__":
    main()
