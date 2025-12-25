# Falcon-7B Grammar Correction Fine-tuning

This project demonstrates fine-tuning the Falcon-7B model for grammar correction tasks using LoRA (Low-Rank Adaptation) and 4-bit quantization for efficient training. The repository includes the complete pipeline from data preprocessing to model training and evaluation.

## Overview

This project showcases:
- Data preprocessing pipeline for grammar correction datasets
- Loading and quantizing the Falcon-7B-Instruct model using 4-bit quantization
- Fine-tuning using LoRA with PEFT (Parameter-Efficient Fine-Tuning)
- Model evaluation and performance assessment
- Saving and deploying the fine-tuned model to Hugging Face Hub

## Model Details

- **Base Model**: `vilsonrodrigues/falcon-7b-instruct-sharded`
- **Fine-tuned Model**: `majed-ai/trained2-grammar-falcon-7b`
- **Task**: Grammar correction
- **Method**: LoRA fine-tuning with 4-bit quantization

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Sufficient GPU memory (8GB+ recommended)

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required packages:
```bash
pip install torch
pip install bitsandbytes
pip install datasets
pip install peft
pip install accelerate
pip install loralib
pip install einops
pip install transformers
```

### Running the Project

1. **Data Preprocessing** (optional - datasets are already processed):
   ```bash
   python "edit csv.py"
   ```
   This script processes the raw dataset and adds instruction prompts.

2. **Model Training**:
   Open and run the Jupyter notebook:
   ```bash
   jupyter notebook "MJ_falcon_7b_finetune_grammar_falcon (1).ipynb"
   ```

3. **Model Usage**:
   The trained model is available on Hugging Face Hub: `majed-ai/trained2-grammar-falcon-7b`

## Requirements

The following packages are required:

```bash
pip install torch
pip install bitsandbytes
pip install datasets
pip install peft
pip install accelerate
pip install loralib
pip install einops
pip install transformers
```

## Configuration

### Model Quantization
- 4-bit quantization using `bitsandbytes`
- Quantization type: NF4 (Normal Float 4)
- Double quantization enabled
- Compute dtype: bfloat16

### LoRA Configuration
- Rank (r): 16
- Alpha: 32
- Target modules: `query_key_value`
- Dropout: 0.05
- Task type: Causal Language Modeling
- **Trainable parameters**: 4.7M (0.13% of total 3.6B parameters)

### Training Configuration
- Batch size: 4 per device
- Gradient accumulation steps: 4
- Number of epochs: 2
- Learning rate: 5e-5
- Optimizer: Paged AdamW 8-bit
- LR scheduler: Polynomial decay
- Warmup ratio: 0.05
- FP16 training enabled

## Dataset

The model is trained on a custom grammar correction dataset that went through several preprocessing stages:

1. **Raw Data (`dataset3.csv`)**: Original dataset with 9 sample grammar correction pairs
2. **Intermediate Processing (`my_grammar.csv`)**: Cleaned and indexed version
3. **Final Training Data (`my_grammar1.csv`)**: Formatted with instruction prompts for fine-tuning

### Dataset Statistics
- **Format**: CSV with input-target pairs
- **Sample Size**: 9 examples (expanded to 6,004 in full training)
- **Task**: Grammar correction with instructional prompting

### Data Processing Pipeline
The `edit csv.py` script transforms raw sentences into instruction-following format:
- **Input Format**: `"correct the following sentence: [incorrect sentence]"`
- **Target Format**: `"[corrected sentence]"`

### Example Transformations
```
Original Input: "New and new technology has been introduced to the society."
Processed Input: "correct the following sentence: New and new technology has been introduced to the society."
Target Output: "New technology has been introduced to society."
```

## Training Results

- **Training steps**: 750
- **Training loss**: 1.662
- **Training runtime**: ~6,477 seconds (~1.8 hours)
- **Training samples per second**: 1.854

## Usage

### Before Fine-tuning
```python
prompt = """
<human>: correct the following sentence: New and new technology has been introduced to the society .
<assistant>:
"""
```
Output: No correction (model repeats the incorrect sentence)

### After Fine-tuning
```python
prompt = """
<human>: correct the following sentence: Haven't nobody told you about the changes to the schedule? I thought everybody was knowing by now.
<assistant>:
"""
```
Output: "Haven't you heard about the changes to the schedule? I thought everybody knew about it by now."

## Loading the Fine-tuned Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch

# Load configuration
config = PeftConfig.from_pretrained("majed-ai/trained2-grammar-falcon-7b")

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Load fine-tuned weights
model = PeftModel.from_pretrained(model, "majed-ai/trained2-grammar-falcon-7b")
```

## Generation Parameters

- Max new tokens: 200
- Temperature: 0.7
- Top-p: 0.7
- Number of return sequences: 1

## Project Structure

```
.
├── MJ_falcon_7b_finetune_grammar_falcon (1).ipynb  # Main training notebook
├── edit csv.py                                      # Data preprocessing script
├── dataset3.csv                                     # Original raw dataset
├── my_grammar.csv                                   # Intermediate processed dataset
├── my_grammar1.csv                                  # Final training dataset (formatted)
└── README.md                                        # Project documentation
```

## Files Description

### Core Files
- **`MJ_falcon_7b_finetune_grammar_falcon (1).ipynb`**: Complete Jupyter notebook containing the Falcon-7B fine-tuning pipeline for grammar correction. Includes data loading, model configuration, training, evaluation, and model saving.

### Data Files
- **`dataset3.csv`**: Original raw dataset containing input-target pairs for grammar correction (9 sample entries)
- **`my_grammar.csv`**: Intermediate dataset with indexed rows and cleaned formatting
- **`my_grammar1.csv`**: Final processed training dataset with proper prompt formatting ("correct the following sentence: " prefix added to inputs)

### Utilities
- **`edit csv.py`**: Python script for preprocessing the dataset, specifically adding the instruction prompt prefix to input sentences for proper model training format

## Notes

- The model uses gradient checkpointing to reduce memory usage
- Training was performed on CUDA device 0
- The notebook includes visualization of training loss over steps

## License

Please refer to the Falcon-7B model license and Hugging Face terms of service.

## Acknowledgments

- Falcon-7B model by TII
- PEFT library by Hugging Face
- bitsandbytes for efficient quantization
