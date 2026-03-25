# AGENTS.md — Python + Panda + Hugging Face Vision + LLM Model

## 🧠 Project Context
This project is designed to **detect differences in images** using Python and pandas for data handling, and a Large Language Model (LLM) for descriptive analysis.  
It currently supports **PNG images only**.

The differences between images are not just detected but also **described in poetic, human-readable English** using the LLM.

---

## 🗂 Directory Structure
project/
├─ data/ # Source images and datasets
├─ outputs/ # Results, difference images
├─ notebooks/ # Jupyter notebooks for testing and exploration
├─ src/ # Python scripts and modules
│ ├─ agents.py # LLM agent scripts
│ ├─ image_diff.py # Image comparison logic
│ └─ utils.py # Helper functions
├─ .venv/ # Virtual environment
├─ requirements.txt # Python dependencies
└─ AGENTS.md # This documentation

---

## 🤖 Agents

### 1. Image Difference Agent
- **Language**: Python  
- **Libraries**: `Pillow`, `numpy`, `pandas`  
- **Function**: Compares two PNG images pixel by pixel or region by region, logs differences into structured data (pandas DataFrame).

### 2. LLM Description Agent
- **Language**: Python  
- **Library**: `OpenAI` / `ultralytics` / other LLM wrapper  
- **Function**: Reads the structured differences from the Image Difference Agent and generates **poetic, human-like English descriptions**.

---

## ⚡ Usage Example
```python
from src.image_diff import compare_images
from src.agents import describe_differences

# Compare images
diff_data = compare_images("data/image1.png", "data/image2.png")

# Generate poetic description
description = describe_differences(diff_data)
print(description)

