# Image Comparison LLM Tool

This Python-based Large Language Model (LLM) tool leverages vision-capable models (e.g., via Ollama or OpenAI API) to analyze and compare two input images. It generates detailed textual descriptions of each image and highlights key differences, including visual elements, colors, compositions, objects, and contextual changes. Ideal for content creators, QA testers, and developers needing automated image analysis without manual inspection.

## Features
- **Image Upload**: Accepts two image files (JPEG, PNG, etc.) as input.
- **LLM-Powered Analysis**: Uses prompts to extract descriptions and comparisons from the model.
- **Difference Highlighting**: Outputs structured reports on similarities and discrepancies.
- **Customizable Prompts**: Easily modify prompts for specific comparison criteria (e.g., focus on text, layout, or objects).
- **API Integration**: Connects to local Ollama instance or cloud APIs for processing.

## Usage
1. Install dependencies: `pip install requests pillow`
2. Run the script: `python compare_images.py image1.jpg image2.jpg`
3. Review the output for detailed comparison notes.

## Requirements
- Python 3.8+
- Ollama server running locally (for llama3 model) or API key for vision-enabled LLM.
- Image files in supported formats.

This tool streamlines visual comparison tasks, reducing manual effort and improving accuracy through AI-driven insights.

