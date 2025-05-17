# ComfyUI-HunyuanCustom-GGUF Quickstart Guide

This guide provides the fastest way to get started with ComfyUI-HunyuanCustom-GGUF.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ComfyUI-HunyuanCustom-GGUF.git
cd ComfyUI-HunyuanCustom-GGUF
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Install as a ComfyUI custom node:**
```bash
# If ComfyUI is installed in a different location, adjust the path accordingly
ln -s $(pwd) /path/to/ComfyUI/custom_nodes/ComfyUI-HunyuanCustom-GGUF
```

4. **Download required models:**
```bash
python -m comfyui_hunyuancustom_gguf.cli --download-models
```

## Usage

### Basic Audio Processing

1. Start ComfyUI:
```bash
cd /path/to/ComfyUI
python main.py
```

2. Load the sample workflow:
   - Open your browser and navigate to ComfyUI (typically at http://127.0.0.1:8188)
   - Click on the "Load" button
   - Select the workflow file from `workflows/audio/multi_audio_workflow.json`

3. Configure your settings:
   - Set the path to your audio file
   - Adjust processing parameters if needed
   - Click "Run" to process the audio

### Command Line Interface

Process a single audio file:
```bash
python -m comfyui_hunyuancustom_gguf.cli --model-path /path/to/model.gguf --audio-path /path/to/audio.wav
```

Process all audio files in a directory:
```bash
python -m comfyui_hunyuancustom_gguf.cli --model-path /path/to/model.gguf --audio-path /path/to/audio_dir
```

### Batch Processing

Use the provided script to process multiple audio files:
```bash
./examples/process_batch.sh --mode multi --max-files 5 --weight-scheme equal
```

## Available Models

### HunyuanCustom GGUF Models

Download your preferred model and place it in `ComfyUI/models/unet/`:
```bash
# Example for Q4_K_M quantization (recommended)
wget https://huggingface.co/YarvixPA/HunyuanCustom-gguf/resolve/main/hunyuan_video_custom_720p-Q4_K_M.gguf -O /path/to/ComfyUI/models/unet/hunyuan_video_custom_720p-Q4_K_M.gguf
```

## Next Steps

- Check the detailed documentation in `README.md` and `README_AUDIO.md`
- Explore example workflows in the `workflows/audio` directory
- Try the batch processing capabilities with your own audio files 