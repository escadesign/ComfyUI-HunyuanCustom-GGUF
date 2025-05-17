# ComfyUI-HunyuanCustom-GGUF 🏞️ + 🎤 = 🎥

A ComfyUI custom node extension for running Hunyuan models in GGUF format with advanced audio processing capabilities.

## Features

- Support for Hunyuan models in GGUF format
- Audio processing capabilities
- Stem separation using UVR5
- VR architecture for vocal processing
- Batch processing audio files

## Project Overview

ComfyUI-HunyuanCustom-GGUF is a custom node extension for ComfyUI that enables the use of Hunyuan models in GGUF format, along with advanced audio processing capabilities. Here's a detailed overview of the project's functionality:

### Core Functionality

- **Custom Nodes for ComfyUI**:  
  The extension provides a set of custom nodes that integrate seamlessly with ComfyUI's workflow system. These nodes handle the loading and processing of Hunyuan models, audio files, and other inputs.

- **Hunyuan Model Integration**:  
  The project supports Hunyuan models in GGUF format, which are optimized for text, image, and audio generation. These models are loaded using custom nodes like `UnetLoaderGGUF`, `CLIPLoaderGGUF`, and others, ensuring efficient and flexible usage within ComfyUI.

- **Audio Processing**:  
  The extension includes advanced audio processing capabilities:
  - **Stem Separation**: Using UVR5 models, the extension can separate audio into vocals and instrumentals.
  - **Vocal Processing**: The VR architecture models enable echo removal, noise reduction, and reverb removal, enhancing the quality of audio outputs.
  - **Batch Processing**: Users can process multiple audio files simultaneously, with options to mix and weight the outputs.

### Workflow Integration

- **Workflow Management**:  
  Users can load predefined workflows from the `workflows` directory, which are designed to showcase the capabilities of the custom nodes. These workflows can be customized to suit specific needs.

- **CLI and Scripting**:  
  The project includes a command-line interface (`cli.py`) and batch processing scripts (`batch_process.py`, `process_batch.sh`), allowing users to automate and streamline their audio processing tasks.

### Technical Implementation

- **Modular Design**:  
  The project is structured as a Python package, with clear separation of concerns:
  - `nodes.py`: Defines the custom nodes for ComfyUI.
  - `audio_utils.py`: Provides utilities for audio file handling and processing.
  - `cli.py`: Implements the command-line interface for batch processing.
  - `__init__.py`: Exports the necessary classes and functions for easy integration.

- **Dependencies**:  
  The project relies on a set of Python libraries, including `torch`, `librosa`, `soundfile`, and others, which are listed in `requirements.txt`.

### Usage

- **Installation**:  
  The extension is installed as a custom node in ComfyUI, with detailed instructions provided in the README.

- **Running Workflows**:  
  Users can start ComfyUI, load a workflow, configure settings, and run the workflow to generate outputs.

- **Batch Processing**:  
  The provided scripts allow users to process multiple audio files in batch mode, with options to customize the processing parameters.

### Conclusion

ComfyUI-HunyuanCustom-GGUF is a powerful and flexible extension that enhances ComfyUI's capabilities with advanced audio processing and model integration. It is designed to be user-friendly, modular, and extensible, making it a valuable tool for developers and users working with audio and model generation.

## Installation

1. **Clone this repository:**
```bash
git clone https://github.com/yourusername/ComfyUI-HunyuanCustom-GGUF.git
cd ComfyUI-HunyuanCustom-GGUF
```

2. **Activate environment & install dependencies:**

bsh/zsh:  
```bash
source env/bin/activate
```
fish:
```bash
source env/bin/activate.fish
```
csh/tcsh:
```bash
source env/bin/activate.csh
```
pwsh:
```bash
env/bin/Activate.ps1
```
cmd.exe:
```bash
env\Scripts\activate.bat
```
PowerShell:
```bash
env\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

3. **Install as a ComfyUI custom node:**
- Copy or symlink this folder into your ComfyUI `custom_nodes/` directory:
```bash
ln -s /path/to/ComfyUI-HunyuanCustom-GGUF /path/to/ComfyUI/custom_nodes/ComfyUI-HunyuanCustom-GGUF
```

4. **Download required models:**
```bash
python examples/batch_process.py --download-models
```

## Usage

### Basic Usage

1. Start ComfyUI:
```bash
python main.py
```
2. Load a workflow from the `workflows` directory
3. Configure your settings
4. Run the workflow

### Batch Processing

Use the provided script to process multiple audio files:

```bash
./examples/process_batch.sh --mode multi --max-files 5 --weight-scheme equal
```

Available options:
- `--mode`: Processing mode (single/multi)
- `--max-files`: Maximum number of files to process
- `--weight-scheme`: Weighting scheme (equal/custom)
- `--stem-separator`: Separator for stem names
- `--model-name`: Model to use
- `--separate-stems`: Enable stem separation
- `--de-echo`: Enable echo removal
- `--de-noise`: Enable noise reduction
- `--de-reverb`: Enable reverb removal

## Models

### HunyuanCustom GGUF Models

The HunyuanCustom-GGUF models are available on Hugging Face:
[https://huggingface.co/YarvixPA/HunyuanCustom-gguf/tree/main](https://huggingface.co/YarvixPA/HunyuanCustom-gguf/tree/main)

**Available Quantizations:**
- `hunyuan_video_custom_720p-Q2_K.gguf` (4.8 GB) – fastest, lower quality
- `hunyuan_video_custom_720p-Q3_K_M.gguf` (6.22 GB) – good compromise
- `hunyuan_video_custom_720p-Q4_K_M.gguf` (7.86 GB) – recommended
- `hunyuan_video_custom_720p-Q5_K_M.gguf` (9.43 GB) – high quality
- `hunyuan_video_custom_720p-Q6_K.gguf` (10.9 GB) – maximum quality

**Installation:**
Download your preferred model and place it in:
`ComfyUI/models/unet/`

Example:
```bash
wget https://huggingface.co/YarvixPA/HunyuanCustom-gguf/resolve/main/hunyuan_video_custom_720p-Q4_K_M.gguf -O /path/to/ComfyUI/models/unet/hunyuan_video_custom_720p-Q4_K_M.gguf
```

**Note:** No separate tokenizer model is required; everything is included in the GGUF file.

### UVR5/VR Architecture Models (for audio stem separation and vocal processing)

These models are required for advanced audio processing (stem separation, echo/noise/reverb removal):

- `models/uvr5/mdx23c_instvoc_hq_2/model.onnx`  
  → MDX23C: Model for stem separation (vocals/instrumentals)
- `models/uvr5/echo/model.pth`  
  → VR Architecture: Echo removal
- `models/uvr5/noise/model.pth`  
  → VR Architecture: Noise reduction
- `models/uvr5/reverb/model.onnx`  
  → VR Architecture: Reverb removal

**Installation:**
- These models are automatically downloaded with:
```bash
python examples/batch_process.py --download-models
```
- Or you can place them manually in the specified folders as shown above.

## Directory Structure

```
ComfyUI-HunyuanCustom-GGUF/
├── examples/
│   ├── batch_process.py
│   └── process_batch.sh
├── workflows/
│   ├── multi_audio_workflow.json
│   └── single_audio_workflow.json
├── models/
│   └── uvr5/
├── audio/
├── output/
└── requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

This project builds upon the following open-source repositories:

- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)  
  → The original ComfyUI framework, which provides the base for custom nodes and workflows.

- [HunyuanCustom-gguf](https://huggingface.co/YarvixPA/HunyuanCustom-gguf)  
  → The source of the HunyuanCustom-GGUF models used for text, image, and audio generation.

- [UVR5](https://github.com/Anjok07/ultimatevocalremovergui)  
  → The Ultimate Vocal Remover GUI, which provides the stem separation and vocal processing models.

- [VR Architecture](https://github.com/Anjok07/ultimatevocalremovergui/tree/master/lib_v5)  
  → The VR architecture models for echo removal, noise reduction, and reverb removal.

We are grateful to the developers and contributors of these projects for their work and support.
