# Audio Processing Guide

This guide explains how to use the audio processing features of ComfyUI-GGUF.

## Features

- Audio file processing
- Stem separation using UVR5
- VR architecture for vocal processing
- Batch processing capabilities

## Audio Processing

### Supported Formats

- WAV (recommended)
- MP3
- FLAC
- OGG

### Audio Parameters

- Sample rate: 16000 Hz (automatic conversion)
- Channels: Mono (automatic conversion from stereo)
- Bit depth: 16-bit

## Stem Separation

The UVR5 processor can separate audio into vocals and instrumentals:

```python
from batch_process import UVR5Processor

# Initialize processor
processor = UVR5Processor()

# Separate stems
vocals, instrumentals = processor.separate_stems("input.wav")
```

### VR Architecture Features

The VR architecture provides additional vocal processing:

1. Echo Removal
2. Noise Reduction
3. Reverb Removal

Enable these features using the corresponding flags:

```bash
./process_batch.sh --de-echo --de-noise --de-reverb
```

## Batch Processing

Use the provided script to process multiple audio files:

```bash
./process_batch.sh --mode multi --max-files 5
```

### Processing Modes

1. Single Mode
   - Process one audio file at a time
   - Basic audio processing

2. Multi Mode
   - Process multiple audio files
   - Combine audio with weights
   - Stem separation
   - VR architecture features

### Weight Schemes

1. Equal Weights
   - All audio files have equal influence
   - Default option

2. Custom Weights
   - Specify weights for each audio file
   - Example: `--weight-scheme custom --weights 0.7 0.3`

## Output

Processed files are saved in the following structure:

```
output/
├── processed/
│   ├── vocals/
│   └── instrumentals/
├── stems/
│   ├── vocals/
│   └── instrumentals/
└── combined/
    └── mixed/
```

## Error Handling

The system provides comprehensive error handling:

1. Format Validation
   - File extension check
   - Audio metadata validation
   - Warnings for unexpected formats

2. Automatic Conversion
   - Sample rate adjustment
   - Stereo to mono conversion
   - Audio signal normalization

3. Error Messages
   - `AudioFormatError`: For invalid audio formats
   - `ValueError`: For invalid parameters
   - Detailed warnings for format deviations

## Examples

### 1. Basic Audio Processing

```python
from audio_utils import compute_audio_embedding

# Process single audio file
embedding = compute_audio_embedding("music.wav")
```

### 2. Weighted Multi-Audio Processing

```python
from audio_utils import compute_multi_audio_embedding

# Mix multiple audio files with different weights
mixed = compute_multi_audio_embedding(
    audio_paths=["speech.wav", "music.wav", "effects.wav"],
    weights=[0.5, 0.3, 0.2]
)
```

### 3. Batch Processing with CLI

```bash
# Process all WAV files in directory
./process_batch.sh \
  --mode multi \
  --max-files 5 \
  --weight-scheme equal \
  --separate-stems \
  --de-echo \
  --de-noise
```

## Development

### Testing

Run tests with:

```bash
python -m unittest tests/test_audio_utils.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License 