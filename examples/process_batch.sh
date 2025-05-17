#!/bin/bash

# Define directories
WORKFLOW_DIR="./workflows"
AUDIO_DIR="./audio"
OUTPUT_DIR="./output"

# Check if audio files exist
if [ ! -d "$AUDIO_DIR" ] || [ -z "$(ls -A $AUDIO_DIR/*.wav 2>/dev/null)" ]; then
    echo "Error: No .wav files found in $AUDIO_DIR"
    exit 1
fi

# Default parameters
MODE="multi"  # or "single"
MAX_FILES=5
WEIGHT_SCHEME="equal"  # or "custom"
STEM_SEPARATOR="_"
MODEL_NAME="hunyuan-v1.0.gguf"
SEPARATE_STEMS=false
DE_ECHO=false
DE_NOISE=false
DE_REVERB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        --weight-scheme)
            WEIGHT_SCHEME="$2"
            shift 2
            ;;
        --stem-separator)
            STEM_SEPARATOR="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --separate-stems)
            SEPARATE_STEMS=true
            shift
            ;;
        --de-echo)
            DE_ECHO=true
            shift
            ;;
        --de-noise)
            DE_NOISE=true
            shift
            ;;
        --de-reverb)
            DE_REVERB=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Validate weight scheme
case $WEIGHT_SCHEME in
    "equal"|"custom")
        ;;
    *)
        echo "Error: Invalid weight scheme. Use 'equal' or 'custom'"
        exit 1
        ;;
esac

# Set VR architecture flags
VR_FLAGS=""
if [ "$DE_ECHO" = true ]; then
    VR_FLAGS="$VR_FLAGS --de-echo"
fi
if [ "$DE_NOISE" = true ]; then
    VR_FLAGS="$VR_FLAGS --de-noise"
fi
if [ "$DE_REVERB" = true ]; then
    VR_FLAGS="$VR_FLAGS --de-reverb"
fi

# Set stem separation flag
STEM_FLAG=""
if [ "$SEPARATE_STEMS" = true ]; then
    STEM_FLAG="--separate-stems"
fi

# Execute Python script
python batch_process.py \
    --workflow-dir "$WORKFLOW_DIR" \
    --audio-dir "$AUDIO_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --mode "$MODE" \
    --max-files "$MAX_FILES" \
    --weight-scheme "$WEIGHT_SCHEME" \
    --stem-separator "$STEM_SEPARATOR" \
    --model-name "$MODEL_NAME" \
    $STEM_FLAG \
    $VR_FLAGS

# Print summary
echo "Processing completed with the following settings:"
echo "Mode: $MODE"
echo "Weight scheme: $WEIGHT_SCHEME"
echo "Stem separator: $STEM_SEPARATOR"
echo "Model: $MODEL_NAME"
echo "Stem separation: $SEPARATE_STEMS"
echo "VR architecture features:"
echo "  - Echo removal: $DE_ECHO"
echo "  - Noise reduction: $DE_NOISE"
echo "  - Reverb removal: $DE_REVERB" 