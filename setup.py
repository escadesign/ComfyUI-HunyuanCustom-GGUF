from setuptools import setup

setup(
    name="comfyui_hunyuancustom_gguf",
    version="0.1.0",
    packages=["comfyui_hunyuancustom_gguf"],
    package_dir={"": "."},
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "safetensors",
        "tqdm",
        "requests",
        "huggingface_hub",
        "einops",
        "accelerate",
        "bitsandbytes",
        "scipy",
        "soundfile",
        "librosa",
        "matplotlib",
        "pydub",
    ],
)
