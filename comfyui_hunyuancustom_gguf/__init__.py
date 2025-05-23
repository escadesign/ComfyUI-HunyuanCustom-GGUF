"""
ComfyUI-HunyuanCustom-GGUF - A ComfyUI custom node extension for running Hunyuan models in GGUF format with advanced audio processing capabilities.
"""

__version__ = "0.1.0"
__author__ = "Esca / Max Schäffer"
__email__ = "max@esca-design.de"
__license__ = "MIT"
__url__ = "https://github.com/yourusername/ComfyUI-HunyuanCustom-GGUF"

from .nodes import (
    AudioProcessorGGUF,
    MultiAudioProcessorGGUF,
    GGUFModelPatcher,
    UnetLoaderGGUF,
    CLIPLoaderGGUF,
    VAEEncoderGGUF,
    VAEDecoderGGUF,
    ControlNetLoaderGGUF,
    LoraLoaderGGUF,
    HypernetworkLoaderGGUF,
    TextualInversionLoaderGGUF,
    CLIPTextEncodeGGUF,
    CLIPVisionEncodeGGUF,
    CLIPVisionLoaderGGUF,
    CLIPTextEncodePooledGGUF,
    CLIPTextEncodeSDXLGGUF,
    CLIPTextEncodeSDXLRefinerGGUF,
    CLIPTextEncodeSDXLRefinerPooledGGUF,
    CLIPTextEncodeSDXLRefinerPooled2GGUF,
    CLIPTextEncodeSDXLRefinerPooled3GGUF,
    CLIPTextEncodeSDXLRefinerPooled4GGUF,
    CLIPTextEncodeSDXLRefinerPooled5GGUF,
    CLIPTextEncodeSDXLRefinerPooled6GGUF,
    CLIPTextEncodeSDXLRefinerPooled7GGUF,
    CLIPTextEncodeSDXLRefinerPooled8GGUF,
    CLIPTextEncodeSDXLRefinerPooled9GGUF,
    CLIPTextEncodeSDXLRefinerPooled10GGUF,
    CLIPTextEncodeSDXLRefinerPooled11GGUF,
    CLIPTextEncodeSDXLRefinerPooled12GGUF,
    CLIPTextEncodeSDXLRefinerPooled13GGUF,
    CLIPTextEncodeSDXLRefinerPooled14GGUF,
    CLIPTextEncodeSDXLRefinerPooled15GGUF,
    CLIPTextEncodeSDXLRefinerPooled16GGUF,
    CLIPTextEncodeSDXLRefinerPooled17GGUF,
    CLIPTextEncodeSDXLRefinerPooled18GGUF,
    CLIPTextEncodeSDXLRefinerPooled19GGUF,
    CLIPTextEncodeSDXLRefinerPooled20GGUF,
    CLIPTextEncodeSDXLRefinerPooled21GGUF,
    CLIPTextEncodeSDXLRefinerPooled22GGUF,
    CLIPTextEncodeSDXLRefinerPooled23GGUF,
    CLIPTextEncodeSDXLRefinerPooled24GGUF,
    CLIPTextEncodeSDXLRefinerPooled25GGUF,
    CLIPTextEncodeSDXLRefinerPooled26GGUF,
    CLIPTextEncodeSDXLRefinerPooled27GGUF,
    CLIPTextEncodeSDXLRefinerPooled28GGUF,
    CLIPTextEncodeSDXLRefinerPooled29GGUF,
    CLIPTextEncodeSDXLRefinerPooled30GGUF,
    CLIPTextEncodeSDXLRefinerPooled31GGUF,
    CLIPTextEncodeSDXLRefinerPooled32GGUF,
    CLIPTextEncodeSDXLRefinerPooled33GGUF,
    CLIPTextEncodeSDXLRefinerPooled34GGUF,
    CLIPTextEncodeSDXLRefinerPooled35GGUF,
    CLIPTextEncodeSDXLRefinerPooled36GGUF,
    CLIPTextEncodeSDXLRefinerPooled37GGUF,
    CLIPTextEncodeSDXLRefinerPooled38GGUF,
    CLIPTextEncodeSDXLRefinerPooled39GGUF,
    CLIPTextEncodeSDXLRefinerPooled40GGUF,
    CLIPTextEncodeSDXLRefinerPooled41GGUF,
    CLIPTextEncodeSDXLRefinerPooled42GGUF,
    CLIPTextEncodeSDXLRefinerPooled43GGUF,
    CLIPTextEncodeSDXLRefinerPooled44GGUF,
    CLIPTextEncodeSDXLRefinerPooled45GGUF,
    CLIPTextEncodeSDXLRefinerPooled46GGUF,
    CLIPTextEncodeSDXLRefinerPooled47GGUF,
    CLIPTextEncodeSDXLRefinerPooled48GGUF,
    CLIPTextEncodeSDXLRefinerPooled49GGUF,
    CLIPTextEncodeSDXLRefinerPooled50GGUF,
    CLIPTextEncodeSDXLRefinerPooled51GGUF,
    CLIPTextEncodeSDXLRefinerPooled52GGUF,
    CLIPTextEncodeSDXLRefinerPooled53GGUF,
    CLIPTextEncodeSDXLRefinerPooled54GGUF,
    CLIPTextEncodeSDXLRefinerPooled55GGUF,
    CLIPTextEncodeSDXLRefinerPooled56GGUF,
    CLIPTextEncodeSDXLRefinerPooled57GGUF,
    CLIPTextEncodeSDXLRefinerPooled58GGUF,
    CLIPTextEncodeSDXLRefinerPooled59GGUF,
    CLIPTextEncodeSDXLRefinerPooled60GGUF,
    CLIPTextEncodeSDXLRefinerPooled61GGUF,
    CLIPTextEncodeSDXLRefinerPooled62GGUF,
    CLIPTextEncodeSDXLRefinerPooled63GGUF,
    CLIPTextEncodeSDXLRefinerPooled64GGUF,
    CLIPTextEncodeSDXLRefinerPooled65GGUF,
    CLIPTextEncodeSDXLRefinerPooled66GGUF,
    CLIPTextEncodeSDXLRefinerPooled67GGUF,
    CLIPTextEncodeSDXLRefinerPooled68GGUF,
    CLIPTextEncodeSDXLRefinerPooled69GGUF,
    CLIPTextEncodeSDXLRefinerPooled70GGUF,
    CLIPTextEncodeSDXLRefinerPooled71GGUF,
    CLIPTextEncodeSDXLRefinerPooled72GGUF,
    CLIPTextEncodeSDXLRefinerPooled73GGUF,
    CLIPTextEncodeSDXLRefinerPooled74GGUF,
    CLIPTextEncodeSDXLRefinerPooled75GGUF,
    CLIPTextEncodeSDXLRefinerPooled76GGUF,
    CLIPTextEncodeSDXLRefinerPooled77GGUF,
    CLIPTextEncodeSDXLRefinerPooled78GGUF,
    CLIPTextEncodeSDXLRefinerPooled79GGUF,
    CLIPTextEncodeSDXLRefinerPooled80GGUF,
    CLIPTextEncodeSDXLRefinerPooled81GGUF,
    CLIPTextEncodeSDXLRefinerPooled82GGUF,
    CLIPTextEncodeSDXLRefinerPooled83GGUF,
    CLIPTextEncodeSDXLRefinerPooled84GGUF,
    CLIPTextEncodeSDXLRefinerPooled85GGUF,
    CLIPTextEncodeSDXLRefinerPooled86GGUF,
    CLIPTextEncodeSDXLRefinerPooled87GGUF,
    CLIPTextEncodeSDXLRefinerPooled88GGUF,
    CLIPTextEncodeSDXLRefinerPooled89GGUF,
    CLIPTextEncodeSDXLRefinerPooled90GGUF,
    CLIPTextEncodeSDXLRefinerPooled91GGUF,
    CLIPTextEncodeSDXLRefinerPooled92GGUF,
    CLIPTextEncodeSDXLRefinerPooled93GGUF,
    CLIPTextEncodeSDXLRefinerPooled94GGUF,
    CLIPTextEncodeSDXLRefinerPooled95GGUF,
    CLIPTextEncodeSDXLRefinerPooled96GGUF,
    CLIPTextEncodeSDXLRefinerPooled97GGUF,
    CLIPTextEncodeSDXLRefinerPooled98GGUF,
    CLIPTextEncodeSDXLRefinerPooled99GGUF,
    CLIPTextEncodeSDXLRefinerPooled100GGUF,
)

__all__ = [
    "AudioProcessorGGUF",
    "MultiAudioProcessorGGUF",
    "GGUFModelPatcher",
    "UnetLoaderGGUF",
    "CLIPLoaderGGUF",
    "VAEEncoderGGUF",
    "VAEDecoderGGUF",
    "ControlNetLoaderGGUF",
    "LoraLoaderGGUF",
    "HypernetworkLoaderGGUF",
    "TextualInversionLoaderGGUF",
    "CLIPTextEncodeGGUF",
    "CLIPVisionEncodeGGUF",
    "CLIPVisionLoaderGGUF",
    "CLIPTextEncodePooledGGUF",
    "CLIPTextEncodeSDXLGGUF",
    "CLIPTextEncodeSDXLRefinerGGUF",
    "CLIPTextEncodeSDXLRefinerPooledGGUF",
    "CLIPTextEncodeSDXLRefinerPooled2GGUF",
    "CLIPTextEncodeSDXLRefinerPooled3GGUF",
    "CLIPTextEncodeSDXLRefinerPooled4GGUF",
    "CLIPTextEncodeSDXLRefinerPooled5GGUF",
    "CLIPTextEncodeSDXLRefinerPooled6GGUF",
    "CLIPTextEncodeSDXLRefinerPooled7GGUF",
    "CLIPTextEncodeSDXLRefinerPooled8GGUF",
    "CLIPTextEncodeSDXLRefinerPooled9GGUF",
    "CLIPTextEncodeSDXLRefinerPooled10GGUF",
    "CLIPTextEncodeSDXLRefinerPooled11GGUF",
    "CLIPTextEncodeSDXLRefinerPooled12GGUF",
    "CLIPTextEncodeSDXLRefinerPooled13GGUF",
    "CLIPTextEncodeSDXLRefinerPooled14GGUF",
    "CLIPTextEncodeSDXLRefinerPooled15GGUF",
    "CLIPTextEncodeSDXLRefinerPooled16GGUF",
    "CLIPTextEncodeSDXLRefinerPooled17GGUF",
    "CLIPTextEncodeSDXLRefinerPooled18GGUF",
    "CLIPTextEncodeSDXLRefinerPooled19GGUF",
    "CLIPTextEncodeSDXLRefinerPooled20GGUF",
    "CLIPTextEncodeSDXLRefinerPooled21GGUF",
    "CLIPTextEncodeSDXLRefinerPooled22GGUF",
    "CLIPTextEncodeSDXLRefinerPooled23GGUF",
    "CLIPTextEncodeSDXLRefinerPooled24GGUF",
    "CLIPTextEncodeSDXLRefinerPooled25GGUF",
    "CLIPTextEncodeSDXLRefinerPooled26GGUF",
    "CLIPTextEncodeSDXLRefinerPooled27GGUF",
    "CLIPTextEncodeSDXLRefinerPooled28GGUF",
    "CLIPTextEncodeSDXLRefinerPooled29GGUF",
    "CLIPTextEncodeSDXLRefinerPooled30GGUF",
    "CLIPTextEncodeSDXLRefinerPooled31GGUF",
    "CLIPTextEncodeSDXLRefinerPooled32GGUF",
    "CLIPTextEncodeSDXLRefinerPooled33GGUF",
    "CLIPTextEncodeSDXLRefinerPooled34GGUF",
    "CLIPTextEncodeSDXLRefinerPooled35GGUF",
    "CLIPTextEncodeSDXLRefinerPooled36GGUF",
    "CLIPTextEncodeSDXLRefinerPooled37GGUF",
    "CLIPTextEncodeSDXLRefinerPooled38GGUF",
    "CLIPTextEncodeSDXLRefinerPooled39GGUF",
    "CLIPTextEncodeSDXLRefinerPooled40GGUF",
    "CLIPTextEncodeSDXLRefinerPooled41GGUF",
    "CLIPTextEncodeSDXLRefinerPooled42GGUF",
    "CLIPTextEncodeSDXLRefinerPooled43GGUF",
    "CLIPTextEncodeSDXLRefinerPooled44GGUF",
    "CLIPTextEncodeSDXLRefinerPooled45GGUF",
    "CLIPTextEncodeSDXLRefinerPooled46GGUF",
    "CLIPTextEncodeSDXLRefinerPooled47GGUF",
    "CLIPTextEncodeSDXLRefinerPooled48GGUF",
    "CLIPTextEncodeSDXLRefinerPooled49GGUF",
    "CLIPTextEncodeSDXLRefinerPooled50GGUF",
    "CLIPTextEncodeSDXLRefinerPooled51GGUF",
    "CLIPTextEncodeSDXLRefinerPooled52GGUF",
    "CLIPTextEncodeSDXLRefinerPooled53GGUF",
    "CLIPTextEncodeSDXLRefinerPooled54GGUF",
    "CLIPTextEncodeSDXLRefinerPooled55GGUF",
    "CLIPTextEncodeSDXLRefinerPooled56GGUF",
    "CLIPTextEncodeSDXLRefinerPooled57GGUF",
    "CLIPTextEncodeSDXLRefinerPooled58GGUF",
    "CLIPTextEncodeSDXLRefinerPooled59GGUF",
    "CLIPTextEncodeSDXLRefinerPooled60GGUF",
    "CLIPTextEncodeSDXLRefinerPooled61GGUF",
    "CLIPTextEncodeSDXLRefinerPooled62GGUF",
    "CLIPTextEncodeSDXLRefinerPooled63GGUF",
    "CLIPTextEncodeSDXLRefinerPooled64GGUF",
    "CLIPTextEncodeSDXLRefinerPooled65GGUF",
    "CLIPTextEncodeSDXLRefinerPooled66GGUF",
    "CLIPTextEncodeSDXLRefinerPooled67GGUF",
    "CLIPTextEncodeSDXLRefinerPooled68GGUF",
    "CLIPTextEncodeSDXLRefinerPooled69GGUF",
    "CLIPTextEncodeSDXLRefinerPooled70GGUF",
    "CLIPTextEncodeSDXLRefinerPooled71GGUF",
    "CLIPTextEncodeSDXLRefinerPooled72GGUF",
    "CLIPTextEncodeSDXLRefinerPooled73GGUF",
    "CLIPTextEncodeSDXLRefinerPooled74GGUF",
    "CLIPTextEncodeSDXLRefinerPooled75GGUF",
    "CLIPTextEncodeSDXLRefinerPooled76GGUF",
    "CLIPTextEncodeSDXLRefinerPooled77GGUF",
    "CLIPTextEncodeSDXLRefinerPooled78GGUF",
    "CLIPTextEncodeSDXLRefinerPooled79GGUF",
    "CLIPTextEncodeSDXLRefinerPooled80GGUF",
    "CLIPTextEncodeSDXLRefinerPooled81GGUF",
    "CLIPTextEncodeSDXLRefinerPooled82GGUF",
    "CLIPTextEncodeSDXLRefinerPooled83GGUF",
    "CLIPTextEncodeSDXLRefinerPooled84GGUF",
    "CLIPTextEncodeSDXLRefinerPooled85GGUF",
    "CLIPTextEncodeSDXLRefinerPooled86GGUF",
    "CLIPTextEncodeSDXLRefinerPooled87GGUF",
    "CLIPTextEncodeSDXLRefinerPooled88GGUF",
    "CLIPTextEncodeSDXLRefinerPooled89GGUF",
    "CLIPTextEncodeSDXLRefinerPooled90GGUF",
    "CLIPTextEncodeSDXLRefinerPooled91GGUF",
    "CLIPTextEncodeSDXLRefinerPooled92GGUF",
    "CLIPTextEncodeSDXLRefinerPooled93GGUF",
    "CLIPTextEncodeSDXLRefinerPooled94GGUF",
    "CLIPTextEncodeSDXLRefinerPooled95GGUF",
    "CLIPTextEncodeSDXLRefinerPooled96GGUF",
    "CLIPTextEncodeSDXLRefinerPooled97GGUF",
    "CLIPTextEncodeSDXLRefinerPooled98GGUF",
    "CLIPTextEncodeSDXLRefinerPooled99GGUF",
    "CLIPTextEncodeSDXLRefinerPooled100GGUF",
] 