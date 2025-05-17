#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from itertools import combinations
import numpy as np
import torch
import torchaudio
import requests
from tqdm import tqdm
from uvr5.vr import AudioPre, AudioPreDeEcho
from uvr5.mdxnet import MDXNetDereverb
from uvr5.mdx import MDXNet

def download_file(url: str, output_path: str) -> None:
    """
    Downloads a file from a URL.
    
    Args:
        url: File URL
        output_path: Path to save the file
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_models(models_dir: str = "models", quantization: str = "Q4_K_M") -> None:
    """
    Downloads all required models.
    
    Args:
        models_dir: Directory to store models
        quantization: Quantization level (Q2_K, Q3_K_M, Q3_K_S, Q4_K_M, Q4_K_S, Q5_K_M, Q5_K_S, Q6_K)
    """
    # Model URLs
    model_urls = {
        # HunyuanCustom GGUF Models
        "hunyuan": {
            "url": f"https://huggingface.co/YarvixPA/HunyuanCustom-gguf/resolve/main/hunyuan_video_custom_720p-{quantization}.gguf",
            "path": os.path.join(models_dir, "hunyuan", f"hunyuan_video_custom_720p-{quantization}.gguf")
        },
        
        # UVR5 Models
        "mdx23c": {
            "url": "https://huggingface.co/spaces/Anjok07/ultimatevocalremovergui/resolve/main/models/MDX_Net_Models/UVR-MDX-NET-Voc_FT.onnx",
            "path": os.path.join(models_dir, "mdx23c_instvoc_hq_2", "model.onnx")
        },
        "echo": {
            "url": "https://huggingface.co/spaces/Anjok07/ultimatevocalremovergui/resolve/main/models/VR_Models/1_HP-UVR.pth",
            "path": os.path.join(models_dir, "echo", "model.pth")
        },
        "noise": {
            "url": "https://huggingface.co/spaces/Anjok07/ultimatevocalremovergui/resolve/main/models/VR_Models/2_HP-UVR.pth",
            "path": os.path.join(models_dir, "noise", "model.pth")
        },
        "reverb": {
            "url": "https://huggingface.co/spaces/Anjok07/ultimatevocalremovergui/resolve/main/models/MDX_Net_Models/UVR-MDX-NET-Voc_FT.onnx",
            "path": os.path.join(models_dir, "reverb", "model.onnx")
        }
    }
    
    # Create directories
    for model_info in model_urls.values():
        os.makedirs(os.path.dirname(model_info["path"]), exist_ok=True)
    
    # Download models
    for model_name, model_info in model_urls.items():
        if not os.path.exists(model_info["path"]):
            print(f"Downloading {model_name} model...")
            download_file(model_info["url"], model_info["path"])
            
            # Create config.json for UVR5 models
            if model_name in ["mdx23c", "echo", "noise", "reverb"]:
                config_path = os.path.join(os.path.dirname(model_info["path"]), "config.json")
                if not os.path.exists(config_path):
                    config = {
                        "model_type": model_name,
                        "model_path": model_info["path"],
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    }
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=4)

def calculate_weights(
    audio_files: List[str],
    weight_scheme: str = "equal",
    custom_weights: List[float] = None
) -> List[float]:
    """
    Calculates weights based on the chosen scheme.
    
    Args:
        audio_files: List of audio files
        weight_scheme: Weighting scheme ("equal", "linear", "exponential", "custom")
        custom_weights: User-defined weights for "custom" scheme
    
    Returns:
        List of normalized weights
    """
    n_files = len(audio_files)
    
    if weight_scheme == "equal":
        return [1.0 / n_files] * n_files
    
    elif weight_scheme == "linear":
        # Linear decrease: [0.5, 0.3, 0.2] for 3 files
        weights = np.linspace(1.0, 0.2, n_files)
        return list(weights / np.sum(weights))
    
    elif weight_scheme == "exponential":
        # Exponential decrease: [0.6, 0.3, 0.1] for 3 files
        weights = np.exp(-np.linspace(0, 2, n_files))
        return list(weights / np.sum(weights))
    
    elif weight_scheme == "custom":
        if custom_weights is None:
            raise ValueError("Custom weights must be provided for 'custom' scheme")
        if len(custom_weights) != n_files:
            raise ValueError(f"Number of weights ({len(custom_weights)}) does not match number of files ({n_files})")
        return list(np.array(custom_weights) / np.sum(custom_weights))
    
    else:
        raise ValueError(f"Unknown weighting scheme: {weight_scheme}")

def create_audio_weights(audio_files: List[str], weights: List[float]) -> Tuple[str, str]:
    """Creates a string with audio paths and weights."""
    return ",".join(audio_files), ",".join(map(str, weights))

def update_workflow_audio(
    workflow: Dict[str, Any],
    audio_paths: str,
    weights: str
) -> Dict[str, Any]:
    """Updates audio parameters in the workflow."""
    for node in workflow["nodes"]:
        if node["type"] == "MultiAudioLoader":
            node["inputs"][0]["value"] = audio_paths
            node["inputs"][1]["value"] = weights
    return workflow

class UVR5Processor:
    def __init__(
        self,
        model_path: str = "mdx23c_instvoc_hq_2",
        de_echo: bool = False,
        de_noise: bool = False,
        de_reverb: bool = False
    ):
        """
        Initializes the UVR5 processor with the specified model.
        
        Args:
            model_path: Path to MDX model
            de_echo: Whether to enable echo removal
            de_noise: Whether to enable noise reduction
            de_reverb: Whether to enable reverb removal
        """
        # Download models
        download_models()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # MDX for stem separation
        self.mdx_model = MDXNet(model_path)
        self.mdx_model.to(self.device)
        
        # VR architecture models
        self.de_echo = de_echo
        self.de_noise = de_noise
        self.de_reverb = de_reverb
        
        if de_echo:
            self.echo_model = AudioPreDeEcho()
            self.echo_model.to(self.device)
        
        if de_noise:
            self.noise_model = AudioPre()
            self.noise_model.to(self.device)
        
        if de_reverb:
            self.reverb_model = MDXNetDereverb()
            self.reverb_model.to(self.device)
    
    def process_vocals(self, vocals: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Processes vocals with enabled VR models.
        
        Args:
            vocals: Vocal tensor
            sr: Sample rate
            
        Returns:
            Processed vocal tensor
        """
        processed = vocals
        
        if self.de_echo:
            with torch.no_grad():
                processed = self.echo_model(processed.to(self.device))
        
        if self.de_noise:
            with torch.no_grad():
                processed = self.noise_model(processed.to(self.device))
        
        if self.de_reverb:
            with torch.no_grad():
                processed = self.reverb_model(processed.to(self.device))
        
        return processed.cpu()
    
    def separate_stems(
        self,
        audio_path: str,
        output_dir: str,
        process_vocals: bool = True
    ) -> Dict[str, str]:
        """
        Separates audio file into vocals and instrumentals.
        
        Args:
            audio_path: Path to audio file
            output_dir: Output directory
            process_vocals: Whether to process vocals
            
        Returns:
            Dictionary with paths to separated stems
        """
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:  # Convert stereo to mono
            audio = audio.mean(dim=0, keepdim=True)
        
        # Perform separation
        with torch.no_grad():
            vocals, inst = self.mdx_model.separate(audio.to(self.device))
        
        # Process vocals if requested
        if process_vocals and (self.de_echo or self.de_noise or self.de_reverb):
            vocals = self.process_vocals(vocals, sr)
        
        # Generate output paths
        base_name = Path(audio_path).stem
        vocals_path = os.path.join(output_dir, f"{base_name}_vocals.wav")
        inst_path = os.path.join(output_dir, f"{base_name}_instrumental.wav")
        
        # Save separated stems
        torchaudio.save(vocals_path, vocals.cpu(), sr)
        torchaudio.save(inst_path, inst.cpu(), sr)
        
        return {
            "vocals": vocals_path,
            "instrumental": inst_path
        }

def process_audio_combinations(
    workflow_path: str,
    audio_dir: str,
    output_dir: str,
    max_files: int = 3,
    weight_scheme: str = "equal",
    custom_weights: List[float] = None,
    stem_separator: str = "_",
    model_name: str = "hunyuan",
    separate_stems: bool = False,
    de_echo: bool = False,
    de_noise: bool = False,
    de_reverb: bool = False
) -> None:
    """
    Processes combinations of audio files with the multi-audio workflow.
    
    Args:
        workflow_path: Path to base workflow
        audio_dir: Directory with audio files
        output_dir: Output directory
        max_files: Maximum number of audio files per combination
        weight_scheme: Weighting scheme ("equal", "linear", "exponential", "custom")
        custom_weights: User-defined weights for "custom" scheme
        stem_separator: Separator for filenames (default: "_")
        model_name: Name of model to use (default: "hunyuan")
        separate_stems: Whether to separate audio files into stems
        de_echo: Whether to enable echo removal
        de_noise: Whether to enable noise reduction
        de_reverb: Whether to enable reverb removal
    """
    # Initialize UVR5 processor if needed
    uvr5 = UVR5Processor(
        de_echo=de_echo,
        de_noise=de_noise,
        de_reverb=de_reverb
    ) if separate_stems else None
    
    # Create directory for separated stems
    stems_dir = os.path.join(output_dir, "stems") if separate_stems else None
    if stems_dir:
        os.makedirs(stems_dir, exist_ok=True)
    
    # Load base workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all WAV files
    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    
    if not audio_files:
        print(f"No WAV files found in {audio_dir}")
        return
    
    # Process combinations of audio files
    for n in range(1, min(max_files + 1, len(audio_files) + 1)):
        for combo in combinations(audio_files, n):
            try:
                # Create audio paths
                audio_paths = []
                for audio_file in combo:
                    if separate_stems:
                        # Separate stems
                        stems = uvr5.separate_stems(str(audio_file), stems_dir)
                        audio_paths.extend([stems["vocals"], stems["instrumental"]])
                    else:
                        audio_paths.append(str(audio_file))
                
                # Calculate weights
                combo_weights = calculate_weights(
                    audio_paths,
                    weight_scheme,
                    custom_weights[:len(audio_paths)] if custom_weights else None
                )
                
                # Create strings for workflow
                audio_paths_str, weights_str = create_audio_weights(
                    audio_paths,
                    combo_weights
                )
                
                # Update workflow
                updated_workflow = update_workflow_audio(
                    workflow,
                    audio_paths_str,
                    weights_str
                )
                
                # Generate output name
                combo_name = stem_separator.join(f.stem for f in combo)
                stem_suffix = "_stems" if separate_stems else ""
                output_name = f"combo{stem_separator}{combo_name}{stem_separator}{weight_scheme}{stem_separator}{model_name}{stem_suffix}"
                output_path = os.path.join(output_dir, f"{output_name}.json")
                
                # Save adapted workflow
                with open(output_path, 'w') as f:
                    json.dump(updated_workflow, f, indent=4)
                
                print(f"Processed: {combo_name} ({weight_scheme}) -> {output_path}")
                
            except Exception as e:
                print(f"Error processing combination {combo}: {str(e)}")
                continue

def process_single_audio(
    workflow_path: str,
    audio_dir: str,
    output_dir: str,
    weight_scheme: str = "equal",
    custom_weights: List[float] = None,
    stem_separator: str = "_",
    model_name: str = "hunyuan",
    separate_stems: bool = False,
    de_echo: bool = False,
    de_noise: bool = False,
    de_reverb: bool = False
) -> None:
    """
    Processes single audio files with the multi-audio workflow.
    
    Args:
        workflow_path: Path to base workflow
        audio_dir: Directory with audio files
        output_dir: Output directory
        weight_scheme: Weighting scheme ("equal", "linear", "exponential", "custom")
        custom_weights: User-defined weights for "custom" scheme
        stem_separator: Separator for filenames (default: "_")
        model_name: Name of model to use (default: "hunyuan")
        separate_stems: Whether to separate audio files into stems
        de_echo: Whether to enable echo removal
        de_noise: Whether to enable noise reduction
        de_reverb: Whether to enable reverb removal
    """
    # Initialize UVR5 processor if needed
    uvr5 = UVR5Processor(
        de_echo=de_echo,
        de_noise=de_noise,
        de_reverb=de_reverb
    ) if separate_stems else None
    
    # Create directory for separated stems
    stems_dir = os.path.join(output_dir, "stems") if separate_stems else None
    if stems_dir:
        os.makedirs(stems_dir, exist_ok=True)
    
    # Load base workflow
    with open(workflow_path, 'r') as f:
        workflow = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all WAV files
    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    
    if not audio_files:
        print(f"No WAV files found in {audio_dir}")
        return
    
    # Process each audio file
    for audio_file in audio_files:
        try:
            # Create audio paths
            audio_paths = []
            if separate_stems:
                # Separate stems
                stems = uvr5.separate_stems(str(audio_file), stems_dir)
                audio_paths.extend([stems["vocals"], stems["instrumental"]])
            else:
                audio_paths.append(str(audio_file))
            
            # Calculate weights
            weights = calculate_weights(
                audio_paths,
                weight_scheme,
                custom_weights[:len(audio_paths)] if custom_weights else None
            )
            
            audio_paths_str, weight_str = create_audio_weights(
                audio_paths,
                weights
            )
            
            # Update workflow
            updated_workflow = update_workflow_audio(
                workflow,
                audio_paths_str,
                weight_str
            )
            
            # Generate output name
            stem_suffix = "_stems" if separate_stems else ""
            output_name = f"{audio_file.stem}{stem_separator}{weight_scheme}{stem_separator}{model_name}{stem_suffix}"
            output_path = os.path.join(output_dir, f"{output_name}.json")
            
            # Save adapted workflow
            with open(output_path, 'w') as f:
                json.dump(updated_workflow, f, indent=4)
            
            print(f"Processed: {audio_file.name} ({weight_scheme}) -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(
        description="Batch processing for multi-audio conditioning"
    )
    
    parser.add_argument(
        "--workflow",
        required=True,
        help="Path to multi-audio workflow"
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        help="Directory with audio files"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--weight-scheme",
        choices=["equal", "linear", "exponential", "custom"],
        default="equal",
        help="Weighting scheme for audio files"
    )
    parser.add_argument(
        "--custom-weights",
        type=float,
        nargs="+",
        help="User-defined weights for 'custom' scheme"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "combinations"],
        default="single",
        help="Processing mode: single or combinations"
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=3,
        help="Maximum number of audio files per combination"
    )
    parser.add_argument(
        "--stem-separator",
        default="_",
        help="Separator for filenames (default: _)"
    )
    parser.add_argument(
        "--model-name",
        default="hunyuan",
        help="Name of model to use (default: hunyuan)"
    )
    parser.add_argument(
        "--separate-stems",
        action="store_true",
        help="Separate audio files into vocals and instrumentals"
    )
    parser.add_argument(
        "--de-echo",
        action="store_true",
        help="Enable echo removal for vocals"
    )
    parser.add_argument(
        "--de-noise",
        action="store_true",
        help="Enable noise reduction for vocals"
    )
    parser.add_argument(
        "--de-reverb",
        action="store_true",
        help="Enable reverb removal for vocals"
    )
    
    args = parser.parse_args()
    
    if args.mode == "single":
        process_single_audio(
            workflow_path=args.workflow,
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            weight_scheme=args.weight_scheme,
            custom_weights=args.custom_weights,
            stem_separator=args.stem_separator,
            model_name=args.model_name,
            separate_stems=args.separate_stems,
            de_echo=args.de_echo,
            de_noise=args.de_noise,
            de_reverb=args.de_reverb
        )
    else:
        process_audio_combinations(
            workflow_path=args.workflow,
            audio_dir=args.audio_dir,
            output_dir=args.output_dir,
            max_files=args.max_files,
            weight_scheme=args.weight_scheme,
            custom_weights=args.custom_weights,
            stem_separator=args.stem_separator,
            model_name=args.model_name,
            separate_stems=args.separate_stems,
            de_echo=args.de_echo,
            de_noise=args.de_noise,
            de_reverb=args.de_reverb
        )

if __name__ == "__main__":
    main() 