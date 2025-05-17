import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple, Optional, Dict, List
import warnings
import os

class AudioFormatError(Exception):
    """Exception raised for invalid audio formats."""
    pass

def validate_audio_file(audio_path: str) -> Dict[str, Any]:
    """
    Validates an audio file and returns metadata.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary containing metadata (sample_rate, channels, duration)
        
    Raises:
        AudioFormatError: For invalid formats
    """
    if not os.path.exists(audio_path):
        raise AudioFormatError(f"File not found: {audio_path}")
        
    if not audio_path.lower().endswith('.wav'):
        warnings.warn(f"Unexpected file format: {audio_path}. WAV is recommended.")
    
    try:
        info = torchaudio.info(audio_path)
        return {
            "sample_rate": info.sample_rate,
            "channels": info.num_channels,
            "duration": info.num_frames / info.sample_rate
        }
    except Exception as e:
        raise AudioFormatError(f"Error reading audio file: {str(e)}")

def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Loads an audio file and converts it to the desired format.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate (default: 16000 Hz)
        
    Returns:
        Tuple containing (audio tensor, sampling rate)
        
    Raises:
        AudioFormatError: For invalid formats
    """
    # Validate audio file
    metadata = validate_audio_file(audio_path)
    
    # Warn if sample rate differs
    if metadata["sample_rate"] != target_sr:
        warnings.warn(
            f"Unexpected sample rate: {metadata['sample_rate']}Hz. "
            "Automatic conversion will be performed."
        )
    
    # Warn if stereo
    if metadata["channels"] > 1:
        warnings.warn(
            f"Stereo audio detected ({metadata['channels']} channels). "
            "Conversion to mono will be performed."
        )
    
    try:
        # Load audio with librosa for better compatibility
        audio, sr = librosa.load(
            audio_path,
            sr=target_sr,
            mono=True,
            res_type='kaiser_fast'  # Faster resampling method
        )
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Add batch and channel dimensions
        audio_tensor = audio_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        
        return audio_tensor, target_sr
        
    except Exception as e:
        raise AudioFormatError(f"Error loading audio file: {str(e)}")

def compute_mel_spectrogram(
    audio: torch.Tensor,
    sr: int,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    fmin: int = 0,
    fmax: Optional[int] = None
) -> torch.Tensor:
    """
    Computes mel spectrogram from audio signal.
    
    Args:
        audio: Audio tensor
        sr: Sampling rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        fmin: Minimum frequency for mel filterbank
        fmax: Maximum frequency for mel filterbank
        
    Returns:
        Mel spectrogram tensor
    """
    # Compute STFT
    stft = torch.stft(
        audio.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length).to(audio.device),
        return_complex=True
    )
    
    # Convert to power spectrogram
    power_spec = torch.abs(stft) ** 2
    
    # Create mel filterbank
    mel_filters = torch.from_numpy(
        librosa.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax
        )
    ).to(audio.device)
    
    # Apply mel filterbank
    mel_spec = torch.matmul(mel_filters, power_spec)
    
    # Convert to log scale
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    return mel_spec

def compute_audio_embedding(
    audio_path: str,
    target_sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024
) -> torch.Tensor:
    """
    Computes audio embeddings from an audio file.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sampling rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        
    Returns:
        Audio embedding tensor
        
    Raises:
        AudioFormatError: For invalid formats
    """
    try:
        # Load and preprocess audio
        audio, sr = load_audio(audio_path, target_sr)
        
        # Compute mel spectrogram
        mel_spec = compute_mel_spectrogram(
            audio,
            sr=target_sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
        
        return mel_spec
        
    except Exception as e:
        raise AudioFormatError(f"Error in audio processing: {str(e)}")

def mix_audio_embeddings(
    embeddings: List[torch.Tensor],
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    Mixes multiple audio embeddings.
    
    Args:
        embeddings: List of embedding tensors
        weights: Optional list of weights for each embedding
        
    Returns:
        Mixed embedding tensor
        
    Raises:
        ValueError: For invalid weights or shapes
    """
    if not embeddings:
        raise ValueError("No embeddings to mix")
    
    # Normalize weights
    if weights is None:
        weights = [1.0 / len(embeddings)] * len(embeddings)
    else:
        if len(weights) != len(embeddings):
            raise ValueError("Number of weights must match number of embeddings")
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
    
    # Ensure all embeddings have the same shape
    target_shape = embeddings[0].shape
    for i, emb in enumerate(embeddings[1:], 1):
        if emb.shape != target_shape:
            raise ValueError(f"Embedding {i} has a different shape: {emb.shape} vs {target_shape}")
    
    # Mix embeddings
    mixed = torch.zeros_like(embeddings[0])
    for emb, weight in zip(embeddings, weights):
        mixed += emb * weight
    
    return mixed

def compute_multi_audio_embedding(
    audio_paths: List[str],
    weights: Optional[List[float]] = None,
    target_sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024
) -> torch.Tensor:
    """
    Computes a mixed audio embedding from multiple audio files.
    
    Args:
        audio_paths: List of paths to audio files
        weights: Optional list of weights for each audio file
        target_sr: Target sampling rate
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length for STFT
        win_length: Window length for STFT
        
    Returns:
        Mixed audio embedding tensor
        
    Raises:
        AudioFormatError: For invalid formats
        ValueError: For invalid weights or shapes
    """
    try:
        # Compute embeddings for all audio files
        embeddings = []
        for audio_path in audio_paths:
            embedding = compute_audio_embedding(
                audio_path,
                target_sr=target_sr,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length
            )
            embeddings.append(embedding)
        
        # Mix embeddings
        return mix_audio_embeddings(embeddings, weights)
        
    except Exception as e:
        raise AudioFormatError(f"Error in multi-audio processing: {str(e)}") 