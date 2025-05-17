#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import torch
from .audio_utils import compute_audio_embedding, AudioFormatError

def load_workflow(workflow_path: str) -> Dict[str, Any]:
    """Lädt ein ComfyUI-Workflow aus JSON."""
    with open(workflow_path, 'r') as f:
        return json.load(f)

def save_workflow(workflow: Dict[str, Any], output_path: str):
    """Speichert ein ComfyUI-Workflow als JSON."""
    with open(output_path, 'w') as f:
        json.dump(workflow, f, indent=4)

def process_batch(
    workflow_path: str,
    audio_dir: str,
    output_dir: str,
    model_name: str,
    clip_name: str,
    vae_name: str,
    prompt: str,
    negative_prompt: str = "",
    batch_size: int = 1
) -> None:
    """
    Verarbeitet eine Batch von Audiodateien mit dem angegebenen Workflow.
    
    Args:
        workflow_path: Pfad zum Basis-Workflow
        audio_dir: Verzeichnis mit Audiodateien
        output_dir: Ausgabeverzeichnis
        model_name: Name des GGUF-Modells
        clip_name: Name des CLIP-Modells
        vae_name: Name des VAE-Modells
        prompt: Text-Prompt
        negative_prompt: Negativer Prompt
        batch_size: Batch-Größe
    """
    # Lade Basis-Workflow
    workflow = load_workflow(workflow_path)
    
    # Erstelle Ausgabeverzeichnis
    os.makedirs(output_dir, exist_ok=True)
    
    # Finde alle WAV-Dateien
    audio_files = list(Path(audio_dir).glob("*.wav"))
    
    # Verarbeite in Batches
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]
        
        for audio_file in batch_files:
            try:
                # Berechne Audio-Embedding
                audio_embedding = compute_audio_embedding(str(audio_file))
                
                # Aktualisiere Workflow-Parameter
                workflow = update_workflow_params(
                    workflow,
                    model_name=model_name,
                    clip_name=clip_name,
                    vae_name=vae_name,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    audio_embedding=audio_embedding
                )
                
                # Generiere eindeutigen Output-Namen
                output_name = f"{audio_file.stem}_generated"
                output_path = os.path.join(output_dir, f"{output_name}.json")
                
                # Speichere angepasstes Workflow
                save_workflow(workflow, output_path)
                
                print(f"Verarbeitet: {audio_file.name} -> {output_path}")
                
            except AudioFormatError as e:
                print(f"Fehler bei {audio_file.name}: {str(e)}")
                continue

def update_workflow_params(
    workflow: Dict[str, Any],
    model_name: str,
    clip_name: str,
    vae_name: str,
    prompt: str,
    negative_prompt: str,
    audio_embedding: torch.Tensor
) -> Dict[str, Any]:
    """Aktualisiert die Parameter eines Workflows."""
    # Aktualisiere Modell-Namen
    for node in workflow["nodes"]:
        if node["type"] == "UnetLoaderGGUFWithAudio":
            node["inputs"]["unet_name"] = model_name
        elif node["type"] == "CLIPLoaderGGUF":
            node["inputs"]["clip_name"] = clip_name
        elif node["type"] == "VAELoader":
            node["inputs"]["vae_name"] = vae_name
        elif node["type"] == "CLIPTextEncode":
            node["inputs"]["text"] = prompt
            # Füge negativen Prompt hinzu, falls vorhanden
            if negative_prompt:
                node["inputs"]["negative_text"] = negative_prompt
    
    return workflow

def main():
    parser = argparse.ArgumentParser(description="Batch-Verarbeitung für Audio-konditionierte GGUF-Generierung")
    
    parser.add_argument("--workflow", required=True, help="Pfad zum Basis-Workflow")
    parser.add_argument("--audio-dir", required=True, help="Verzeichnis mit Audiodateien")
    parser.add_argument("--output-dir", required=True, help="Ausgabeverzeichnis")
    parser.add_argument("--model", required=True, help="Name des GGUF-Modells")
    parser.add_argument("--clip", required=True, help="Name des CLIP-Modells")
    parser.add_argument("--vae", required=True, help="Name des VAE-Modells")
    parser.add_argument("--prompt", required=True, help="Text-Prompt")
    parser.add_argument("--negative-prompt", default="", help="Negativer Prompt")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch-Größe")
    
    args = parser.parse_args()
    
    process_batch(
        workflow_path=args.workflow,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        clip_name=args.clip,
        vae_name=args.vae,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main() 