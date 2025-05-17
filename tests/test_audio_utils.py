import unittest
import torch
import numpy as np
import os
import tempfile
from ..audio_utils import (
    validate_audio_file,
    load_audio,
    compute_mel_spectrogram,
    compute_audio_embedding,
    AudioFormatError
)

class TestAudioUtils(unittest.TestCase):
    def setUp(self):
        """Erstellt temporäre Test-Audiodateien."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Erstelle eine einfache Sinuswelle als Test-Audio
        sample_rate = 16000
        duration = 1.0  # 1 Sekunde
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz Sinuswelle
        
        # Speichere als WAV
        self.valid_audio_path = os.path.join(self.temp_dir, "test.wav")
        torchaudio.save(
            self.valid_audio_path,
            torch.from_numpy(audio).float().unsqueeze(0),
            sample_rate
        )
        
        # Erstelle eine ungültige Datei
        self.invalid_audio_path = os.path.join(self.temp_dir, "invalid.txt")
        with open(self.invalid_audio_path, 'w') as f:
            f.write("Keine Audiodatei")

    def tearDown(self):
        """Räumt temporäre Dateien auf."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_validate_audio_file_valid(self):
        """Test für gültige Audiodatei."""
        metadata = validate_audio_file(self.valid_audio_path)
        self.assertEqual(metadata['sample_rate'], 16000)
        self.assertEqual(metadata['channels'], 1)
        self.assertAlmostEqual(metadata['duration'], 1.0)

    def test_validate_audio_file_invalid(self):
        """Test für ungültige Audiodatei."""
        with self.assertRaises(AudioFormatError):
            validate_audio_file(self.invalid_audio_path)

    def test_validate_audio_file_nonexistent(self):
        """Test für nicht existierende Datei."""
        with self.assertRaises(AudioFormatError):
            validate_audio_file("nonexistent.wav")

    def test_load_audio_valid(self):
        """Test für das Laden einer gültigen Audiodatei."""
        audio, sr = load_audio(self.valid_audio_path)
        self.assertEqual(sr, 16000)
        self.assertEqual(audio.shape[0], 1)  # Batch-Dimension
        self.assertEqual(audio.shape[1], 1)  # Kanal-Dimension
        self.assertEqual(audio.shape[2], 16000)  # Zeit-Dimension

    def test_load_audio_invalid(self):
        """Test für das Laden einer ungültigen Audiodatei."""
        with self.assertRaises(AudioFormatError):
            load_audio(self.invalid_audio_path)

    def test_compute_mel_spectrogram(self):
        """Test für die Mel-Spektrogramm-Berechnung."""
        audio, sr = load_audio(self.valid_audio_path)
        mel_spec = compute_mel_spectrogram(audio, sr)
        
        # Überprüfe Dimensionen
        self.assertEqual(mel_spec.shape[0], 1)  # Batch-Dimension
        self.assertEqual(mel_spec.shape[1], 80)  # Mel-Bänder
        self.assertTrue(mel_spec.shape[2] > 0)  # Zeit-Dimension
        
        # Überprüfe Wertebereich
        self.assertTrue(torch.isfinite(mel_spec).all())
        self.assertTrue((mel_spec >= -20).all())  # Log-Skala sollte nicht zu negativ sein

    def test_compute_audio_embedding(self):
        """Test für die Audio-Embedding-Berechnung."""
        embedding = compute_audio_embedding(self.valid_audio_path)
        
        # Überprüfe Dimensionen
        self.assertEqual(embedding.shape[0], 1)  # Batch-Dimension
        self.assertEqual(embedding.shape[1], 80)  # Mel-Bänder
        self.assertTrue(embedding.shape[2] > 0)  # Zeit-Dimension
        
        # Überprüfe Normalisierung
        self.assertAlmostEqual(embedding.mean().item(), 0, places=1)
        self.assertAlmostEqual(embedding.std().item(), 1, places=1)

    def test_audio_embedding_invalid(self):
        """Test für Embedding-Berechnung mit ungültiger Datei."""
        with self.assertRaises(AudioFormatError):
            compute_audio_embedding(self.invalid_audio_path)

    def test_mix_audio_embeddings(self):
        """Test für das Mischen von Audio-Embeddings."""
        # Erstelle zwei verschiedene Test-Audiodateien
        sample_rate = 16000
        duration = 1.0
        
        # Erste Sinuswelle (440 Hz)
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio1 = np.sin(2 * np.pi * 440 * t)
        path1 = os.path.join(self.temp_dir, "test1.wav")
        torchaudio.save(
            path1,
            torch.from_numpy(audio1).float().unsqueeze(0),
            sample_rate
        )
        
        # Zweite Sinuswelle (880 Hz)
        audio2 = np.sin(2 * np.pi * 880 * t)
        path2 = os.path.join(self.temp_dir, "test2.wav")
        torchaudio.save(
            path2,
            torch.from_numpy(audio2).float().unsqueeze(0),
            sample_rate
        )
        
        # Berechne Embeddings
        emb1 = compute_audio_embedding(path1)
        emb2 = compute_audio_embedding(path2)
        
        # Teste Mischen ohne Gewichte
        mixed = mix_audio_embeddings([emb1, emb2])
        self.assertEqual(mixed.shape, emb1.shape)
        self.assertAlmostEqual(mixed.mean().item(), 0, places=1)
        
        # Teste Mischen mit Gewichten
        weights = [0.7, 0.3]
        mixed_weighted = mix_audio_embeddings([emb1, emb2], weights)
        self.assertEqual(mixed_weighted.shape, emb1.shape)
        
        # Teste Fehlerfälle
        with self.assertRaises(ValueError):
            mix_audio_embeddings([])  # Leere Liste
            
        with self.assertRaises(ValueError):
            mix_audio_embeddings([emb1, emb2], [0.5])  # Falsche Anzahl Gewichte

    def test_compute_multi_audio_embedding(self):
        """Test für die Multi-Audio-Embedding-Berechnung."""
        # Erstelle zwei Test-Audiodateien
        sample_rate = 16000
        duration = 1.0
        
        # Erste Sinuswelle (440 Hz)
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio1 = np.sin(2 * np.pi * 440 * t)
        path1 = os.path.join(self.temp_dir, "test1.wav")
        torchaudio.save(
            path1,
            torch.from_numpy(audio1).float().unsqueeze(0),
            sample_rate
        )
        
        # Zweite Sinuswelle (880 Hz)
        audio2 = np.sin(2 * np.pi * 880 * t)
        path2 = os.path.join(self.temp_dir, "test2.wav")
        torchaudio.save(
            path2,
            torch.from_numpy(audio2).float().unsqueeze(0),
            sample_rate
        )
        
        # Teste Berechnung ohne Gewichte
        mixed = compute_multi_audio_embedding([path1, path2])
        self.assertEqual(mixed.shape[0], 1)  # Batch-Dimension
        self.assertEqual(mixed.shape[1], 80)  # Mel-Bänder
        self.assertTrue(mixed.shape[2] > 0)  # Zeit-Dimension
        
        # Teste Berechnung mit Gewichten
        weights = [0.7, 0.3]
        mixed_weighted = compute_multi_audio_embedding([path1, path2], weights)
        self.assertEqual(mixed_weighted.shape, mixed.shape)
        
        # Teste Fehlerfälle
        with self.assertRaises(AudioFormatError):
            compute_multi_audio_embedding([self.invalid_audio_path])
            
        with self.assertRaises(ValueError):
            compute_multi_audio_embedding([path1, path2], [0.5])  # Falsche Anzahl Gewichte

if __name__ == '__main__':
    unittest.main() 