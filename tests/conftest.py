import os
import sys

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test-Konfiguration
TEST_AUDIO_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
os.makedirs(TEST_AUDIO_DIR, exist_ok=True) 