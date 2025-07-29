"""Simple function that transcribes audio using Whisper model wrapped in WhisperASR."""

import load_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", default="data/sample-000000.wav", type=str, help="Path to the audio to be classified.")
parser.add_argument("--model_name", default="tiny", type=str, choices=["tiny", "base", "small", "medium", "large"], help="Model size: 'tiny', 'base', 'small', 'medium', 'large'")

def transcribe(audio_path="data/sample-000000.wav", model_name="tiny"):
    model = load_model.load_whisper_model(model_name)
    text = model.transcribe(audio_path)
    print(f"Transcription: {text}")
    return text

if __name__ == "__main__":
    args = parser.parse_args()
    adversarial_audio = transcribe(args.audio_path, args.model_name)
