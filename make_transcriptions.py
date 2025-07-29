import whisper
import argparse
import torchaudio
import os

parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", default="audio_files/univ_noise/tiny/1580-141083-0019", 
                    type=str, help="Path to the audio file.")
parser.add_argument("--output_path", default="data/transcriptions", 
                    type=str, help="Path to save the transcription.")
parser.add_argument("--model_name", default="tiny", type=str, help="Name of the Whisper model to use.")

def transcribe_audio(args) -> str:
    attack = args.audio_path.split("/")[-3]
    
    model = whisper.load_model(args.model_name)
    nat = args.audio_path + "_nat.wav"
    adv = args.audio_path + "_adv.wav"
    original_transcription = model.transcribe(nat)
    adversarial_transcription = model.transcribe(adv)

    waveform_nat, sample_rate_nat = torchaudio.load(nat)
    waveform_adv, sample_rate_adv = torchaudio.load(adv)
    
    output_path = os.path.join(args.output_path, attack, args.model_name)
    os.makedirs(output_path, exist_ok=True)
    
    nat_path = os.path.join(output_path, "nat.wav")
    adv_path = os.path.join(output_path, "adv.wav")
    torchaudio.save(nat_path, waveform_nat, sample_rate_nat)
    torchaudio.save(adv_path, waveform_adv, sample_rate_adv)
    transcription_file = os.path.join(output_path, "transcriptions.txt")

    with open(transcription_file, "w") as f:
        f.write(f"Original: {original_transcription['text']}\n")
        f.write(f"Adversarial: {adversarial_transcription['text']}\n")
    print(f"Transcriptions saved to {output_path}")
    return original_transcription["text"], adversarial_transcription["text"]

if __name__ == "__main__":
    args = parser.parse_args()
    transcribe_audio(args)