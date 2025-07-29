"""
File containing functions to load and manipulate Whisper model wrapped in WhisperASR.
"""

import torch
from whisper_attack.sb_whisper_binding import WhisperASR 
from whisper_attack.whisper_with_gradients import WhisperWrapper
from speechbrain.nnet.linear import Linear

def get_model_hparams(model_name="tiny"):
    """
    Get the hyperparameters for the Whisper model.
    """
    return {
        "tiny": "model_configs/tiny.yaml",
        "base": "model_configs/base.yaml",
        "small": "model_configs/small.yaml",
        "medium": "model_configs/medium.yaml",
        "large": "model_configs/large.yaml",
    }[model_name]
    

def load_whisper_model(model_name="tiny"):
    """
    Load the Whisper model with gradients enabled.
    """
    whisper_model = WhisperWrapper(
        name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    placeholder_model = Linear(
        input_size=8,
        n_neurons=8
    )
    modules = {
        "whisper": whisper_model,
        "placeholder_model": placeholder_model,
    }
    hparams = {
        "voting_module": None,  # Required by robust_speech
        "sample_rate": 16000,
    }
    return WhisperASR(
        modules=modules,
        opt_class=None,
        hparams=hparams,
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        checkpointer=None
    )

def change_model_size(model, new_size):
    """
    Change the size of the Whisper model.
    """
    if new_size not in ["tiny", "base", "small", "medium", "large"]:
        raise ValueError("Invalid model size. Choose from: 'tiny', 'base', 'small', 'medium', 'large'.")
    
    model.modules["whisper"].name = new_size
    # Load the new model with gradients
    resized_whisper = WhisperWrapper(
        name=new_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Replace the whisper model in the existing model
    model.modules["whisper"] = resized_whisper
    
    return model

def get_attack_hparams_file(attack_type: str) -> str:
    return {
        "snr pgd": "attack_configs/whisper_1_file/snr_pgd.yaml",
        "pgd": "attack_configs/whisper_1_file/pgd.yaml",
        "modified cw": "attack_configs/whisper_1_file/cw_modified.yaml",
        "cw": "attack_configs/whisper_1_file/cw.yaml",
        "genetic": "attack_configs/whisper_1_file/genetic.yaml",
        "universal pgd": "attack_configs/whisper_1_file/univ_pgd.yaml",
        "universal pgd with noise": "attack_configs/whisper_1_file/univ_noise.yaml",
        "rand": "attack_configs/whisper_1_file/rand.yaml"
    }[attack_type]
    
def change_attacker(model, attack_hparams):
    """
    Change the attacker in the Whisper model.
    """
    attacker = attack_hparams["attack_class"]
    attacker = attacker(model)
    model.attacker = attacker
    return model