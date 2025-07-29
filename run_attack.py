"""
This script runs an attack on a Whisper model using specified hyperparameters and configurations. 
Uses whisper_attack/run_attack.py as a base
"""

import os
import argparse
from whisper_attack.run_attack import evaluate
import torch

parser = argparse.ArgumentParser()

# Attack selection
parser.add_argument("--attack_type", 
                    default="snr_pgd", 
                    type=str,
                    choices=["snr_pgd", "pgd", "modified_cw", "cw", "genetic", "universal", "rand"],
                    help="Type of attack to perform. Choices are: 'snr_pgd', 'pgd', 'modified_cw', 'cw', 'genetic', 'universal', 'rand'")    

# Hyperparameter overrides
parser.add_argument("--root", 
                    default="~/lukas-hofman", 
                    type=str, 
                    help="Root directory")

parser.add_argument("--seed", 
                    default=250, 
                    type=int, 
                    help="Random seed for reproducibility")
parser.add_argument("--model_name", 
                    default="tiny", 
                    type=str, 
                    choices=["tiny", "base", "small", "medium", "large"],
                    help="Whisper model size")
parser.add_argument("--data_csv_name", 
                    default="test-clean-100", 
                    type=str, 
                    help="Name of the CSV file containing the test data")
parser.add_argument("--load_audio",
                    default=False,
                    type=bool,
                    help="Whether to load audio data or not. Set to False if you want to use precomputed features.")
parser.add_argument("--overrides", 
                    default=None,
                    type=str,
                    help="Aditional hyperparameter overrides. For example: 'eps: 0.001\nbatch_size: 5\n'")
    
def get_attack_hparams_file(attack_type: str) -> str:
    return {
        "snr_pgd": "attack_configs/whisper/pgd.yaml",
        "pgd": "attack_configs/whisper/pgdOriginal.yaml",
        "modified_cw": "attack_configs/whisper/cw.yaml",
        "cw": "attack_configs/whisper/cwOriginal.yaml",
        "genetic": "attack_configs/whisper/genetic.yaml",
        "universal": "attack_configs/whisper/univ_pgd.yaml",
        "rand": "attack_configs/whisper/rand.yaml"
    }[attack_type]

def main(args):
    after_home = args.root.replace("~/", "")
    path = os.path.expanduser(f"~/{after_home}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_opts = {'debug': False, 'debug_batches': 2, 'debug_epochs': 2, 'device': device, 
            'data_parallel_backend': False, 'distributed_launch': False, 'distributed_backend': 'nccl', 
            'find_unused_parameters': False, 'tqdm_colored_bar': False}
    hparams_file = get_attack_hparams_file(args.attack_type)
    overrides = f"root: {path}\n" \
                f"model_label: {args.model_name}\n" \
                f"seed: {args.seed}\n" \
                f"data_csv_name: {args.data_csv_name}\n" \
                f"load_audio: {args.load_audio}\n"
    if args.overrides:
        overrides += args.overrides                

    # Run the attack
    evaluate(
        hparams_file=hparams_file,
        run_opts=run_opts,
        overrides=overrides
    )

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
