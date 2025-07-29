"""
Console interface for running Whisper attacks on datasets. .yaml configuration files are setup for test-clean dataset and it's subsets. 
If you want to use other datasets, please change the dataset_csv_name and test_splits in the .yaml files accordingly.
"""

import cmd
import os
import torch
from whisper_attack.run_attack import evaluate

def get_attack_hparams_file(attack_type: str) -> str:
    return {
        "snr_pgd": "attack_configs/whisper/snr_pgd.yaml",
        "pgd": "attack_configs/whisper/pgd.yaml",
        "modified_cw": "attack_configs/whisper/cw_modified.yaml",
        "cw": "attack_configs/whisper/cw.yaml",
        "genetic": "attack_configs/whisper/genetic.yaml",
        "universal": "attack_configs/whisper/univ_pgd.yaml",
        "universal_noise": "attack_configs/whisper/univ_noise.yaml",
        "rand": "attack_configs/whisper/rand.yaml"
    }[attack_type]

class AttackCLI(cmd.Cmd):
    prompt = "(attack) > "
    intro = "Welcome to Whisper Attack Console. Type 'help' for commands."

    def __init__(self):
        super().__init__()
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.run_opts = {'debug': False, 'debug_batches': 2, 'debug_epochs': 2, 'device': self.device, 
            'data_parallel_backend': False, 'distributed_launch': False, 'distributed_backend': 'nccl', 
            'find_unused_parameters': False, 'tqdm_colored_bar': False}        
        self.current_model = "base"
        self.current_seed = 250
        self.current_csv = "test-clean-100"
        self.load_audio = False
        self.skip_prep_dataset = True

    # command methods
    def do_dataset(self, arg):
        """Set dataset CSV name. [default: test-clean-100] Usage: dataset test-clean-100"""
        if arg:
            self.current_csv = arg
            print(f"Dataset set to: {self.current_csv}")
        else:
            print(f"Current dataset: {self.current_csv}")
    def do_seed(self, arg):
        """Set random seed. [default: 250] Usage: seed 250"""
        try:
            if arg:
                self.current_seed = int(arg)
                print(f"Seed set to: {self.current_seed}")
            else:
                print(f"Current seed: {self.current_seed}")
        except ValueError:
            print("Invalid seed. Please enter an integer.")
    def do_load_audio(self, arg):
        """Set whether to load audio data. False if you want to use precomputed features. [default: False] Usage: load_audio True/False"""
        if arg:
            if not self._check_if_bool_input(arg):
                print("Invalid input! Please enter 'y'/'n' or 'True'/'False'.")
            else:
                self.load_audio = self._convert_to_bool(arg)
                print(f"Load audio set to: {self.load_audio}")
        else:
            print(f"Current load_audio setting: {self.load_audio}")
    def do_skip_prep_dataset(self, arg):
        """Skip dataset preparation. [default: True] Usage: skip_prep_dataset True/False"""
        if arg:
            if not self._check_if_bool_input(arg):
                print("Invalid input! Please enter 'y'/'n' or 'True'/'False'.")
            else:
                self.skip_prep_dataset = self._convert_to_bool(arg)
                print(f"Skip dataset preparation set to: {self.skip_prep_dataset}")
        else:
            print(f"Current skip dataset preparation setting: {self.skip_prep_dataset}")
    def do_model(self, arg):
        """Set model size (tiny, base [default], small, medium, large). Usage: model base"""
        valid_models = ['tiny', 'base', 'small', 'medium', 'large']
        if arg:
            if arg.lower() in valid_models:
                self.current_model = arg.lower()
                print(f"Model set to: {self.current_model}")
            else:
                print(f"Invalid model. Choose from: {', '.join(valid_models)}")
        else:
            print(f"Current model: {self.current_model}")
    def do_attack(self, arg):
        """Run an attack. Options: snr_pgd, pgd, cw, modified_cw, genetic, rand, universal, universal_noise. Usage: attack snr_pgd"""
        attack_methods = {
            "snr_pgd": self._setup_snr_pgd,
            "pgd": self._setup_pgd,
            "cw": self._setup_cw,
            "modified_cw": self._setup_modified_cw,
            "genetic": self._setup_genetic,
            "universal": self._setup_universal,
            "universal_noise": self._setup_universal_noise,
            "rand": self._setup_rand
        }
        if arg.lower() in attack_methods:
            hparams_file = get_attack_hparams_file(arg.lower())
            overrides = f"root: {self.path}\n" \
                        f"model_label: {self.current_model}\n" \
                        f"seed: {self.current_seed}\n" \
                        f"data_csv_name: {self.current_csv}\n" \
                        f"skip_prep: {self.skip_prep_dataset}\n" \
                        f"load_audio: {self.load_audio}\n"
            overrides += attack_methods[arg.lower()]()
            if arg == "universal_noise" or arg == "universal" and self.current_model == "large":
                print("Warning: Due to hardware constraints universal attack was not trained on large models. Using perturbation trained using medium model instead.")                
            evaluate(
                hparams_file=hparams_file,
                run_opts=self.run_opts,
                overrides=overrides
            )
        else:
            print(f"Unknown attack. Available: {', '.join(attack_methods.keys())}")

    def do_exit(self, arg):
        """Exit the console."""
        print("Exiting...")
        return True
    
    def _check_if_bool_input(self, arg):
        if arg in ('y', 'yes', 'true', 'True'):
            return True
        elif arg in ('n', 'no', 'false', 'False'):
            return True
        return False
    
    def _convert_to_bool(self, arg):
        if arg in ('y', 'yes', 'true', 'True'):
            return True
        elif arg in ('n', 'no', 'false', 'False'):
            return False
        else:
            raise ValueError("Invalid boolean input! Please enter 'y'/'n' or 'True'/'False'.")
    
    def _get_int_input(self, prompt, default=250):
        while True:
            user_input = input(f"{prompt} (default {default}): ")
            if user_input == "":
                print(f"Using default value for {prompt}: {default}")
                return default
            try:
                num = int(user_input)
                print(f"Using value for {prompt}: {num}")
                return num
            except ValueError:
                print("Invalid input! Please enter an integer.")
    def _get_float_input(self, prompt, default=0.001):
        while True:
            user_input = input(f"{prompt} (default {default}): ")
            if user_input == "":
                print(f"Using default value for {prompt}: {default}")
                return default
            try:
                num = float(user_input)
                print(f"Using value for {prompt}: {num}")
                return num
            except ValueError:
                print("Invalid input! Please enter a float.")

    # Parameter preparation methods
    def _get_iters_and_batch_size(self):
        nb_iter = self._get_int_input("Number of iterations", 200)
        batch_size = self._get_int_input("Batch size", 1)
        return f"nb_iter: {nb_iter}\nbatch_size: {batch_size}\n"
    def _setup_rand(self):
        sigma = self._get_float_input("Sigma", 0.001)
        return f"sigma: {sigma}\n"
    def _setup_genetic(self):
        overrides = self._setup_pgd()
        population_size = self._get_int_input("Population size", 50)
        return overrides + f"population_size: {population_size}\n"
    def _setup_pgd(self):
        overrides = self._get_iters_and_batch_size()
        eps = self._get_float_input("Epsilon", 0.001)
        return overrides + f"eps: {eps}\n"
    def _setup_snr_pgd(self):
        overrides = self._get_iters_and_batch_size()
        snr = self._get_int_input("SNR", 35)
        return overrides + f"snr: {snr}\n"
    def _setup_cw(self):
        overrides = self._get_iters_and_batch_size()
        eps = self._get_float_input("epsilon", 0.001)
        # Limits the number of times the attack reduces the perturbation strength
        max_decr = self._get_int_input("Maximum number of decrease attempts", 3) 
        # Controls the balance between perturbation magnitude and misclassification confidence in the CW loss function.
        const = self._get_float_input("Trade-off constant", 100.00)
        lr = self._get_float_input("Learning rate", 0.01)
        decrease_factor_eps = self._get_float_input("Decrease factor for epsilon", 0.8)
        target = input("Target phrase: ")
        return overrides + f"eps: {eps}\nmax_decr: {max_decr}\nconst: {const}\nlr: {lr}\ndecrease_factor_eps: {decrease_factor_eps}\ntarget_sentence: {target}\n"
    def _setup_modified_cw(self):
        return self._setup_cw()
    def _setup_universal(self):
        # Universal perturbation is already precomputed 
        return ""
    def _setup_universal_noise(self):
        return ""

if __name__ == "__main__":
    AttackCLI().cmdloop()