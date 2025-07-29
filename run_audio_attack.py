"""
This script is an adaptation of the whisper_attack/run_attack.py script. Instead of running an attack on a dataset, it runs a transcription on a single audio file.
This file includes two main functions:
1. `generate`: This function reads hyperparameters from a YAML file, initializes the model, performs an attack on the specified audio file, then saves the adversarial audio.
2. `generate_example`: This function is similar to `generate`, but it takes brain as an argument and instead of saving the adversarial audio, it returns the perturbed audio.
It is used in the GUI to generate adversarial audio for a single example.
"""

import os
import sys

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import torchaudio
import whisper
import torch
from speechbrain.dataio.batch import PaddedBatch
from robust_speech.adversarial.utils import TargetGeneratorFromFixedTargets
from whisper_attack.run_attack import read_brains 

import robust_speech as rs
import load_model

def generate(hparams_file, run_opts, overrides, audio_path):
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if "pretrainer" in hparams:  # load parameters
        # the tokenizer currently is loaded from the main hparams file and set
        # in all brain classes
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])

    if "tokenizer_builder" in hparams:
        tokenizer = hparams["tokenizer_builder"](hparams["tokenizer_name"])
        hparams["tokenizer"] = tokenizer
    else:
        tokenizer=hparams["tokenizer"]

    source_brain = None
    if "source_brain_class" in hparams:  # loading source model
        source_brain = read_brains(
            hparams["source_brain_class"],
            hparams["source_brain_hparams_file"],
            run_opts=run_opts,
            overrides={"root": hparams["root"]},
            tokenizer=tokenizer,
        )
    attacker = hparams["attack_class"]
    if source_brain and attacker:
        # instanciating with the source model if there is one.
        # Otherwise, AdvASRBrain will handle instanciating the attacker with
        # the target model.
        if isinstance(source_brain, rs.adversarial.brain.EnsembleASRBrain):
            if "source_ref_attack" in hparams:
                source_brain.ref_attack = hparams["source_ref_attack"]
            if "source_ref_train" in hparams:
                source_brain.ref_train = hparams["source_ref_train"]
            if "source_ref_valid_test" in hparams:
                source_brain.ref_valid_test = hparams["source_ref_valid_test"]
        attacker = attacker(source_brain)

    # Target model initialization
    target_brain_class = hparams["target_brain_class"]
    target_hparams = (
        hparams["target_brain_hparams_file"]
        if hparams["target_brain_hparams_file"]
        else hparams
    )
    if source_brain is not None and target_hparams == hparams["source_brain_hparams_file"]:
        # avoid loading the same network twice
        sc_brain = source_brain
        sc_class = target_brain_class
        if isinstance(source_brain, rs.adversarial.brain.EnsembleASRBrain):
            sc_brain = source_brain.asr_brains[source_brain.ref_valid_test]
            sc_class = target_brain_class[source_brain.ref_valid_test]
        target_brain = sc_class(
            modules=sc_brain.modules,
            hparams=sc_brain.hparams.__dict__,
            run_opts=run_opts,
            checkpointer=None,
            attacker=attacker,
        )
        target_brain.tokenizer = tokenizer
    else:
        target_brain = read_brains(
            target_brain_class,
            target_hparams,
            attacker=attacker,
            run_opts=run_opts,
            overrides={"root": hparams["root"]},
            tokenizer=tokenizer,
        )

    # Generation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    sig = whisper.load_audio(audio_path, sr=hparams["sample_rate"])
    text = target_brain.transcribe(audio_path)
    tokens = tokenizer.encode(text)
    
    batch = PaddedBatch([{
        "id": audio_path.split("/")[-1],
        "sig": torch.tensor(sig),
        "wav": audio_path,
        "wrd": text,
        "tokens": torch.tensor(tokens),
        "tokens_bos": torch.tensor(tokens),
        "tokens_eos": torch.tensor(tokens),
    }]).to(device)
    
    target = None
    if "target_generator" in hparams:
        target = hparams["target_generator"]
    elif "target_sentence" in hparams:
        target = TargetGeneratorFromFixedTargets(
            target=hparams["target_sentence"])
        
    if target is not None and attacker.targeted:
        batch_to_attack = target.replace_tokens_in_batch(batch, tokenizer, target_brain.hparams)
    else:
        batch_to_attack = batch
    _, adv_wav = target_brain.compute_forward_adversarial(batch_to_attack, stage=rs.Stage.ATTACK)
    adv_wav = adv_wav.to("cpu")
    
    save_audio_path = hparams["save_audio_path"] if "save_audio_path" in hparams else None
    if save_audio_path:
        if not os.path.exists(save_audio_path):
            os.makedirs(save_audio_path)
        audio_id = batch.id[0]
        audio_id = audio_id.split(".")[0]
        adv_wav_path = os.path.join(save_audio_path, f"{audio_id}_adv.wav")
        torchaudio.save(adv_wav_path, adv_wav, hparams["sample_rate"])

def generate_example(hparams_file, overrides, audio_path, audio, brain):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_opts = {'debug': False, 'debug_batches': 2, 'debug_epochs': 2, 'device': device, 
            'data_parallel_backend': False, 'distributed_launch': False, 'distributed_backend': 'nccl', 
            'find_unused_parameters': False, 'tqdm_colored_bar': False} 
    
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    if "pretrainer" in hparams:  # load parameters
        # the tokenizer currently is loaded from the main hparams file and set
        # in all brain classes
        run_on_main(hparams["pretrainer"].collect_files)
        hparams["pretrainer"].load_collected(device=run_opts["device"])

    if "tokenizer_builder" in hparams:
        tokenizer = hparams["tokenizer_builder"](hparams["tokenizer_name"])
        hparams["tokenizer"] = tokenizer
    else:
        tokenizer=hparams["tokenizer"]
 
    brain = load_model.change_attacker(
        brain, hparams
    )
    
    brain.tokenizer = tokenizer
    # Generation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    text = brain.transcribe(audio_path)
    tokens = tokenizer.encode(text)
    
    batch = PaddedBatch([{
        "id": audio_path.split("/")[-1],
        "sig": torch.tensor(audio),
        "wav": audio_path,
        "wrd": text,
        "tokens": torch.tensor(tokens),
        "tokens_bos": torch.tensor(tokens),
        "tokens_eos": torch.tensor(tokens),
    }]).to(device)
    
    target = None
    if "target_generator" in hparams:
        target = hparams["target_generator"]
    elif "target_sentence" in hparams:
        target = TargetGeneratorFromFixedTargets(
            target=hparams["target_sentence"])
        
    if target is not None and brain.attacker.targeted:
        batch_to_attack = target.replace_tokens_in_batch(batch, tokenizer, brain.hparams)
    else:
        batch_to_attack = batch
    _, adv_wav = brain.compute_forward_adversarial(batch_to_attack, stage=rs.Stage.ATTACK)
    
    return adv_wav.to("cpu")

if __name__ == "__main__":

    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)

    generate(hparams_file, run_opts, overrides, audio_path="data/sample-000000.wav")
