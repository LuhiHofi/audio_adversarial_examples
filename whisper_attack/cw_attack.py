"""
Carlini&Wagner attack (https://arxiv.org/abs/1801.01944)
"""
import argparse
import whisper
import scipy.io.wavfile as wav

from typing import List, Optional, Tuple

import numpy as np
import speechbrain as sb
import torch
import torch.nn as nn
import torch.optim as optim

import robust_speech as rs
from robust_speech.adversarial.attacks.imperceptible import ImperceptibleASRAttack


parser = argparse.ArgumentParser()
parser.add_argument("--epsilon", default=0.0001, type=float, help="Epsilon for the perturbation magnitude of the fgsm attack.")
parser.add_argument("--audio_path", default="./data/sample-000001.wav", type=str, help="Path to the audio to be attacked.")
parser.add_argument("--sample_rate", default=16000, type=int, help="Sample rate of the audio file.")
parser.add_argument("--iterations", default=10, type=int, help="Number of iterations for the iterative FGSM attack.")
parser.add_argument("--model_size", default="base", choices=['tiny', 'base', 'small', 'medium', 'large'], help="Model size: 'tiny', 'base', 'small', 'medium', 'large', 'huge'")
parser.add_argument("--target", default="Hello world!", help="Use targeted attack")

class ASRCarliniWagnerAttack(ImperceptibleASRAttack):
    """
    A Carlini&Wagner attack for ASR models.
    The algorithm follows the first attack in https://arxiv.org/abs/1801.01944
    Based on the ART implementation of Imperceptible
    (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/imperceptible_asr/imperceptible_asr_pytorch.py)

    Arguments
    ---------
     asr_brain : rs.adversarial.brain.ASRBrain
        the brain object to attack
     targeted: bool
        if the attack is targeted (always true for now).
     eps: float
        Linf bound applied to the perturbation.
     learning_rate: float
        the learning rate for the attack algorithm
     max_iter: int
        the maximum number of iterations
     clip_min: float
        mininum value per input dimension (ignored: herefor compatibility).
        
     clip_max: float
        maximum value per input dimension (ignored: herefor compatibility).
     train_mode_for_backward: bool
        whether to force training mode in backward passes (necessary for RNN models)
    global_max_length: int
        max length of a perturbation
    initial_rescale: float
        initial factor by which to rescale the perturbation
    num_iter_decrease_eps: int
        Number of times to increase epsilon in case of success
    decrease_factor_eps: int
        Factor by which to decrease epsilon in case of failure
    optimizer: Optional["torch.optim.Optimizer"]
        the optimizer to use
    """

    def __init__(
        self,
        asr_brain: rs.adversarial.brain.ASRBrain,
        eps: float = 0.05,
        max_iter: int = 10,
        learning_rate: float = 0.001,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        global_max_length: int = 200000,
        initial_rescale: float = 1.0,
        decrease_factor_eps: float = 0.8,
        num_iter_decrease_eps: int = 1,
        max_num_decrease_eps: Optional[int] = None,
        targeted: bool = True,
        train_mode_for_backward: bool = True,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        const: float = 1.0,
        confidence: float = 0.0,
        correct_first_word: bool = False
    ):
        super(ASRCarliniWagnerAttack, self).__init__(
            asr_brain,
            eps=eps,
            max_iter_1=max_iter,
            max_iter_2=0,
            learning_rate_1=learning_rate,
            optimizer_1=optimizer,
            global_max_length=global_max_length,
            initial_rescale=initial_rescale,
            decrease_factor_eps=decrease_factor_eps,
            num_iter_decrease_eps=num_iter_decrease_eps,
            max_num_decrease_eps=max_num_decrease_eps,
            targeted=targeted,
            train_mode_for_backward=train_mode_for_backward,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.reg_const = 1./const if const is not None else 0.
        self.confidence = confidence
        self.correct_first_word = correct_first_word
        self.asr_brain.hparams.confidence = self.confidence
        self.asr_brain.hparams.correct_first_word = self.correct_first_word

    # ADDED
    def on_generation_start(self, load_audio=False, save_audio_path=None, sample_rate=16000, log_snr=True):
        """
        Method to run at the beginning of an evaluation phase with adverersarial attacks.

        Arguments
        ---------
        save_audio_path: optional string
            path to the folder in which to save audio files
        sample_rate: int
            audio sample rate for wav encoding
        """

        self.on_evaluation_start(
            load_audio=load_audio,
            save_audio_path=save_audio_path,
            sample_rate=sample_rate,
            log_snr=log_snr,
        )
        
    def _forward_1st_stage(
        self,
        original_input: np.ndarray,
        batch: sb.dataio.batch.PaddedBatch,
        local_batch_size: int,
        local_max_length: int,
        rescale: np.ndarray,
        input_mask: np.ndarray,
        real_lengths: np.ndarray,
    ):

        # Compute perturbed inputs
        local_delta = self.global_optimal_delta[:
                                                local_batch_size, :local_max_length]
        local_delta_rescale = torch.clamp(local_delta, -self.eps, self.eps).to(
            self.asr_brain.device
        )
        local_delta_rescale *= torch.tensor(rescale).to(self.asr_brain.device)
        adv_input = local_delta_rescale + torch.tensor(original_input).to(
            self.asr_brain.device
        )
        masked_adv_input = adv_input * torch.tensor(input_mask).to(
            self.asr_brain.device
        )

        # Compute loss and decoded output
        batch.sig = masked_adv_input, batch.sig[1]
        predictions = self.asr_brain.compute_forward(batch, rs.Stage.ATTACK)
        loss = self.asr_brain.compute_objectives(
            predictions, batch, rs.Stage.ATTACK, reduction="batchmean")
        loss_backward = loss.mean() + self.reg_const * torch.norm(local_delta_rescale)
        decoded_output = self.asr_brain.get_tokens(predictions)
        # print(decoded_output,batch.tokens)
        # if teacher forcing prediction is correct, check decoder transcription
        if (decoded_output[0].view(-1) == batch.tokens_eos[0].cpu().view(-1)).all():
            self.asr_brain.module_eval()
            val_predictions = self.asr_brain.compute_forward(
                batch, sb.Stage.VALID)
            val_decoded_output = self.asr_brain.get_tokens(val_predictions)
            decoded_output = val_decoded_output
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        if len(loss.size()) == 0:
            loss = loss.view(1)
        return loss_backward, loss, local_delta, decoded_output, masked_adv_input, local_delta_rescale