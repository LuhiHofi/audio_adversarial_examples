"""
Universal adversarial perturbation for Whisper ASR with background noise.
This code implements a universal adversarial perturbation attack for Whisper ASR models with Linf PGD attack as the base adversarial attack with a background noise component, that is also trainable.
"""

from robust_speech.adversarial.attacks.pgd import ASRLinfPGDAttack
from whisper import load_audio
import robust_speech as rs
import torch
from robust_speech.adversarial.attacks.attacker import TrainableAttacker
from robust_speech.adversarial.utils import linf_clamp
from jiwer import cer
from tqdm import tqdm
import os

MAXLEN=16000*30
class UniversalWithNoiseWhisperPGDAttack(TrainableAttacker,ASRLinfPGDAttack):
    def __init__(self,asr_brain,*args,nb_epochs=10,eps_item=0.001,success_every=10,univ_perturb=None,epoch_counter=None, background_audio_path="audio_backgrounds/raining-in-backyard.wav", t=0.8, background_noise_strength=0.5, **kwargs):
        ASRLinfPGDAttack.__init__(self,asr_brain,*args, **kwargs)
        self.univ_perturb = univ_perturb
        if self.univ_perturb is None:
            self.univ_perturb = rs.adversarial.utils.TensorModule(size=(MAXLEN,))
        self.nb_epochs=nb_epochs
        self.eps_item=eps_item
        self.success_every=success_every
        self.epoch_counter = epoch_counter
        if self.epoch_counter is None:
            self.epoch_counter = range(100)
        
        self.background_audio = background_noise_strength * load_audio(background_audio_path)
        self.t = t
        self.background_noise_strength = background_noise_strength

    def fit(self, loader):
        return self._compute_universal_perturbation(loader)

    def _compute_universal_perturbation(self, loader):
        self.univ_perturb=self.univ_perturb.to(self.asr_brain.device)
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()

        delta = self.univ_perturb.tensor.data
        success_rate = 0

        best_success_rate = -100
        best_CER = 0
        for epoch in self.epoch_counter:
            print(f'epoch {epoch}/{self.nb_epochs}')
            # GENERATE CANDIDATE FOR UNIVERSAL PERTURBATION
            for idx, batch in enumerate(loader):
                batch = batch.to(self.asr_brain.device)
                batch_size = batch.batchsize
                wav_init, wav_lens = batch.sig
                max_len = wav_init.shape[1]
                actual_lens = (wav_lens * max_len).long() 
                
                background_audio = torch.from_numpy(self.background_audio).to(self.asr_brain.device)       
                
                if wav_init.shape[1] <= background_audio.shape[0]:
                    background_x = background_audio[:wav_init.shape[1]]
                else:
                    background_x = torch.zeros_like(wav_init[0])
                    background_x[:background_audio.shape[0]] = background_audio
                    
                background_batch = background_x.unsqueeze(0).expand(wav_init.size())
                background_batch = background_batch[:, :max_len]
        
                delta_batch = torch.zeros((batch_size, max_len), device=self.asr_brain.device)
                delta_len = delta.shape[0]

                longer_mask = actual_lens > delta_len
                shorter_mask = ~longer_mask
                
                if longer_mask.any():
                    delta_x_long = delta.expand(longer_mask.sum(), -1)
                    delta_batch[longer_mask, :delta_len] = delta_x_long
                if shorter_mask.any():
                    actual_lens_short = actual_lens[shorter_mask]
                    max_start = delta_len - actual_lens_short - 1
                    begins = (torch.rand_like(actual_lens_short, dtype=torch.float) * max_start).floor().long()

                    idxs = torch.arange(max_len, device=delta.device).unsqueeze(0)  # (1, max_len)
                    begins_exp = begins.unsqueeze(1)  # (batch, 1)
                    lens_exp = actual_lens_short.unsqueeze(1)  # (batch, 1)

                    mask = idxs < lens_exp

                    offsets = begins_exp + idxs[:, :lens_exp.max()].to(begins.device)
                    offsets = torch.where(mask, offsets, torch.zeros_like(offsets))

                    delta_x_short = delta[offsets.clamp(0, delta_len - 1)]
                    delta_x_short = delta_x_short * mask

                    delta_batch[shorter_mask, :lens_exp.max()] = delta_x_short
                    
                r = torch.zeros_like(delta_batch)
                r.requires_grad_()

                for i in range(self.nb_iter):
                    perturbed_input = wav_init + delta_batch + r + background_batch

                    batch.sig = perturbed_input, wav_lens
                    
                    predictions = self.asr_brain.compute_forward(
                        batch, rs.Stage.ATTACK)
                    model_loss = self.asr_brain.compute_objectives(
                            predictions, batch, rs.Stage.ATTACK)
                    l2_norm = r.view(r.size(0), -1).norm(p=2, dim=1).mean()
                    if torch.all(model_loss < 0.1):
                            break
                    loss = 0.5 * l2_norm + model_loss.mean()
                    loss.backward()
                    grad_sign = r.grad.data.sign()
                    alpha = self.rel_eps_iter * self.eps_item
                    r.data = r.data - alpha * grad_sign
                    r.data = linf_clamp(r.data, self.eps_item)
                    r.data = linf_clamp(delta_batch + r.data, self.eps) - delta_batch

                    r.grad.data.zero_()

                delta_batch  = linf_clamp(delta_batch + r.data, self.eps)
                updated_delta = delta.clone()
                
                if longer_mask.any():
                    longer_samples = delta_batch[longer_mask, :delta.shape[0]]
                    mean_update_long = longer_samples.sum(dim=0)
                    updated_delta += mean_update_long.detach()
                if shorter_mask.any():
                    delta_len = delta.shape[0]
                    shorter_indices = shorter_mask.nonzero(as_tuple=False).squeeze(1)

                    for i, sample_idx in enumerate(shorter_indices):
                        begin = begins[i].item()
                        length = actual_lens[sample_idx].item()
                        updated_delta[begin:begin+length] += delta_batch[sample_idx, :length].detach()

                delta = linf_clamp(delta + ((updated_delta - delta) / batch.batchsize), self.eps)

            if (epoch % self.success_every)==0:
                print(f'Check success rate after epoch {epoch}')
                total_samples = 0
                fooled_samples = 0
                loss=0

                for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
                    batch = batch.to(self.asr_brain.device)
                    wav_init, wav_lens = batch.sig
                    max_len = wav_init.shape[1]
                    actual_lens = (wav_lens * max_len).long() 
                    batch_size = batch.batchsize
                    
                    delta_batch = torch.zeros((batch_size, max_len), device=self.asr_brain.device)
                    delta_len = delta.shape[0]

                    longer_mask = actual_lens > delta_len
                    shorter_mask = ~longer_mask
                    
                    if longer_mask.any():
                        delta_x_long = delta.expand(longer_mask.sum(), -1)
                        delta_batch[longer_mask, :delta_len] = delta_x_long
                    if shorter_mask.any():
                        actual_lens_short = actual_lens[shorter_mask]
                        max_start = delta_len - actual_lens_short - 1
                        begins = (torch.rand_like(actual_lens_short, dtype=torch.float) * max_start).floor().long()

                        idxs = torch.arange(max_len, device=delta.device).unsqueeze(0)  # (1, max_len)
                        begins_exp = begins.unsqueeze(1)  # (batch, 1)
                        lens_exp = actual_lens_short.unsqueeze(1)  # (batch, 1)

                        mask = idxs < lens_exp

                        offsets = begins_exp + idxs[:, :lens_exp.max()].to(begins.device)
                        offsets = torch.where(mask, offsets, torch.zeros_like(offsets))

                        delta_x_short = delta[offsets.clamp(0, delta_len - 1)]
                        delta_x_short = delta_x_short * mask
                        
                        shorter_indices = shorter_mask.nonzero(as_tuple=False).squeeze(1)
                        for i, sample_idx in enumerate(shorter_indices):
                            length = actual_lens[sample_idx].item()
                            delta_batch[sample_idx, :length] = delta_x_short[i, :length]
                    
                    batch.sig = wav_init + delta_batch, wav_lens
                    adv_wavs, _ = batch.sig
                    adversarial_texts = [self.asr_brain.modules.whisper.model.transcribe(
                        wav, task="transcribe")["text"] for wav in adv_wavs]
                    
                    batch_cer = torch.tensor([cer(ref, hyp) 
                                            for ref, hyp in zip(batch.wrd, adversarial_texts)])

                    fooled_samples += (batch_cer > self.t).sum().item()
                    total_samples += batch.batchsize
                    loss += self.asr_brain.compute_objectives(
                            predictions, batch, rs.Stage.ATTACK).item()
                    mean_CER = batch_cer.mean().item()

                success_rate = fooled_samples / total_samples
                print(f'MEAN CER IS {mean_CER:.4f}')
                print(f'LOSS IS {(loss/(idx+1)):.4f}')
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    self.univ_perturb.tensor.data = delta.detach()
                    self.checkpointer.save_and_keep_only()                    
                elif success_rate == best_success_rate and mean_CER > best_CER:
                    best_CER = mean_CER
                    self.univ_perturb.tensor.data = delta.detach()
                    print(mean_CER)
                    self.checkpointer.save_and_keep_only()
        print(f"Training finisihed. Best success rate: {best_success_rate:.2f}%") 

    def perturb(self, batch):
        """
        
        Compute an adversarial perturbation
        Arguments
        ---------
        batch : sb.PaddedBatch
           The input batch to perturb

        Returns
        -------
        the tensor of the perturbed batch
        """
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()

        save_device = batch.sig[0].device
        batch = batch.to(self.asr_brain.device)
        save_input = batch.sig[0]
        wav_init = torch.clone(save_input)

        delta = self.univ_perturb.tensor.data.to(self.asr_brain.device)

        if wav_init.shape[1] <= delta.shape[0]:
            delta_x = delta[:wav_init.shape[1]]
        else:
            delta_x = torch.zeros_like(wav_init[0])
            delta_x[:delta.shape[0]] = delta
            
        delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())
        
        background_audio = torch.from_numpy(self.background_audio).to(self.asr_brain.device)       
        
        if wav_init.shape[1] <= background_audio.shape[0]:
            background_x = background_audio[:wav_init.shape[1]]
        else:
            background_x = torch.zeros_like(wav_init[0])
            background_x[:background_audio.shape[0]] = background_audio
            
        background_batch = background_x.unsqueeze(0).expand(wav_init.size())

        wav_adv = wav_init + delta_batch + background_batch
        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()
        return wav_adv.data.to(save_device)
