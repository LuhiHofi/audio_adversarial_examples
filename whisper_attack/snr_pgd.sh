#! /bin/bash
RSROOT=/path/to/your/repo
export PYTHONPATH="${PYTHONPATH}:${RSROOT}"

NBITER=200
LOAD=False
DATA=test-clean-100
SNR=35
SEED=235

python whisper_attack/run_attack.py attack_configs/whisper/snr_pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --snr=$SNR 
python whisper_attack/run_attack.py attack_configs/whisper/snr_pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --snr=$SNR
python whisper_attack/run_attack.py attack_configs/whisper/snr_pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --snr=$SNR
python whisper_attack/run_attack.py attack_configs/whisper/snr_pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --snr=$SNR
SNR=40
SEED=240
python whisper_attack/run_attack.py attack_configs/whisper/snr_pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --snr=$SNR
python whisper_attack/run_attack.py attack_configs/whisper/snr_pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --snr=$SNR
python whisper_attack/run_attack.py attack_configs/whisper/snr_pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --snr=$SNR
python whisper_attack/run_attack.py attack_configs/whisper/snr_pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --snr=$SNR
