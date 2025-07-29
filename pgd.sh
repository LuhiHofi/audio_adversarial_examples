#! /bin/bash
RSROOT=/path/to/your/project
export PYTHONPATH="${PYTHONPATH}:${RSROOT}"

NBITER=200
LOAD=False
DATA=test-clean-100
EPS=0.01
SEED=235

python whisper_attack/run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --eps=$EPS 
python whisper_attack/run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --eps=$EPS
python whisper_attack/run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --eps=$EPS
python whisper_attack/run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --eps=$EPS

EPS=0.03
SEED=240
python whisper_attack/run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --eps=$EPS
python whisper_attack/run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --eps=$EPS
python whisper_attack/run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --eps=$EPS
python whisper_attack/run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --eps=$EPS
