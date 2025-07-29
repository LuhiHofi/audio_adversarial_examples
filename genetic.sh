#! /bin/bash
RSROOT=/path/to/your/repo
export PYTHONPATH="${PYTHONPATH}:${RSROOT}"

SEED=200
NBITER=200
LOAD=False
DATA=test-clean-20

python whisper_attack/run_attack.py attack_configs/whisper/genetic.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED 
# python whisper_attack/run_attack.py attack_configs/whisper/genetic.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED 
# python whisper_attack/run_attack.py attack_configs/whisper/genetic.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED
# python whisper_attack/run_attack.py attack_configs/whisper/genetic.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED

# python whisper_attack/run_attack.py attack_configs/whisper/geneticTargeted.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED
# python whisper_attack/run_attack.py attack_configs/whisper/geneticTargeted.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED
# python whisper_attack/run_attack.py attack_configs/whisper/geneticTargeted.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED
# python whisper_attack/run_attack.py attack_configs/whisper/geneticTargeted.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED