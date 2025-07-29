#! /bin/bash
RSROOT=/path/to/your/repo
export PYTHONPATH="${PYTHONPATH}:${RSROOT}"

DATA=test-clean-100
SIGMA=0.005
python whisper_attack/run_attack.py attack_configs/whisper/rand.yaml --data_csv_name=$DATA --model_label=tiny --root=$RSROOT --sigma=$SIGMA --save_audio=False --load_audio=False
python whisper_attack/run_attack.py attack_configs/whisper/rand.yaml --data_csv_name=$DATA --model_label=base --root=$RSROOT --sigma=$SIGMA --save_audio=False --load_audio=False
python whisper_attack/run_attack.py attack_configs/whisper/rand.yaml --data_csv_name=$DATA --model_label=small --root=$RSROOT --sigma=$SIGMA --save_audio=False --load_audio=False
python whisper_attack/run_attack.py attack_configs/whisper/rand.yaml --data_csv_name=$DATA --model_label=medium --root=$RSROOT --sigma=$SIGMA --save_audio=False --load_audio=False
