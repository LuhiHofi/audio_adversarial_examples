#! /bin/bash
RSROOT=/path/to/your/repo
export PYTHONPATH="${PYTHONPATH}:${RSROOT}"

NBITER=2000
SEED=2000
LOAD=False
EPS=0.1
MAXDECR=8
DATA=test-clean-20
CONF=0.0
DECRFACTOR=0.7
CST=4
LR=0.01

python whisper_attack/run_attack.py attack_configs/whisper/cw_modified.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=tiny --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python whisper_attack/run_attack.py attack_configs/whisper/cw_modified.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python whisper_attack/run_attack.py attack_configs/whisper/cw_modified.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
python whisper_attack/run_attack.py attack_configs/whisper/cw_modified.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium --nb_iter=$NBITER --load_audio=$LOAD --seed=$SEED  --eps=$EPS --const=$CST --max_decr=$MAXDECR --lr=$LR --confidence=$CONF --decrease_factor_eps=$DECRFACTOR
