#! /bin/bash
RSROOT=~/path/to/your/repo
export PYTHONPATH="${PYTHONPATH}:${RSROOT}"

EPOCHS=100
NBITER=200
SEED=35
EVERY=10
BATCH=64
EPS=0.02
EPS_ITEMS=0.1
RELEPS=0.05
BACKGROUND_AUDIO_PATH=audio_backgrounds/raining-in-backyard.wav  # Path to the background audio file

LOAD=False
DATA=test-clean-100
TRAIN_DATA=train-clean-100-128
params_subfolder=CKPT+2025-07-12+16-25-28+00

# python whisper_attack/fit_attacker.py attack_configs/whisper/univ_noise_fit.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=tiny --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY
# python whisper_attack/fit_attacker.py attack_configs/whisper/univ_noise_fit.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=base --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY
# python whisper_attack/fit_attacker.py attack_configs/whisper/univ_noise_fit.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=small --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY
python whisper_attack/fit_attacker.py attack_configs/whisper/univ_noise_fit.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=medium --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY

# python whisper_attack/fit_attacker.py attack_configs/whisper/univ_noise_fit.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=tiny --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY --params_subfolder=$params_subfolder
# python whisper_attack/fit_attacker.py attack_configs/whisper/univ_noise_fit.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=base --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY --params_subfolder=$params_subfolder
# python whisper_attack/fit_attacker.py attack_configs/whisper/univ_noise_fit.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=small --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY --params_subfolder=$params_subfolder
# python whisper_attack/fit_attacker.py attack_configs/whisper/univ_noise_fit.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --epochs=$EPOCHS --batch_size=$BATCH --data_csv_name=$DATA --data_csv_name_train=$TRAIN_DATA --model_label=medium --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --rel_eps_iter=$RELEPS --success_every=$EVERY --params_subfolder=$params_subfolder

# python whisper_attack/run_attack.py attack_configs/whisper/univ_noise.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --model_label=tiny --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --params_subfolder=$params_subfolder
# python whisper_attack/run_attack.py attack_configs/whisper/univ_noise.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --model_label=base --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --params_subfolder=$params_subfolder
# python whisper_attack/run_attack.py attack_configs/whisper/univ_noise.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --model_label=small --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --params_subfolder=$params_subfolder
# python whisper_attack/run_attack.py attack_configs/whisper/univ_noise.yaml --background_audio_path=$BACKGROUND_AUDIO_PATH --model_label=medium --data_csv_name=$DATA --root=$RSROOT --load_audio=$LOAD --seed=$SEED --nb_iter=$NBITER --eps=$EPS --params_subfolder=$params_subfolder
