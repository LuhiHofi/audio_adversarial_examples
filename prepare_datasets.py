from robust_speech.data.librispeech import prepare_librispeech 
import argparse
from pathlib import Path
import os

parser = argparse.ArgumentParser()
parser.add_argument("--libri_speech_path", default='path_to_repository/data/LibriSpeech', 
                    type=str, help="Path to the LibriSpeech dataset.")


train_splits = ['train-clean-100']
test_splits = ['test-clean']

if __name__ == '__main__':
    args = parser.parse_args()
    
    data_folder = args.libri_speech_path
    save_folder = args.libri_speech_path + '/csv'

    # Ensure the save folder exists
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    # Prepare the datasets
    prepare_librispeech(
        data_folder=os.path.expanduser(data_folder),
        save_folder=os.path.expanduser(save_folder),
        tr_splits=train_splits,
        te_splits=test_splits,
    )