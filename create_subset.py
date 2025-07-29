"""
This script creates a subset of a CSV file with a specified number of samples.
"""

import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", default="/home/luhi/lukas-hofman/data/LibriSpeech/csv/test-clean.csv", 
                    type=str, help="Path to the CSV file.")
parser.add_argument("--num_samples", default=100, type=int, help="Number of samples to include in the subset.")


def create_sample_csv(num_samples, input_csv_path):
    df = pd.read_csv(input_csv_path)
    output_csv_path = input_csv_path.replace(".csv", f"-{num_samples}.csv")
    df_sample = df.sample(n=num_samples, random_state=42)
    
    df_sample.to_csv(output_csv_path, index=False)

def main(args: argparse.Namespace) -> None:
    create_sample_csv(args.num_samples, input_csv_path=args.csv_path)
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)