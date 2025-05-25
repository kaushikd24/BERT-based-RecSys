import pandas as pd
from collections import defaultdict
import argparse
import os

def preprocess(input_path, output_path, min_interactions=5):
    df = pd.read_csv(input_path)
    
    
    df = df.sort_values(by=["user_id", "timestamp"])

    
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_interactions].index
    df = df[df["user_id"].isin(valid_users)]

    
    user_sequences = df.groupby("user_id")["item_id"].apply(list)

    
    user_sequences.to_pickle(output_path)
    print(f"Saved processed sequences to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to raw ratings CSV")
    parser.add_argument("--output", type=str, required=True, help="Path to save processed pickle")
    args = parser.parse_args()

    preprocess(args.input, args.output)
