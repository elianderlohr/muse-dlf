import argparse
import json
import logging
from transformers import RobertaModel, RobertaTokenizer
import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

from src.preprocessing.frameaxis_processor import FrameAxisProcessor


def main():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate word embeddings using RoBERTa"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the data in format: { 'articles': ['article1', 'article2', ...]}",
    )
    parser.add_argument(
        "--path_antonym_pairs",
        type=str,
        required=True,
        help="Path to the JSON file containing the antonym pairs of shape { 'key': { 'dim1': ['pro_word'], 'dim2': ['anti_word'] } }",
    )
    parser.add_argument(
        "--dim_names",
        type=str,
        required=True,
        help="Name of the dimensions to be used for the frame axis in the format: 'dim1,dim2'",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="roberta-base",
        help="Path to the RoBERTa model or 'roberta-base' for the pre-trained model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the embeddings JSON file will be saved",
    )
    args = parser.parse_args()

    with open(args.data_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    frameaxis_processor = FrameAxisProcessor(
        df,
        dataframe_path=args.output_path,
        force_recalculate=True,
        bert_model_name="roberta-base",
        name_tokenizer="roberta-base",
        path_name_bert_model=args.model_path,
        path_antonym_pairs=args.path_antonym_pairs,
        save_type="pickle",
        dim_names=args.dim_names,
    )

    frameaxis_df = frameaxis_processor.get_frameaxis_data()

    print(frameaxis_df.head())


if __name__ == "__main__":
    main()
