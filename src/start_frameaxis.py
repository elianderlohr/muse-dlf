import argparse
from ast import arg
import json
import logging
import pandas as pd

from preprocessing.frameaxis_processor import FrameAxisProcessor


def main():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate word embeddings using RoBERTa"
    )
    # save_file_name
    parser.add_argument(
        "--save_file_name",
        type=str,
        required=True,
        help="Path where the embeddings JSON file will be saved",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the data in format: { 'articles': ['article1', 'article2', ...]}",
    )
    parser.add_argument(
        "--path_keywords",
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
        "--roberta_model_path",
        type=str,
        default="roberta-base",
        help="Path to the RoBERTa model or 'roberta-base' for the pre-trained model",
    )
    # path_microframes
    parser.add_argument(
        "--path_microframes",
        type=str,
        default=None,
        help="Path to the pickle file containing the calculated microframes.",
    )
    # blacklisted words
    parser.add_argument(
        "--word_blacklist",
        nargs="+",
        type=str,
        default=[],
        help="List of words to be blacklisted",
    )

    # sample size
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to use from the data",
    )

    args = parser.parse_args()

    logging.info(f"Ignoring the following words: {args.word_blacklist}")

    with open(args.data_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    if args.sample_size:
        df = df.sample(n=args.sample_size)

    frameaxis_processor = FrameAxisProcessor(
        df,
        path_keywords=args.path_keywords,
        save_file_name=args.save_file_name,
        path_microframes=args.path_microframes,
        bert_model_name="roberta-base",
        name_tokenizer="roberta-base",
        path_name_bert_model=args.roberta_model_path,
        force_recalculate=True,
        save_type="pickle",
        dim_names=args.dim_names.split(","),
        word_blacklist=args.word_blacklist,
    )

    frameaxis_df = frameaxis_processor.get_frameaxis_data()

    # print statistics
    logging.info(f"FrameAxis data shape: {frameaxis_df.shape}")
    logging.info(f"FrameAxis data columns: {frameaxis_df.columns}")
    # no of rows with nan
    logging.info(f"FrameAxis data rows with NaN: {frameaxis_df.isnull().sum().sum()}")


if __name__ == "__main__":
    main()
