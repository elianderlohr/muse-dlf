import argparse
import json
import logging
import pandas as pd

from preprocessing.srl_processor import SRLProcessor


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
        "--output_path",
        type=str,
        required=True,
        help="Path where the embeddings JSON file will be saved",
    )
    args = parser.parse_args()

    with open(args.data_path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    srl_processor = SRLProcessor(
        df,
        dataframe_path=args.output_path,
        force_recalculate=True,
        save_type="pickle",
        device=0,
    )

    srl_df = srl_processor.get_srl_embeddings()

    # print statistics
    logging.info(f"FrameAxis data shape: {srl_df.shape}")
    logging.info(f"FrameAxis data columns: {srl_df.columns}")


if __name__ == "__main__":
    main()
