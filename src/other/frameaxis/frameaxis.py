import argparse
import json
import logging
from transformers import RobertaModel, RobertaTokenizer
import torch
import numpy as np
import os
from tqdm import tqdm


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
        "--tfidf_path", type=str, required=True, help="Path to the TF-IDF JSON file"
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

    # Load TF-IDF data
    with open(args.tfidf_path, "r") as file:
        tf_idf = json.load(file)

    # Assuming frame_words data structure is provided; otherwise, it needs to be loaded or defined
    frame_words = {}  # Placeholder, needs actual data

    # Load RoBERTa model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_path)
    model = RobertaModel.from_pretrained(args.model_path)

    def get_word_embeddings(sentence, word):
        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        word_tokens = tokenizer.tokenize(word)
        word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
        word_embeddings = []
        for word_id in word_ids:
            try:
                word_index = (
                    inputs["input_ids"][0].tolist().index(word_id)
                )  # Assumes the word occurs once
                word_embedding = last_hidden_states[0, word_index, :].detach().numpy()
                word_embeddings.append(word_embedding)
            except ValueError:
                logging.warning(f"Word '{word}' not found in the sentence.")
        return (
            np.mean(word_embeddings, axis=0)
            if word_embeddings
            else np.zeros(model.config.hidden_size)
        )

    # Dictionary to store the word embeddings
    embeddings_dict = {}

    # Generate embeddings
    for key, value in tqdm(tf_idf.items(), desc="Processing keys"):
        sorted_words = sorted(value.items(), key=lambda x: x[1], reverse=True)
        embeddings_dict[key] = {}
        for word, score in tqdm(sorted_words, desc=f"Processing words for key: {key}"):
            if word in [fw["word"] for fw in frame_words.get(key, [])]:
                for fw in frame_words.get(key, []):
                    if fw["word"] == word:
                        text = fw["text"]
                        word_embedding = get_word_embeddings(text, word)
                        embeddings_dict[key].setdefault(word, []).append(word_embedding)

    # check if path exists otherwise create it
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Save the embeddings dictionary to a JSON file
    if os.path.isdir(args.output_path):
        logging.error(
            f"The specified output path '{args.output_path}' is a directory. Please specify a path ending with a filename."
        )
    else:
        # Save the embeddings dictionary to a JSON file
        try:
            with open(args.output_path, "w") as outfile:
                json.dump(embeddings_dict, outfile, ensure_ascii=False, indent=4)
            logging.info(f"Embeddings saved to {args.output_path}")
        except Exception as e:
            logging.error(f"Failed to save embeddings: {e}")

    logging.info(f"Embeddings saved to {args.output_path}")


if __name__ == "__main__":
    main()
