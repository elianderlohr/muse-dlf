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

    # Load RoBERTa model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained(args.model_path)

    def get_antonym_embedding(sentence, words):
        embeddings = {}

        inputs = tokenizer(sentence, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        for word in words:
            word_tokens = tokenizer.tokenize(word)
            word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
            word_embeddings = []
            for word_id in word_ids:
                try:
                    word_index = (
                        inputs["input_ids"][0].tolist().index(word_id)
                    )  # Assumes the word occurs once
                    word_embedding = (
                        last_hidden_states[0, word_index, :].detach().numpy()
                    )
                    word_embeddings.append(word_embedding)
                except ValueError:
                    logging.warning(f"Word '{word}' not found in the sentence.")

            embeddings[word] = (
                np.mean(word_embeddings, axis=0)
                if word_embeddings
                else np.zeros(model.config.hidden_size)
            )

        return embeddings

    print(
        """##################################################
          #                                                #
          #                                                #
          #        WELCOME TO FRAMEAXIS PREPARE            #
          #                                                #
          #                                                #
          ##################################################"""
    )

    print("##################################################")
    print("Step 1: Loading the data...")

    # read json file from data_path
    with open(args.data_path, "r") as f:
        data = json.load(f)

    print("Data loaded successfully!")
    print("Found {} articles".format(len(data["articles"])))

    print("##################################################")

    print("Step 2: Loading the antonym pairs...")

    # read json file from path_antonym_pairs
    with open(args.path_antonym_pairs, "r") as f:
        antonym_pairs = json.load(f)

    print("Antonym pairs loaded successfully!")
    print("Found {} antonym pairs".format(len(antonym_pairs)))
    print("##################################################")

    print("Step 3: Generate antonym word embeddings...")

    # Generate microframes by getting the word embedding of the antonym pairs from the provided data articles

    frame_axis_words = []
    for key, value in antonym_pairs.items():
        for dim, words in value.items():
            frame_axis_words.extend(words)

    # loop over all articles and extract the word embeddings of the antonym pairs
    antonym_embeddings = {}

    for article in data["articles"]:
        embeddings = get_antonym_embedding(article, frame_axis_words)

        # add the embeddings to the microframes based on the word
        for word, embedding in embeddings.items():
            antonym_embeddings.setdefault(word, []).append(embedding)

    print("Microframes generated successfully!")

    print("##################################################")

    print("Step 4: Generate avg vector per word...")

    antonym_avg_embeddings = {}

    for key, value in antonym_pairs.items():
        antonym_avg_embeddings[key] = {}
        for dim, words in value.items():
            antonym_avg_embeddings[key][dim] = {}

            for word in words:
                word_embed = antonym_embeddings[word]

                # get the average of the word embeddings
                avg_word_embed = np.mean(word_embed, axis=0)

                antonym_avg_embeddings[key][dim][word] = avg_word_embed

    print("Avg vector generated successfully!")

    print("##################################################")

    print("Step 5: Generate microframe...")

    microframes = {}

    for key, value in antonym_avg_embeddings.items():
        microframes[key] = {}
        for dim, words in value.items():
            microframes[key][dim] = {}

            # get the average of the word embeddings for the dimension
            dim_embed = np.mean([embed for embed in words.values()], axis=0)

            microframes[key][dim] = dim_embed

    print("Microframe generated successfully!")

    print("##################################################")

    # save the microframes to the output path
    print("Step 6: Saving the microframes...")

    # create the output directory if it does not exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with open(args.output_path + "/mft_microframes.json", "w") as f:
        json.dump(microframes, f)

    print("Microframes saved successfully!")

    print("##################################################")


if __name__ == "__main__":
    main()
