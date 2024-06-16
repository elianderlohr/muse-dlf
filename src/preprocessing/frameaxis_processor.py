import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import (
    BertTokenizerFast,
    BertModel,
    RobertaTokenizerFast,
    RobertaModel,
)
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import string
import json

from utils.logging_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class FrameAxisProcessor:
    def __init__(
        self,
        df,
        path_antonym_pairs="frameaxis/axes/custom.tsv",
        dataframe_path=None,
        path_microframes="",
        bert_model_name="bert-base-uncased",
        name_tokenizer="bert-base-uncased",
        path_name_bert_model="bert-base-uncased",
        force_recalculate=False,
        save_type="pickle",
        dim_names=["positive", "negative"],
        word_blacklist=[],
    ):
        """
        FrameAxisProcessor constructor

        Args:
        df (pd.DataFrame): DataFrame with text data
        path_antonym_pairs (str): Path to the antonym pairs file
        dataframe_path (str): Path to save the dataframe
        path_microframes (str): Path to the microframes
        bert_model_name (str): Name of the BERT model
        name_tokenizer (str): Name of the tokenizer
        path_name_bert_model (str): Path to the BERT model
        force_recalculate (bool): Force recalculation of the frameaxis
        save_type (str): Type of file to save the dataframe
        dim_names (list): List of dimension names
        word_blacklist (list): List of words to blacklist

        Returns:
        None
        """
        self.df = df
        self.force_recalculate = force_recalculate
        self.dataframe_path = dataframe_path
        self.path_microframes = path_microframes

        self.word_blacklist = word_blacklist

        self.lemmatizer = WordNetLemmatizer()

        if bert_model_name == "bert-base-uncased":
            self.tokenizer = BertTokenizerFast.from_pretrained(name_tokenizer)
            self.model = BertModel.from_pretrained(path_name_bert_model)
        elif bert_model_name == "roberta-base":
            self.tokenizer = RobertaTokenizerFast.from_pretrained(name_tokenizer)
            self.model = RobertaModel.from_pretrained(path_name_bert_model)

        if torch.cuda.is_available():
            logger.info("Using CUDA")
            self.model.cuda()

        self.antonym_pairs = {}
        with open(path_antonym_pairs) as f:
            self.antonym_pairs = json.load(f)

        self.dim_names = dim_names

        # Load the stopwords and non-word characters
        nltk.download("stopwords")
        self.stopwords = set(stopwords.words("english"))
        self.non_word_characters = set(string.punctuation)

        # allowed save types
        self.save_type = save_type
        if save_type not in ["csv", "pickle", "json"]:
            raise ValueError(
                "Invalid save_type. Must be one of 'csv', 'pickle', or 'json'."
            )

    def _load_antonym_pairs(self, axis_path):
        axes_df = pd.read_csv(axis_path, sep="\t", header=None)
        return [tuple(x) for x in axes_df.values]

    def precompute_antonym_embeddings(self):
        frame_axis_words = []
        for _, pairs in self.antonym_pairs.items():
            for dim, words in pairs.items():
                frame_axis_words.extend(words)

        antonym_embeddings = {}

        for _, row in tqdm(
            self.df.iterrows(),
            desc="Generating antonym embeddings",
            total=self.df.shape[0],
        ):
            # Access the article text from the 'text' column
            article_text = row["text"]
            embeddings = self.get_embeddings_for_words(
                article_text, frame_axis_words, use_lemmatization=True
            )

            # Add the embeddings to the microframes based on the word
            for word, embedding in embeddings.items():
                antonym_embeddings.setdefault(word, []).append(embedding)

        antonym_avg_embeddings = {}

        for key, value in tqdm(
            self.antonym_pairs.items(), desc="Generating average embeddings"
        ):
            antonym_avg_embeddings[key] = {}
            for dim, words in tqdm(
                value.items(), desc="Processing dimension", leave=False
            ):
                antonym_avg_embeddings[key][dim] = {}

                for word in tqdm(words, desc="Processing word", leave=False):
                    # Ensure the word is in antonym_embeddings to handle cases where it might not be found
                    if word in antonym_embeddings:
                        word_embed = antonym_embeddings[word]

                        # Convert each tensor in word_embed to the appropriate device (GPU if available)
                        word_embed = [
                            embed.to(self.model.device) for embed in word_embed
                        ]

                        # Get the average of the torch word embeddings, ensuring computation happens on the same device
                        avg_word_embed = torch.mean(torch.stack(word_embed), dim=0)

                        antonym_avg_embeddings[key][dim][word] = avg_word_embed

        microframes = {}

        for key, value in tqdm(
            antonym_avg_embeddings.items(), desc="Generating microframes"
        ):
            microframes[key] = {}

            pos_embeddings = antonym_avg_embeddings[key][self.dim_names[0]]
            neg_embeddings = antonym_avg_embeddings[key][self.dim_names[1]]

            pos_embeddings_list = [embed for embed in pos_embeddings.values()]
            neg_embeddings_list = [embed for embed in neg_embeddings.values()]

            # only stack if not empty
            if pos_embeddings_list:
                pos_embedding_avg = torch.mean(torch.stack(pos_embeddings_list), dim=0)
            else:
                pos_embedding_avg = torch.zeros(768)

            if neg_embeddings_list:
                neg_embedding_avg = torch.mean(torch.stack(neg_embeddings_list), dim=0)
            else:
                neg_embedding_avg = torch.zeros(768)

            microframes[key] = {
                self.dim_names[0]: pos_embedding_avg,
                self.dim_names[1]: neg_embedding_avg,
            }

        return microframes

    def calculate_word_contributions(self, df, antonym_pairs_embeddings):
        """
        Calculates the bias scores for each word in each document and aggregates them into a list of dictionaries.
        :param df: A DataFrame containing the articles.
        :param antonym_pairs_embeddings: A dictionary containing the embeddings for antonym pairs for each dimension.
        :return: A DataFrame with each row containing a list of dictionaries, each representing a word and its corresponding bias score.
        """

        def calculate_word_contribution(article_id, text, method="cosine"):
            words, embeddings = self.get_embeddings_for_text(text)

            if embeddings.numel() == 0:
                print(f"No embeddings found for article {article_id}, words: {words}")
                return []

            # List to collect word contribution dictionaries
            word_contributions = []

            for word, embedding in zip(words, embeddings):
                word_dict = {"word": word}
                for dimension in antonym_pairs_embeddings:
                    pos_embedding = antonym_pairs_embeddings[dimension][
                        self.dim_names[0]
                    ]
                    neg_embedding = antonym_pairs_embeddings[dimension][
                        self.dim_names[1]
                    ]
                    vf = (pos_embedding - neg_embedding).to(self.model.device)

                    vw = embedding.to(self.model.device)

                    if method == "cosine":
                        # Calculate cosine similarity using the formula provided
                        cos_sim = (
                            F.cosine_similarity(vw.unsqueeze(0), vf.unsqueeze(0))
                            .cpu()
                            .item()
                        )

                        word_dict[dimension] = cos_sim
                    if method == "projection":
                        projection = torch.dot(embedding, vf) / torch.norm(vf)

                        word_dict[dimension] = projection.cpu().item()

                word_contributions.append(word_dict)

            return word_contributions

        tqdm.pandas(desc="Calculating Word Contributions")
        df["word_contributions"] = df.progress_apply(
            lambda row: calculate_word_contribution(row["article_id"], row["text"]),
            axis=1,
        )

        return df

    def calculate_microframe_bias(self, df):
        # Initialize a DataFrame to collect microframe bias results
        bias_results = []

        # Iterate over each row in the DataFrame
        for idx, row in df.iterrows():
            # Each 'word_contributions' entry is a list of dictionaries with words and their contributions
            word_contributions = row["word_contributions"]

            # Initialize a dictionary to hold bias calculations for this article
            bias_dict = {"article_id": row["article_id"]}

            if word_contributions:
                dimensions = [
                    k for k in word_contributions[0].keys() if k not in ["word"]
                ]
                for dimension in dimensions:
                    # Calculate weighted contributions for each dimension
                    weighted_contributions = sum(
                        d[dimension] for d in word_contributions if dimension in d
                    )

                    # Calculate microframe bias for the dimension
                    microframe_bias = weighted_contributions / len(word_contributions)
                    bias_dict[dimension + "_bias"] = microframe_bias

            # Append the results for this article to the results list
            bias_results.append(bias_dict)

        # Convert the results list to a DataFrame
        bias_df = pd.DataFrame(bias_results)
        bias_df = bias_df.set_index("article_id")

        return bias_df

    def calculate_baseline_bias(self, df):
        """
        Calculate the baseline microframe bias for the entire corpus.

        :param df: A DataFrame with columns for article_id, word, and microframe cosine similarities.
        :return: A dictionary of baseline biases for each microframe dimension.
        """

        baseline_bias = {}
        for idx, row in df.iterrows():
            word_contributions = row["word_contributions"]

            if word_contributions:
                dimensions = [
                    k for k in word_contributions[0].keys() if k not in ["word"]
                ]
                for dimension in dimensions:
                    baseline_bias.setdefault(dimension, []).extend(
                        [d[dimension] for d in word_contributions if dimension in d]
                    )

        for dimension in baseline_bias:
            baseline_bias[dimension] = sum(baseline_bias[dimension]) / len(
                baseline_bias[dimension]
            )

        return baseline_bias

    def calculate_microframe_intensity(self, df):
        """
        Calculate the microframe intensity for each document in the DataFrame.

        :param df: A DataFrame containing the word contributions and article IDs.
        :return: A DataFrame with the microframe intensity for each article and dimension.
        """
        # First, calculate the baseline bias for the corpus
        baseline_bias = self.calculate_baseline_bias(df)

        # Initialize DataFrame to store intensity results
        intensity_df = pd.DataFrame()

        for idx, row in df.iterrows():
            word_contributions = row["word_contributions"]

            # Initialize a DataFrame to store the intensity results for this article
            intensity_dict = {"article_id": row["article_id"]}
            total_contributions = len(word_contributions)
            if word_contributions:
                for dimension in [
                    k for k in word_contributions[0].keys() if k not in ["word"]
                ]:
                    # Calculate the second moment for the dimension
                    deviations_squared = sum(
                        (d[dimension] - baseline_bias[dimension]) ** 2
                        for d in word_contributions
                        if dimension in d
                    )
                    microframe_intensity = (
                        deviations_squared / total_contributions
                        if total_contributions
                        else 0
                    )

                    # Store the results
                    intensity_dict[dimension + "_intensity"] = microframe_intensity

            # Append to the intensity DataFrame
            intensity_df = pd.concat(
                [intensity_df, pd.DataFrame([intensity_dict])], ignore_index=True
            )

        intensity_df = intensity_df.set_index("article_id")

        return intensity_df

    def calculate_all_metrics(self, df, antonym_pairs_embeddings):
        """
        Executes the calculation of word contributions, microframe bias, and microframe intensity for each article.

        :param df: A DataFrame containing articles with 'article_id' and 'text' columns.
        :param antonym_pairs_embeddings: A dictionary containing the embeddings for antonym pairs for each dimension.
        :return: A DataFrame with the structure article_id | dim1_bias | dim1_intensity | ...
        """

        logger.info("Calculating all metrics...")
        logger.info("Step 1: Calculating word contributions...")
        logger.info("     DEBUG: df shape: " + str(df.shape))
        # Step 1: Calculate word contributions for each article and dimension
        word_contributions_df = self.calculate_word_contributions(
            df, antonym_pairs_embeddings
        )

        logger.info(
            "     DEBUG: word_contributions_df shape: "
            + str(word_contributions_df.shape)
        )

        # create new file name for contributions by append contributions to filename
        contributions_filename = self.dataframe_path.replace(
            ".pkl", "_contributions.pkl"
        )

        # dump to pickle
        with open(contributions_filename, "wb") as f:
            print("Saving contributions to " + contributions_filename)
            pickle.dump(word_contributions_df, f)

        logger.info("Step 2: Calculating microframe bias...")
        # Step 2: Calculate microframe bias for each article and dimension
        microframe_bias_df = self.calculate_microframe_bias(word_contributions_df)

        logger.info(
            "     DEBUG: microframe_bias_df shape: " + str(microframe_bias_df.shape)
        )

        logger.info("Step 3: Calculating microframe intensity...")
        # Step 3: Calculate microframe intensity for each article and dimension
        microframe_intensity_df = self.calculate_microframe_intensity(
            word_contributions_df
        )

        logger.info(
            "     DEBUG: microframe_intensity_df shape: "
            + str(microframe_intensity_df.shape)
        )

        logger.info("Step 4: Merging bias and intensity dataframes...")
        # Merge the bias and intensity DataFrames row-wise
        final_df = pd.concat([microframe_bias_df, microframe_intensity_df], axis=1)

        logger.info("     DEBUG: final_df shape: " + str(final_df.shape))

        # Reformat the final DataFrame to match the desired structure
        final_df.reset_index(inplace=True)
        final_columns = ["article_id"]
        for dimension in [
            col.replace("_bias", "")
            for col in microframe_bias_df.columns
            if "_bias" in col
        ]:
            final_columns.append(dimension + "_bias")
            final_columns.append(dimension + "_intensity")
        final_df = final_df[final_columns]

        return final_df

    def get_embeddings_for_text(
        self,
        text,
        remove_stopwords=True,
        remove_non_words=True,
        remove_numbers=True,
        remove_word_blacklist=True,
    ):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        ).to(self.model.device)

        # Obtain the embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.squeeze(0)

        # Initialize lists for filtered tokens' embeddings and words
        filtered_embeddings = []
        filtered_words = []

        # Obtain a list mapping words to tokens
        word_ids = inputs.word_ids()

        for w_idx in set(word_ids):
            if w_idx is None:
                continue

            # Obtain the start and end token positions for the current word
            word_tokens_range = inputs.word_to_tokens(w_idx)

            if word_tokens_range is None:
                continue

            start, end = word_tokens_range

            # Reconstruct the word from tokens to check against stopwords and non-word characters
            word = self.tokenizer.decode(inputs.input_ids[0][start:end])

            # Normalize the word for checks
            normalized_word = word.lower().strip(string.punctuation).strip()

            if remove_stopwords and normalized_word in self.stopwords:
                continue

            if remove_non_words and all(
                char in self.non_word_characters for char in normalized_word
            ):
                continue

            if remove_numbers and normalized_word.isnumeric():
                continue

            if remove_word_blacklist and normalized_word in self.word_blacklist:
                continue

            # If the word passes the filters, append its embeddings and the word itself
            word_embeddings = embeddings[start:end]
            filtered_embeddings.append(word_embeddings.mean(dim=0))
            filtered_words.append(normalized_word)

        # Stack the filtered embeddings
        filtered_embeddings_tensor = (
            torch.stack(filtered_embeddings)
            if filtered_embeddings
            else torch.tensor([])
        )

        # if no embeddings were found, logger.info debug info
        if filtered_embeddings_tensor.numel() == 0:
            logger.info(
                f"No embeddings found for input text: {text}, after filtering: {filtered_words}"
            )

        return filtered_words, filtered_embeddings_tensor

    def get_embeddings_for_words(self, sentence, words, use_lemmatization=False):
        """
        Get the contextualized embeddings for a list of words using the sentence as context.

        :param sentence: The sentence to get embeddings from.
        :param words: A list of words to get embeddings for.
        :param use_lemmatization: Boolean to control lemmatization.
        :return: A dictionary containing the average embeddings for each word.
        """
        sentence_words, word_embeddings = self.get_embeddings_for_text(
            sentence, remove_stopwords=False, remove_non_words=False
        )

        if use_lemmatization:
            sentence_words = [
                self.lemmatizer.lemmatize(word) for word in sentence_words
            ]
            words = [self.lemmatizer.lemmatize(word) for word in words]

        # Initialize dictionary to hold word embeddings
        embeddings = {}

        # Iterate over each word to get its embedding
        for word in words:
            if word in sentence_words:
                word_idx = sentence_words.index(word)

                embedding = word_embeddings[word_idx]

                embeddings[word] = embedding

        return embeddings

    def get_frameaxis_data(self):
        """
        Calculate the FrameAxis Values for the DataFrame

        Returns:
        pd.DataFrame: DataFrame with FrameAxis Embeddings
        """
        # check if self.dataframe_path is None
        if not self.force_recalculate and (
            not self.dataframe_path or not os.path.exists(self.dataframe_path)
        ):
            self.force_recalculate = True

        if self.force_recalculate:
            logger.info("Calculating FrameAxis Embeddings")

            if (
                self.path_microframes
                and len(self.path_microframes) > 0
                and os.path.exists(self.path_microframes)
            ):
                antonym_pairs_embeddings_filename = self.path_microframes
            else:
                # create new file name for antonym_pairs_embeddings by append antonym_pairs_embeddings to filename
                antonym_pairs_embeddings_filename = self.dataframe_path.replace(
                    ".pkl", "_antonym_embeddings.pkl"
                )

            # load from frameaxis_antonym_embeddings if exists
            if os.path.exists(antonym_pairs_embeddings_filename):
                logger.info("Loading FrameAxis Embeddings")
                with open(antonym_pairs_embeddings_filename, "rb") as f:
                    antonym_pairs_embeddings = pickle.load(f)
            else:
                logger.info("Precomputing FrameAxis Embeddings")
                antonym_pairs_embeddings = self.precompute_antonym_embeddings()

                # dump to pickle
                with open(antonym_pairs_embeddings_filename, "wb") as f:
                    logger.info(
                        "Saving FrameAxis Embeddings to "
                        + antonym_pairs_embeddings_filename
                    )
                    pickle.dump(antonym_pairs_embeddings, f)

            frameaxis_df = self.calculate_all_metrics(self.df, antonym_pairs_embeddings)

            if self.dataframe_path:
                if self.save_type == "csv":
                    frameaxis_df.to_csv(self.dataframe_path, index=False)
                if self.save_type == "json":
                    frameaxis_df.to_json(self.dataframe_path)
                elif self.save_type == "pickle":
                    with open(self.dataframe_path, "wb") as f:
                        pickle.dump(frameaxis_df, f)

            return frameaxis_df
        else:
            # load from pickle
            logger.info("Loading FrameAxis Embeddings")
            with open(self.dataframe_path, "rb") as f:
                frameaxis_df = pickle.load(f)

            return frameaxis_df
