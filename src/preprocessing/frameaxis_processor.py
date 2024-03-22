import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertModel, RobertaTokenizerFast, RobertaModel
from nltk.corpus import stopwords
import nltk
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import string
import json


class FrameAxisProcessor:
    def __init__(
        self,
        df,
        path_antonym_pairs="frameaxis/axes/custom.tsv",
        dataframe_path=None,
        bert_model_name="bert-base-uncased",
        name_tokenizer="bert-base-uncased",
        path_name_bert_model="bert-base-uncased",
        force_recalculate=False,
        save_type="pickle",
        dim_names=["positive", "negative"],
    ):
        """
        FrameAxisProcessor constructor

        Args:
        df (pd.DataFrame): DataFrame with text data
        path_antonym_pairs (str): Path to the antonym pairs file
        dataframe_path (str): Path to save the FrameAxis Embeddings DataFrame for saving and loading
        name_tokenizer (str): Name or path of the model
        path_name_bert_model (str): Name or path of the model
        force_recalculate (bool): If True, recalculate the FrameAxis Embeddings
        save_type (str): Type of file to save the FrameAxis Embeddings DataFrame
        """
        self.df = df
        self.force_recalculate = force_recalculate
        self.dataframe_path = dataframe_path

        if bert_model_name == "bert-base-uncased":
            self.tokenizer = BertTokenizer.from_pretrained(name_tokenizer)
            self.model = BertModel.from_pretrained(path_name_bert_model)
        elif bert_model_name == "roberta-base":
            self.tokenizer = RobertaTokenizerFast.from_pretrained(name_tokenizer)
            self.model = RobertaModel.from_pretrained(path_name_bert_model)

        if torch.cuda.is_available():
            print("Using CUDA")
            self.model.cuda()

        antonym_pairs = {}
        with open(path_antonym_pairs) as f:
            antonym_pairs = json.load(f)

        self.dim_names = dim_names

        self.antonym_pairs_embeddings = self.precompute_antonym_embeddings(
            antonym_pairs
        )

        self.save_type = save_type

        # allowed save types
        if save_type not in ["csv", "pickle", "json"]:
            raise ValueError(
                "Invalid save_type. Must be one of 'csv', 'pickle', or 'json'."
            )

    def _load_antonym_pairs(self, axis_path):
        axes_df = pd.read_csv(axis_path, sep="\t", header=None)
        return [tuple(x) for x in axes_df.values]

    def precompute_antonym_embeddings(self, antonym_pairs):
        frame_axis_words = []
        for dimension, pairs in antonym_pairs.items():
            for dim, words in pairs.items():
                frame_axis_words.extend(words)

        antonym_embeddings = {}

        for index, row in tqdm(
            self.df.iterrows(),
            desc="Generating antonym embeddings",
            total=self.df.shape[0],
        ):
            # Access the article text from the 'text' column
            article_text = row["text"]
            embeddings = self._get_contextualized_embedding(
                article_text, frame_axis_words
            )

            # Add the embeddings to the microframes based on the word
            for word, embedding in embeddings.items():
                antonym_embeddings.setdefault(word, []).append(embedding)

        antonym_avg_embeddings = {}

        for key, value in tqdm(
            antonym_pairs.items(), desc="Generating average embeddings"
        ):
            antonym_avg_embeddings[key] = {}
            for dim, words in tqdm(value.items(), desc="Processing dimension"):
                antonym_avg_embeddings[key][dim] = {}

                for word in tqdm(words, desc="Processing word"):
                    # Ensure the word is in antonym_embeddings to handle cases where it might not be found
                    if word in antonym_embeddings:
                        word_embed = antonym_embeddings[word]

                        # Get the average of the word embeddings
                        avg_word_embed = np.mean(word_embed, axis=0)

                        antonym_avg_embeddings[key][dim][word] = avg_word_embed

        microframes = {}

        for key, value in tqdm(
            antonym_avg_embeddings.items(), desc="Generating microframes"
        ):
            microframes[key] = {}

            pos_embeddings = antonym_avg_embeddings[key][self.dim_names[0]]
            neg_embeddings = antonym_avg_embeddings[key][self.dim_names[1]]

            pos_embedding_avg = torch.mean(torch.stack(pos_embeddings), dim=0)
            neg_embedding_avg = torch.mean(torch.stack(neg_embeddings), dim=0)

            microframes[key] = {
                self.dim_names[0]: pos_embedding_avg,
                self.dim_names[1]: neg_embedding_avg,
            }

        return microframes

    def _calculate_cosine_similarities(self, df):
        def process_row(row):
            sentence_embeddings, words = self._get_embeddings(row["text"])
            cos_sims = {}

            for dimension, embeddings in self.antonym_pairs_embeddings.items():
                pos_embedding = embeddings[self.dim_names[0]].to(self.model.device)
                neg_embedding = embeddings[self.dim_names[1]].to(self.model.device)
                diff_vector = neg_embedding - pos_embedding

                sims = []
                for word_embedding in sentence_embeddings:
                    cos_sim = (
                        1
                        - cosine_similarity(
                            diff_vector.cpu().numpy().reshape(1, -1),
                            word_embedding.cpu().numpy().reshape(1, -1),
                        )[0][0]
                    )
                    sims.append(cos_sim)

                # Ensure dimension names are unique by appending a suffix if needed and replace spaces with underscores
                dimension_name = (
                    dimension.replace(" ", "_").replace("-", "_").replace(",", "")
                )

                # to lower case
                dimension_name = dimension_name.lower()

                cos_sims[dimension_name] = np.mean(sims)

            return pd.Series(cos_sims)

        tqdm.pandas(desc="Calculating Cosine Similarities")
        cos_sim_columns = df.progress_apply(process_row, axis=1)

        # Before joining, check if any column names would overlap and adjust if necessary
        overlapping_columns = df.columns.intersection(cos_sim_columns.columns)
        if not overlapping_columns.empty:
            cos_sim_columns = cos_sim_columns.rename(
                columns={col: col + "_cos_sim" for col in overlapping_columns}
            )

        return df[["article_id"]].join(cos_sim_columns)

    def _get_embeddings(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.squeeze(0)
        stop_words = set(stopwords.words("english"))
        token_ids = inputs["input_ids"].squeeze(0)
        words = [
            self.tokenizer.decode([token_id]).strip(string.punctuation)
            for token_id in token_ids
        ]

        filtered_embeddings = []
        filtered_words = []
        for word, embedding, token_id in zip(words, embeddings, token_ids):
            if (
                token_id not in self.tokenizer.all_special_ids
                and word.lower() not in stop_words
                and word.isalpha()
            ):
                filtered_embeddings.append(embedding)
                filtered_words.append(word)

        return filtered_embeddings, filtered_words

    def _get_contextualized_embedding(self, sentence, words):
        """
        Get the contextualized embedding of a word by extracting it from text with its surrounding context
        """
        embeddings = {}

        inputs = self.tokenizer(sentence, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        for word in words:
            word_tokens = self.tokenizer.tokenize(word)
            word_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            word_embeddings = []
            for word_id in word_ids:
                try:
                    word_index = (
                        inputs["input_ids"][0].tolist().index(word_id)
                    )  # Assumes the word occurs once
                    word_embedding = last_hidden_states[0, word_index, :].detach()
                    word_embeddings.append(word_embedding)
                except ValueError:
                    pass

            embeddings[word] = (
                torch.mean(torch.stack(word_embeddings), dim=0)
                if word_embeddings
                else torch.zeros(self.model.config.hidden_size)
            )

        return embeddings

    def get_frameaxis_data(self):
        """
        Calculate the FrameAxis Values for the DataFrame

        Returns:
        pd.DataFrame: DataFrame with FrameAxis Embeddings
        """
        if self.force_recalculate or self.dataframe_path is None:
            print("Calculating FrameAxis Embeddings")

            nltk.download("stopwords")

            frameaxis_df = self._calculate_cosine_similarities(self.df)

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
            print("Loading FrameAxis Embeddings")
            with open(self.dataframe_path, "rb") as f:
                frameaxis_df = pickle.load(f)

            return frameaxis_df
