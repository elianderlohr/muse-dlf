import os
import io
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


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


class FrameAxisProcessor:
    def __init__(
        self,
        df,
        path_antonym_pairs="frameaxis/axes/custom.tsv",
        save_path=None,
        path_microframes=None,
        bert_model_name="bert-base-uncased",
        name_tokenizer="bert-base-uncased",
        path_name_bert_model="bert-base-uncased",
        force_recalculate=False,
        save_type="pickle",
        dim_names=["positive", "negative"],
        word_blacklist=[],
    ):
        self.df = df
        self.force_recalculate = force_recalculate
        self.save_path = save_path
        self.path_microframes = path_microframes
        self.path_antonym_pairs = path_antonym_pairs
        self.word_blacklist = word_blacklist
        self.dim_names = dim_names
        self.save_type = save_type
        self.lemmatizer = WordNetLemmatizer()

        self._initialize_tokenizer_and_model(
            bert_model_name, name_tokenizer, path_name_bert_model
        )
        self._load_antonym_pairs()
        self._initialize_stopwords_and_non_word_chars()
        self._validate_save_type()

    def _initialize_tokenizer_and_model(
        self, bert_model_name, name_tokenizer, path_name_bert_model
    ):
        if bert_model_name == "bert-base-uncased":
            self.tokenizer = BertTokenizerFast.from_pretrained(name_tokenizer)
            self.model = BertModel.from_pretrained(path_name_bert_model)
        elif bert_model_name == "roberta-base":
            self.tokenizer = RobertaTokenizerFast.from_pretrained(name_tokenizer)
            self.model = RobertaModel.from_pretrained(path_name_bert_model)

        if torch.cuda.is_available():
            logger.info("Using CUDA")
            self.model.cuda()
        else:
            logger.info("Using CPU")
            self.model.to("cpu")

    def _load_antonym_pairs(self):
        with open(self.path_antonym_pairs) as f:
            self.antonym_pairs = json.load(f)

    def _initialize_stopwords_and_non_word_chars(self):
        nltk.download("stopwords")
        self.stopwords = set(stopwords.words("english"))
        self.non_word_characters = set(string.punctuation)

    def _validate_save_type(self):
        allowed_save_types = ["csv", "pickle", "json"]
        if self.save_type not in allowed_save_types:
            raise ValueError(f"Invalid save_type. Must be one of {allowed_save_types}")

    def _save_to_file(self, data, path, file_type):
        if file_type == "csv":
            data.to_csv(path, index=False)
        elif file_type == "json":
            data.to_json(path)
        elif file_type == "pickle":
            with open(path, "wb") as f:
                pickle.dump(data, f)

    def _load_from_file(self, path, file_type):
        if file_type == "csv":
            return pd.read_csv(path)
        elif file_type == "json":
            return pd.read_json(path)
        elif file_type == "pickle":
            with open(path, "rb") as f:
                if torch.cuda.is_available():
                    return pickle.load(f)
                else:
                    return CPU_Unpickler(f).load()

    def _get_save_path(self, suffix):
        if self.save_path is None:
            raise ValueError(
                "save_path is None, cannot generate save path with suffix."
            )
        base, ext = os.path.splitext(self.save_path)
        return f"{base}_{suffix}{ext}"

    def _load_or_calculate_antonym_embeddings(self):
        antonym_pairs_embeddings_filename = (
            self.path_microframes
            if self.path_microframes
            else self._get_save_path("antonym_embeddings")
        )

        if os.path.exists(antonym_pairs_embeddings_filename):
            logger.info("Loading antonym embeddings from file.")
            return self._load_from_file(antonym_pairs_embeddings_filename, "pickle")

        logger.info("Calculating antonym embeddings.")
        antonym_pairs_embeddings = self.precompute_antonym_embeddings()
        if self.save_path:
            self._save_to_file(
                antonym_pairs_embeddings, antonym_pairs_embeddings_filename, "pickle"
            )
        return antonym_pairs_embeddings

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
            article_text = row["text"]
            embeddings = self.get_embeddings_for_words(
                article_text, frame_axis_words, use_lemmatization=True
            )
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
                    if word in antonym_embeddings:
                        word_embed = antonym_embeddings[word]
                        word_embed = [
                            embed.to(self.model.device) for embed in word_embed
                        ]
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

            pos_embedding_avg = (
                torch.mean(torch.stack(pos_embeddings_list), dim=0)
                if pos_embeddings_list
                else torch.zeros(768)
            )
            neg_embedding_avg = (
                torch.mean(torch.stack(neg_embeddings_list), dim=0)
                if neg_embeddings_list
                else torch.zeros(768)
            )

            microframes[key] = {
                self.dim_names[0]: pos_embedding_avg,
                self.dim_names[1]: neg_embedding_avg,
            }

        return microframes

    def calculate_word_contributions(self, df, antonym_pairs_embeddings):
        def calculate_word_contribution(article_id, text, method="cosine"):
            words, embeddings = self.get_embeddings_for_text(text)

            if embeddings.numel() == 0:
                logger.info(
                    f"No embeddings found for article {article_id}, words: {words}"
                )
                return []

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
        bias_results = []

        for idx, row in df.iterrows():
            word_contributions = row["word_contributions"]
            bias_dict = {"article_id": row["article_id"]}

            if word_contributions:
                dimensions = [
                    k for k in word_contributions[0].keys() if k not in ["word"]
                ]
                for dimension in dimensions:
                    weighted_contributions = sum(
                        d[dimension] for d in word_contributions if dimension in d
                    )
                    microframe_bias = weighted_contributions / len(word_contributions)
                    bias_dict[dimension + "_bias"] = microframe_bias

            bias_results.append(bias_dict)

        bias_df = pd.DataFrame(bias_results)
        bias_df = bias_df.set_index("article_id")

        return bias_df

    def calculate_baseline_bias(self, df):
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
        baseline_bias = self.calculate_baseline_bias(df)
        intensity_df = pd.DataFrame()

        for idx, row in df.iterrows():
            word_contributions = row["word_contributions"]
            intensity_dict = {"article_id": row["article_id"]}
            total_contributions = len(word_contributions)
            if word_contributions:
                for dimension in [
                    k for k in word_contributions[0].keys() if k not in ["word"]
                ]:
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
                    intensity_dict[dimension + "_intensity"] = microframe_intensity

            intensity_df = pd.concat(
                [intensity_df, pd.DataFrame([intensity_dict])], ignore_index=True
            )

        intensity_df = intensity_df.set_index("article_id")

        return intensity_df

    def calculate_all_metrics(self, df, antonym_pairs_embeddings):
        logger.info("Calculating all metrics...")
        logger.info("Step 1: Calculating word contributions...")
        logger.info("     DEBUG: df shape: " + str(df.shape))
        word_contributions_df = self.calculate_word_contributions(
            df, antonym_pairs_embeddings
        )
        logger.info(
            "     DEBUG: word_contributions_df shape: "
            + str(word_contributions_df.shape)
        )

        if self.save_path:
            contributions_filename = self._get_save_path("contributions")
            self._save_to_file(word_contributions_df, contributions_filename, "pickle")

        logger.info("Step 2: Calculating microframe bias...")
        microframe_bias_df = self.calculate_microframe_bias(word_contributions_df)
        logger.info(
            "     DEBUG: microframe_bias_df shape: " + str(microframe_bias_df.shape)
        )

        logger.info("Step 3: Calculating microframe intensity...")
        microframe_intensity_df = self.calculate_microframe_intensity(
            word_contributions_df
        )
        logger.info(
            "     DEBUG: microframe_intensity_df shape: "
            + str(microframe_intensity_df.shape)
        )

        logger.info("Step 4: Merging bias and intensity dataframes...")
        final_df = pd.concat([microframe_bias_df, microframe_intensity_df], axis=1)
        logger.info("     DEBUG: final_df shape: " + str(final_df.shape))

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

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.squeeze(0)
        filtered_embeddings = []
        filtered_words = []
        word_ids = inputs.word_ids()

        for w_idx in set(word_ids):
            if w_idx is None:
                continue

            word_tokens_range = inputs.word_to_tokens(w_idx)

            if word_tokens_range is None:
                continue

            start, end = word_tokens_range
            word = self.tokenizer.decode(inputs.input_ids[0][start:end])
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

            word_embeddings = embeddings[start:end]
            filtered_embeddings.append(word_embeddings.mean(dim=0))
            filtered_words.append(normalized_word)

        filtered_embeddings_tensor = (
            torch.stack(filtered_embeddings)
            if filtered_embeddings
            else torch.tensor([])
        )

        if filtered_embeddings_tensor.numel() == 0:
            logger.info(
                f"No embeddings found for input text: {text}, after filtering: {filtered_words}"
            )

        return filtered_words, filtered_embeddings_tensor

    def get_embeddings_for_words(self, sentence, words, use_lemmatization=False):
        sentence_words, word_embeddings = self.get_embeddings_for_text(
            sentence, remove_stopwords=False, remove_non_words=False
        )

        if use_lemmatization:
            sentence_words = [
                self.lemmatizer.lemmatize(word) for word in sentence_words
            ]
            words = [self.lemmatizer.lemmatize(word) for word in words]

        embeddings = {}

        for word in words:
            if word in sentence_words:
                word_idx = sentence_words.index(word)
                embedding = word_embeddings[word_idx]
                embeddings[word] = embedding

        return embeddings

    def get_frameaxis_data(self):
        if not self.force_recalculate and (
            not self.save_path or not os.path.exists(self.save_path)
        ):
            self.force_recalculate = True

        if self.force_recalculate:
            logger.info("Calculating FrameAxis Embeddings")
            antonym_pairs_embeddings = self._load_or_calculate_antonym_embeddings()

            frameaxis_df = self.calculate_all_metrics(self.df, antonym_pairs_embeddings)

            if self.save_path:
                self._save_to_file(frameaxis_df, self.save_path, self.save_type)

            return frameaxis_df
        else:
            logger.info("Loading FrameAxis Embeddings")
            return self._load_from_file(self.save_path, self.save_type)
