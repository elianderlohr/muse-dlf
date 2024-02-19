import json
import os
import pandas as pd
from preprocessing.datasets.article_dataset import ArticleDataset, custom_collate_fn
from preprocessing.frameaxis_processor import FrameAxisProcessor
from preprocessing.srl_processor import SRLProcessor
import torch
from torch.utils.data import DataLoader
from allennlp.predictors.predictor import Predictor
from tqdm.notebook import tqdm
import pickle
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize


class PreProcessor:
    def __init__(
        self,
        tokenizer,
        batch_size=16,
        max_sentences_per_article=32,
        max_sentence_length=32,
        max_args_per_sentence=10,
        max_arg_length=16,
        test_size=0.1,
        frameaxis_dim=20,
    ):
        """
        Initializes the PreProcessor.

        Args:
            tokenizer: Tokenizer instance from transformers library.
            data_source (str): Path to the data file or directory containing JSON files.
            tokenizer: Tokenizer instance from transformers library.
            data_format (str): Format of the data source ('json', 'csv', 'pickle').
            force_recalculate (bool): Whether to force recalculation of SRL and FrameAxis.
            srl_model_path (str): Path or URL to the SRL model.
            batch_size, max_sentences_per_article, max_sentence_length,
            max_args_per_sentence, max_arg_length, test_size, frameaxis_dim: Parameters for dataset and dataloader preparation.
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_sentences_per_article = max_sentences_per_article
        self.max_sentence_length = max_sentence_length
        self.max_args_per_sentence = max_args_per_sentence
        self.max_arg_length = max_arg_length
        self.test_size = test_size
        self.frameaxis_dim = frameaxis_dim

    def _load_data(self):
        """
        Loads data from the specified source and format.
        """
        if self.data_format == "json":
            with open(self.data_source) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif self.data_format == "csv":
            df = pd.read_csv(self.data_source)
        elif self.data_format == "pickle":
            with open(self.data_source, "rb") as f:
                df = pickle.load(f)
        else:
            raise ValueError("Unsupported data format specified.")
        return df

    def _preprocess(
        self,
        df,
        model_name="bert-base-uncased",
        dataframe_path={
            "srl": "../notebooks/classifier/X_srl_filtered.pkl",
            "frameaxis": "../notebooks/classifier/X_frameaxis_filtered.pkl",
        },
        force_recalculate={"srl": False, "frameaxis": False},
    ):
        """
        Processes the data by preparing the SRL and FrameAxis components.
        """
        df = df.reset_index(drop=True)

        srl_processor = SRLProcessor(
            df["text"],
            dataframe_path=dataframe_path.get("srl", None),
            force_recalculate=force_recalculate.get("srl", False),
        )
        srl_df = srl_processor.get_srl_embeddings()

        srl_df = srl_df.reset_index(drop=True)

        frameaxis_processor = FrameAxisProcessor(
            df,
            dataframe_path=dataframe_path.get("frameaxis", None),
            force_recalculate=force_recalculate.get("frameaxis", False),
            model_name=model_name,
        )
        frameaxis_df = frameaxis_processor.get_frameaxis_data()

        frameaxis_df = frameaxis_df.reset_index(drop=True)

        y_cols = [
            "Capacity and Resources",
            "Crime and Punishment",
            "Cultural Identity",
            "Economic",
            "External Regulation and Reputation",
            "Fairness and Equality",
            "Health and Safety",
            "Legality, Constitutionality, Jurisdiction",
            "Morality",
            "Other",
            "Policy Prescription and Evaluation",
            "Political",
            "Public Sentiment",
            "Quality of Life",
            "Security and Defense",
        ]

        # Creating y_subset
        y_subset = (
            df.groupby("article_id")[y_cols]
            .apply(lambda x: x.values.tolist())
            .reset_index(name="encoded_values")
        )
        y_subset = y_subset["encoded_values"]

        # Aggregating 'text' column in df into a list of strings for each article_id
        X_subset = df.groupby("article_id")["text"].apply(list).reset_index(name="text")
        X_subset = X_subset["text"]

        # Assuming X_srl follows the same index order as df
        X_srl_subset = (
            srl_df.groupby(df["article_id"])
            .apply(lambda x: x.values.tolist())
            .reset_index(name="srl_values")
        )
        X_srl_subset = X_srl_subset["srl_values"]

        # aggregate frameaxis columns into a list of lists for row
        frameaxis_cols = frameaxis_df.columns.tolist()
        frameaxis_cols.remove("article_id")
        frameaxis_df["frameaxis_values"] = frameaxis_df[frameaxis_cols].apply(
            list, axis=1
        )

        frameaxis_df = frameaxis_df[["article_id", "frameaxis_values"]]

        # Assuming frameaxis_df follows the same index order as df
        frameaxis_df_subset = (
            frameaxis_df.groupby(df["article_id"])["frameaxis_values"]
            .apply(lambda x: x.values.tolist())
            .reset_index(name="frameaxis_values")
        )
        frameaxis_df_subset = frameaxis_df_subset["frameaxis_values"]

        return X_subset, X_srl_subset, frameaxis_df_subset, y_subset

    def get_dataloader(self):
        """
        Returns the train and test datasets.
        """

        df = self._load_data()

        X, X_srl, X_frameaxis, y = self._preprocess(df)

        # Splitting the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        X_srl_train, X_srl_test, _, _ = train_test_split(
            X_srl, y, test_size=self.test_size, random_state=42
        )

        X_frameaxis_train, X_frameaxis_test, _, _ = train_test_split(
            X_frameaxis, y, test_size=self.test_size, random_state=42
        )

        train_dataset = ArticleDataset(
            X_train,
            X_srl_train,
            X_frameaxis_train,
            self.tokenizer,
            y_train,
            max_sentences_per_article=self.max_sentences_per_article,
            max_sentence_length=self.max_sentence_length,
            max_args_per_sentence=self.max_args_per_sentence,
            max_arg_length=self.max_arg_length,
            frameaxis_dim=self.frameaxis_dim,
        )

        test_dataset = ArticleDataset(
            X_test,
            X_srl_test,
            X_frameaxis_test,
            self.tokenizer,
            y_test,
            max_sentences_per_article=self.max_sentences_per_article,
            max_sentence_length=self.max_sentence_length,
            max_args_per_sentence=self.max_args_per_sentence,
            max_arg_length=self.max_arg_length,
            frameaxis_dim=self.frameaxis_dim,
        )

        # create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            drop_last=True,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            drop_last=True,
        )

        return train_dataset, test_dataset, train_dataloader, test_dataloader
