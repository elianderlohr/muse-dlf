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
        model_name="bert-base-uncased",
        path_antonym_pairs="frameaxis/axes/custom.tsv",
    ):
        """
        Initializes the PreProcessor.

        Args:
            tokenizer: The tokenizer to be used for tokenizing the input.
            batch_size: The batch size for the DataLoader.
            max_sentences_per_article: The maximum number of sentences in the input.
            max_sentence_length: The maximum length of a sentence.
            max_args_per_sentence: The maximum number of arguments per sentence.
            max_arg_length: The maximum length of an argument.
            test_size: The size of the test set.
            frameaxis_dim: The dimension of the FrameAxis embeddings.
            model_name: The name of the BERT model to be used.
            path_antonym_pairs: The path to the antonym pairs file.
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_sentences_per_article = max_sentences_per_article
        self.max_sentence_length = max_sentence_length
        self.max_args_per_sentence = max_args_per_sentence
        self.max_arg_length = max_arg_length
        self.test_size = test_size
        self.frameaxis_dim = frameaxis_dim
        self.model_name = model_name
        self.path_antonym_pairs = path_antonym_pairs

    def _load_data(self, path, format):
        """
        Loads data from the specified source and format.
        """
        if format == "json":
            with open(path) as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif format == "csv":
            df = pd.read_csv(path)
        elif format == "pickle":
            with open(path, "rb") as f:
                df = pickle.load(f)
        else:
            raise ValueError("Unsupported data format specified.")
        return df

    def _preprocess(
        self,
        df,
        dataframe_path={
            "srl": "data/srls/mfc/srls.pkl",
            "frameaxis": "data/frameaxis/mfc/frameaxis_frames.pkl",
        },
        force_recalculate={
            "srl": False,
            "frameaxis": False,
        },
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
            model_name=self.model_name,
            path_antonym_pairs=self.path_antonym_pairs,
            save_type="pickle",
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

    def get_dataloader(
        self,
        path,
        format,
        dataframe_path={
            "srl": "data/srls/mfc/srls.pkl",
            "frameaxis": "data/frameaxis/mfc/frameaxis_frames.pkl",
        },
        force_recalculate={
            "srl": False,
            "frameaxis": False,
        },
    ):
        """
        Returns the train and test datasets.
        """

        df = self._load_data(path=path, format=format)

        X, X_srl, X_frameaxis, y = self._preprocess(
            df, dataframe_path, force_recalculate
        )

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
