import json
import pandas as pd
from preprocessing.datasets.article_dataset import ArticleDataset, custom_collate_fn
from preprocessing.frameaxis_processor import FrameAxisProcessor
from preprocessing.srl_processor import SRLProcessor
import torch
from torch.utils.data import DataLoader
import pickle
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize
import re
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from utils.logging_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


def preprocess_text(text):
    text = text.replace("\n\n", ". ")
    text = text.replace(".. ", ". ")
    text = text.replace("  ", " ")
    text = text.strip()
    text = re.sub(r"^IMM-\d+. PRIMARY. ", "", text)
    text = text.strip()
    return text


def expand_row(row):
    sentences = sent_tokenize(row["text"])
    return pd.DataFrame(
        {
            "article_id": [row["article_id"]] * len(sentences),
            "text": sentences,
            **{
                col: [row[col]] * len(sentences)
                for col in row.index
                if col not in ["article_id", "text"]
            },
        }
    )


def split_sentences_in_df(df):
    list_of_dataframes = df.progress_apply(expand_row, axis=1)
    new_df = pd.concat(list_of_dataframes.tolist(), ignore_index=True)
    return new_df


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
        bert_model_name="bert-base-uncased",
        name_tokenizer="bert-base-uncased",
        path_name_bert_model="bert-base-uncased",
        # frameaxis
        path_antonym_pairs="frameaxis/axes/custom.tsv",
        dim_names=["positive", "negative"],
        class_column_names=[
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
        ],
        num_workers=1,
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
            bert_model_name: The name of the BERT model.
            name_tokenizer: The name of the tokenizer.
            path_name_bert_model: The path to the BERT model.
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
        self.bert_model_name = bert_model_name
        self.name_tokenizer = name_tokenizer
        self.path_name_bert_model = path_name_bert_model

        # frameaxis
        self.path_antonym_pairs = path_antonym_pairs
        self.dim_names = dim_names

        self.class_column_names = class_column_names

        self.num_workers = num_workers

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
            "frameaxis_microframe": "data/frameaxis/mfc/frameaxis_microframes.pkl",
        },
        force_recalculate={
            "srl": False,
            "frameaxis": False,
        },
        train_mode=False,
        device=-1,
    ):
        """
        Processes the data by preparing the SRL and FrameAxis components.
        """
        df = df.reset_index(drop=True)

        srl_processor = SRLProcessor(
            df,
            dataframe_path=dataframe_path.get("srl", None),
            force_recalculate=force_recalculate.get("srl", False),
            device=device,
        )
        srl_df = srl_processor.get_srl_embeddings()

        srl_df = srl_df.reset_index(drop=True)

        frameaxis_processor = FrameAxisProcessor(
            df=df,
            path_keywords=self.path_antonym_pairs,
            save_file_name=dataframe_path.get("frameaxis", None),
            path_microframes=dataframe_path.get("frameaxis_microframe", None),
            bert_model_name=self.bert_model_name,
            name_tokenizer=self.name_tokenizer,
            path_name_bert_model=self.path_name_bert_model,
            force_recalculate=force_recalculate.get("frameaxis", False),
            save_type="pickle",
            dim_names=self.dim_names,
            word_blacklist=[],
        )
        frameaxis_df = frameaxis_processor.get_frameaxis_data()

        frameaxis_df = frameaxis_df.reset_index(drop=True)

        # Aggregating 'text' column in df into a list of strings for each article_id
        X_subset = df.groupby("article_id")["text"].apply(list).reset_index(name="text")

        # Assuming X_srl follows the same index order as df
        X_srl_subset = (
            srl_df.groupby(df["article_id"])["srls"]
            .apply(lambda x: x.values.tolist())
            .reset_index(name="srl_values")
        )

        # Aggregate frameaxis columns into a list of lists for each row
        frameaxis_cols = frameaxis_df.columns.tolist()

        if "article_id" in frameaxis_cols:
            frameaxis_cols.remove("article_id")

        if "text" in frameaxis_cols:
            frameaxis_cols.remove("text")

        frameaxis_df["frameaxis_values"] = frameaxis_df[frameaxis_cols].apply(
            list, axis=1
        )

        frameaxis_df = frameaxis_df[["article_id", "frameaxis_values"]]
        # Assuming frameaxis_df follows the same index order as df
        frameaxis_df_subset = (
            frameaxis_df.groupby(frameaxis_df["article_id"])["frameaxis_values"]
            .apply(lambda x: x.values.tolist())
            .reset_index(name="frameaxis_values")
        )

        if not train_mode:
            return X_subset, X_srl_subset, frameaxis_df_subset
        else:
            # Creating y_subset
            y_subset = (
                df.groupby("article_id")[self.class_column_names]
                .apply(lambda x: x.values.tolist())
                .reset_index(name="encoded_values")
            )

            return X_subset, X_srl_subset, frameaxis_df_subset, y_subset

    def get_dataset(
        self,
        path,
        format,
        dataframe_path={
            "srl": "data/srls/mfc/srls.pkl",
            "frameaxis": "data/frameaxis/mfc/frameaxis_frames.pkl",
            "frameaxis_microframe": "data/frameaxis/mfc/frameaxis_microframes.pkl",
        },
        force_recalculate={
            "srl": False,
            "frameaxis": False,
        },
        train_mode=True,
        device=-1,
    ):
        df = self._load_data(path=path, format=format)

        output = self._preprocess(
            df, dataframe_path, force_recalculate, train_mode=train_mode, device=device
        )

        if train_mode:
            X, X_srl, X_frameaxis, y = output

            X = X["text"]
            X_srl = X_srl["srl_values"]
            X_frameaxis = X_frameaxis["frameaxis_values"]

            dataset = ArticleDataset(
                X,
                X_srl,
                X_frameaxis,
                self.tokenizer,
                y,
                max_sentences_per_article=self.max_sentences_per_article,
                max_sentence_length=self.max_sentence_length,
                max_args_per_sentence=self.max_args_per_sentence,
                max_arg_length=self.max_arg_length,
                frameaxis_dim=self.frameaxis_dim,
                train_mode=train_mode,
            )

            return dataset
        else:
            X, X_srl, X_frameaxis = output

            X = X["text"]
            X_srl = X_srl["srl_values"]
            X_frameaxis = X_frameaxis["frameaxis_values"]

            dataset = ArticleDataset(
                X,
                X_srl,
                X_frameaxis,
                self.tokenizer,
                None,
                max_sentences_per_article=self.max_sentences_per_article,
                max_sentence_length=self.max_sentence_length,
                max_args_per_sentence=self.max_args_per_sentence,
                max_arg_length=self.max_arg_length,
                frameaxis_dim=self.frameaxis_dim,
                train_mode=train_mode,
            )

            return dataset

    def get_datasets(
        self,
        path,
        format,
        dataframe_path={
            "srl": "data/srls/mfc/srls.pkl",
            "frameaxis": "data/frameaxis/mfc/frameaxis_frames.pkl",
            "frameaxis_microframe": "data/frameaxis/mfc/frameaxis_microframes.pkl",
        },
        force_recalculate={
            "srl": False,
            "frameaxis": False,
        },
        train_mode=True,
        random_state=None,
        stratification=None,  # None, "single", "multi"
        device=-1,
    ):
        """
        Returns the train and test datasets.
        """
        df = self._load_data(path=path, format=format)

        output = self._preprocess(
            df, dataframe_path, force_recalculate, train_mode=train_mode, device=device
        )

        if train_mode:
            X, X_srl, X_frameaxis, y = output

            merged_df = (
                X.merge(X_srl, on="article_id")
                .merge(X_frameaxis, on="article_id")
                .merge(y, on="article_id")
            )
        else:
            X, X_srl, X_frameaxis = output

            merged_df = X.merge(X_srl, on="article_id").merge(
                X_frameaxis, on="article_id"
            )

        if train_mode:
            if stratification == "single":
                y_stratify = merged_df["encoded_values"].apply(lambda x: x[0].index(1))

                train_df, test_df = train_test_split(
                    merged_df,
                    test_size=self.test_size,
                    random_state=random_state,
                    stratify=y_stratify,
                    shuffle=True,
                )
            elif stratification == "multi":
                # Extract the multi-label data
                y_multi = (
                    merged_df.groupby("article_id")["encoded_values"].first().tolist()
                )

                # Function to flatten the complex structure
                def flatten_labels(labels):
                    flat = []
                    for outer_list in labels:
                        flat_inner = []
                        for inner_list in outer_list:
                            flat_inner.extend(inner_list)
                        flat.append(flat_inner)
                    return flat

                # Flatten the labels
                y_multi_flat = flatten_labels(y_multi)

                # Find the maximum length after flattening
                max_len = max(len(item) for item in y_multi_flat)

                # Pad the flattened lists
                y_multi_padded = [
                    item + [0] * (max_len - len(item)) for item in y_multi_flat
                ]

                # Convert to numpy array
                y_multi_array = np.array(y_multi_padded)

                # Initialize the stratified k-fold splitter
                mskf = MultilabelStratifiedKFold(
                    n_splits=int(1 / self.test_size),
                    random_state=random_state,
                    shuffle=True,
                )

                # Perform the split
                X = merged_df.index.to_numpy().reshape(
                    -1, 1
                )  # We need a 2D array for X

                for train_index, test_index in mskf.split(X, y_multi_array):
                    train_df = merged_df.iloc[train_index]
                    test_df = merged_df.iloc[test_index]
                    break
            else:
                # Original split if stratification is set to None
                train_df, test_df = train_test_split(
                    merged_df, test_size=self.test_size, random_state=random_state
                )

            # Reset indices for train and test DataFrames
            train_df = train_df.reset_index(drop=True)
            test_df = test_df.reset_index(drop=True)

            # Remove 'class_index' column if it was added
            if "class_index" in train_df.columns:
                train_df = train_df.drop("class_index", axis=1)
                test_df = test_df.drop("class_index", axis=1)

            # Extract relevant columns for training and testing
            X_train = train_df["text"]
            X_srl_train = train_df["srl_values"]
            X_frameaxis_train = train_df["frameaxis_values"]

            y_train = train_df["encoded_values"]

            X_test = test_df["text"]
            X_srl_test = test_df["srl_values"]
            X_frameaxis_test = test_df["frameaxis_values"]

            y_test = test_df["encoded_values"]

            # Assert lengths
            assert len(X_train) == len(
                y_train
            ), f"Length mismatch: X_train({len(X_train)}) and y_train({len(y_train)})"
            assert len(X_test) == len(
                y_test
            ), f"Length mismatch: X_test({len(X_test)}) and y_test({len(y_test)})"
            assert len(X_srl_train) == len(
                y_train
            ), f"Length mismatch: X_srl_train({len(X_srl_train)}) and y_train({len(y_train)})"
            assert len(X_srl_test) == len(
                y_test
            ), f"Length mismatch: X_srl_test({len(X_srl_test)}) and y_test({len(y_test)})"
            assert len(X_frameaxis_train) == len(
                y_train
            ), f"Length mismatch: X_frameaxis_train({len(X_frameaxis_train)}) and y_train({len(y_train)})"
            assert len(X_frameaxis_test) == len(
                y_test
            ), f"Length mismatch: X_frameaxis_test({len(X_frameaxis_test)}) and y_test({len(y_test)})"

            # Ensure the length is the same between the three datasets
            assert len(X_train) == len(
                X_srl_train
            ), f"Length mismatch: X_train({len(X_train)}) and X_srl_train({len(X_srl_train)})"
            assert len(X_train) == len(
                X_frameaxis_train
            ), f"Length mismatch: X_train({len(X_train)}) and X_frameaxis_train({len(X_frameaxis_train)})"
            assert len(X_test) == len(
                X_srl_test
            ), f"Length mismatch: X_test({len(X_test)}) and X_srl_test({len(X_srl_test)})"
            assert len(X_test) == len(
                X_frameaxis_test
            ), f"Length mismatch: X_test({len(X_test)}) and X_frameaxis_test({len(X_frameaxis_test)})"

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
                train_mode=train_mode,
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
                train_mode=train_mode,
            )

            return train_dataset, test_dataset, train_df, test_df
        else:
            X_text = merged_df["text"]

            X_srl = merged_df["srl_values"]
            X_frameaxis = merged_df["frameaxis_values"]

            dataset = ArticleDataset(
                X_text,
                X_srl,
                X_frameaxis,
                self.tokenizer,
                None,
                max_sentences_per_article=self.max_sentences_per_article,
                max_sentence_length=self.max_sentence_length,
                max_args_per_sentence=self.max_args_per_sentence,
                max_arg_length=self.max_arg_length,
                frameaxis_dim=self.frameaxis_dim,
                train_mode=train_mode,
            )

            return dataset, merged_df

    def get_dataframe(
        self,
        path,
        format,
        dataframe_path={
            "srl": "data/srls/mfc/srls.pkl",
            "frameaxis": "data/frameaxis/mfc/frameaxis_frames.pkl",
            "frameaxis_microframe": "data/frameaxis/mfc/frameaxis_microframes.pkl",
        },
        force_recalculate={
            "srl": False,
            "frameaxis": False,
        },
        train_mode=True,
        stratification=None,  # None, "single", "multi"
        random_state=None,
    ):
        if train_mode:
            _, _, train_df, test_df = self.get_datasets(
                path,
                format,
                dataframe_path,
                force_recalculate,
                train_mode=train_mode,
                random_state=random_state,
                stratification=stratification,
            )
            return train_df, test_df
        else:
            _, df = self.get_datasets(
                path,
                format,
                dataframe_path,
                force_recalculate,
                train_mode=train_mode,
                random_state=random_state,
            )
            return df

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
        sample_size=-1,
        device=-1,
    ):
        dataset = self.get_dataset(
            path,
            format,
            dataframe_path,
            force_recalculate,
            train_mode=True,
            device=device,
        )

        if sample_size > 0:
            logger.info(f"Sampling {sample_size} examples from the dataset.")
            dataset = torch.utils.data.Subset(dataset, range(sample_size))
            logger.info(f"Dataset size: {len(dataset)}")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return dataset, dataloader

    def get_dataloaders(
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
        sample_size=-1,
        random_state=None,
        stratification=None,  # None, "single", "multi"
    ):
        train_dataset, test_dataset, _, _ = self.get_datasets(
            path,
            format,
            dataframe_path,
            force_recalculate,
            train_mode=True,
            random_state=random_state,
            stratification=stratification,
        )

        if sample_size > 0:
            logger.info(f"Sampling {sample_size} examples from the dataset.")
            train_dataset = torch.utils.data.Subset(train_dataset, range(sample_size))
            test_dataset = torch.utils.data.Subset(test_dataset, range(sample_size))
            logger.info(f"Train dataset size: {len(train_dataset)}")
            logger.info(f"Test dataset size: {len(test_dataset)}")

        # create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            drop_last=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return train_dataset, test_dataset, train_dataloader, test_dataloader

    def preprocess_single_article(
        self,
        text,
        frameaxis_microframe_path="../../data/frameaxis/mfc/frameaxis_mft_microframes.pkl",
        frameaxis_word_blacklist=[],
        cuda=False,
    ):
        text = preprocess_text(text)
        sentences = sent_tokenize(text)
        df = pd.DataFrame({"article_id": [0] * len(sentences), "text": sentences})

        srl_processor = SRLProcessor(
            df,
            dataframe_path=None,
            force_recalculate=True,
            device=-1 if not cuda else 0,
        )
        srl_df = srl_processor.get_srl_embeddings()
        srl_df = srl_df.reset_index(drop=True)

        frameaxis_processor = FrameAxisProcessor(
            df=df,
            path_antonym_pairs=self.path_antonym_pairs,
            save_path=None,
            path_microframes=frameaxis_microframe_path,
            bert_model_name=self.bert_model_name,
            name_tokenizer=self.name_tokenizer,
            path_name_bert_model=self.path_name_bert_model,
            force_recalculate=True,
            save_type="pickle",
            dim_names=self.dim_names,
            word_blacklist=frameaxis_word_blacklist,
        )
        frameaxis_df = frameaxis_processor.get_frameaxis_data()
        frameaxis_df = frameaxis_df.reset_index(drop=True)

        X_srl_subset = (
            srl_df.groupby(df["article_id"])["srls"]
            .apply(lambda x: x.values.tolist())
            .reset_index(name="srl_values")
        )

        frameaxis_cols = frameaxis_df.columns.tolist()
        if "article_id" in frameaxis_cols:
            frameaxis_cols.remove("article_id")
        if "text" in frameaxis_cols:
            frameaxis_cols.remove("text")
        frameaxis_df["frameaxis_values"] = frameaxis_df[frameaxis_cols].apply(
            list, axis=1
        )
        frameaxis_df = frameaxis_df[["article_id", "frameaxis_values"]]

        frameaxis_df_subset = (
            frameaxis_df.groupby(frameaxis_df["article_id"])["frameaxis_values"]
            .apply(lambda x: x.values.tolist())
            .reset_index(name="frameaxis_values")
        )

        X_subset = df.groupby("article_id")["text"].apply(list).reset_index(name="text")

        assert len(X_subset) == len(X_srl_subset)
        assert len(X_subset) == len(frameaxis_df_subset)

        dataset = ArticleDataset(
            X_subset["text"],
            X_srl_subset["srl_values"],
            frameaxis_df_subset["frameaxis_values"],
            self.tokenizer,
            [None] * len(X_subset),
            max_sentences_per_article=self.max_sentences_per_article,
            max_sentence_length=self.max_sentence_length,
            max_args_per_sentence=self.max_args_per_sentence,
            max_arg_length=self.max_arg_length,
            frameaxis_dim=self.frameaxis_dim,
            train_mode=False,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=custom_collate_fn,
            drop_last=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return dataset, dataloader
