from collections import defaultdict
import os
import pandas as pd
import pickle
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

from utils.logging_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class SRLProcessor:
    def __init__(
        self,
        df,
        dataframe_path="../notebooks/classifier/X_srl_filtered.pkl",
        force_recalculate=False,
        save_type="pickle",
        device=0,
    ):
        """
        Initializes the SRLProcessor with a DataFrame, a path to a pickle file, and a flag indicating whether to force recalculation.

        Args:
            df (pd.DataFrame): DataFrame with text data.
            dataframe_path (str): Path to save/load the SRL components DataFrame.
            force_recalculate (bool): If True, forces recalculation of SRL components.
        """
        self.df = df
        self.dataframe_path = dataframe_path
        self.force_recalculate = force_recalculate
        self.save_type = save_type
        self.device = device

        # allowed save types
        if self.save_type not in ["csv", "pickle", "json"]:
            raise ValueError(
                "Invalid save_type. Must be one of 'csv', 'pickle', or 'json'."
            )

    def get_srl_embeddings(self):
        """
        Main method to process the SRL components, either by loading them from a pickle file or by recalculating.
        """
        if self.force_recalculate or not os.path.exists(self.dataframe_path):
            return self._recalculate_srl()
        else:
            return self._load_srl()

    def _recalculate_srl(self):
        """
        Recalculates the SRL components for the sentences in the DataFrame and returns a DataFrame
        with columns for article_id, text, and srls, where srls is a list of SRL components for each text entry.
        """
        logger.info("Recalculating SRL components...")
        predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
            cuda_device=self.device,
        )

        # Directly process each text entry to get SRLs and associate with article_id and text
        srl_data = []
        for _, row in tqdm(
            self.df.iterrows(), desc="Processing SRL Batches", total=len(self.df)
        ):
            article_id, text = row["article_id"], row["text"]
            srls = self._extract_srl_batch(
                [text], predictor
            )  # Process a single text entry as a batch of size 1
            srl_data.append(
                {
                    "article_id": article_id,
                    "text": text,
                    "srls": srls[
                        0
                    ],  # Extract the first (and only) element as each text is processed individually
                }
            )

        # Convert the processed data into a DataFrame
        result_df = pd.DataFrame(srl_data)

        # Save the DataFrame if a path is specified
        if self.dataframe_path:
            if self.save_type == "csv":
                result_df.to_csv(self.dataframe_path, index=False)
            elif self.save_type == "json":
                result_df.to_json(self.dataframe_path)
            elif self.save_type == "pickle":
                with open(self.dataframe_path, "wb") as f:
                    pickle.dump(result_df, f)

        return result_df

    def _load_srl(self):
        """
        Loads the SRL components from a pickle file.
        """
        logger.info("Loading SRL components from pickle...")
        with open(self.dataframe_path, "rb") as f:
            srl_series = pickle.load(f)
        return srl_series

    def _extract_srl_batch(self, batched_sentences, predictor):
        """
        Extracts SRL components for a batch of sentences.
        """
        batched_sentences = [{"sentence": sentence} for sentence in batched_sentences]
        batched_srl = predictor.predict_batch_json(batched_sentences)

        results = []
        for srl in batched_srl:
            sentence_results = []

            for verb_entry in srl["verbs"]:
                arg_components = {"ARG0": [], "ARG1": []}
                for i, tag in enumerate(verb_entry["tags"]):
                    if "ARG0" in tag:
                        arg_components["ARG0"].append(srl["words"][i])
                    elif "ARG1" in tag:
                        arg_components["ARG1"].append(srl["words"][i])

                (
                    sentence_results.append(
                        {
                            "predicate": verb_entry["verb"],
                            "ARG0": " ".join(arg_components["ARG0"]),
                            "ARG1": " ".join(arg_components["ARG1"]),
                        }
                    )
                    if arg_components["ARG0"] or arg_components["ARG1"]
                    else {"predicate": "", "ARG0": "", "ARG1": ""}
                )
            results.append(
                sentence_results
                if sentence_results
                else [{"predicate": "", "ARG0": "", "ARG1": ""}]
            )
        return results

    def _batch_process_srl(self, texts, article_ids, predictor, batch_size=32):
        """
        Extracts SRL components for all sentences in a DataFrame in an optimized, batched manner.
        Now also includes article IDs to ensure SRL components are associated with the correct articles.
        """
        results_by_article = defaultdict(list)
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing SRL Batches"):
            batched_sentences = texts[i : i + batch_size].tolist()
            batch_article_ids = article_ids[i : i + batch_size].tolist()
            batch_results = self._extract_srl_batch(batched_sentences, predictor)
            for article_id, srls in zip(batch_article_ids, batch_results):
                results_by_article[article_id].extend(srls)
        return results_by_article
