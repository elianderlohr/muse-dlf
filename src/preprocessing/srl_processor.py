from collections import defaultdict
import os
import pandas as pd
import pickle
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

    def get_srl_embeddings(self, batch_size=32):
        """
        Main method to process the SRL components, either by loading them from a pickle file or by recalculating.
        """
        if self.force_recalculate or not os.path.exists(self.dataframe_path):
            return self._recalculate_srl(batch_size=batch_size)
        else:
            return self._load_srl()

    def _recalculate_srl(self, batch_size=32):
        """
        Recalculates the SRL components for the sentences in the DataFrame and returns a DataFrame
        with columns for article_id, text, and srls, where srls is a list of SRL components for each text entry.
        """
        logger.info("Recalculating SRL components...")

        from allennlp.predictors.predictor import Predictor

        print(f"Loading SRL predictor using device {self.device}...")

        predictor = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
            cuda_device=self.device,
        )

        # Convert the processed data into a DataFrame
        result_df = self._batch_process_srl_with_ids(self.df, predictor, batch_size)

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

    def _extract_srl_batch(self, batch_records, predictor):
        """
        Extracts SRL components for a batch of records.
        """
        batched_sentences = [{"sentence": record["text"]} for record in batch_records]
        batched_srl = predictor.predict_batch_json(batched_sentences)

        results = []
        for record, srl in zip(batch_records, batched_srl):
            sentence_results = []

            for verb_entry in srl["verbs"]:
                arg_components = {"ARG0": [], "ARG1": []}
                for i, tag in enumerate(verb_entry["tags"]):
                    if "ARG0" in tag:
                        arg_components["ARG0"].append(srl["words"][i])
                    elif "ARG1" in tag:
                        arg_components["ARG1"].append(srl["words"][i])

                if len(arg_components["ARG0"]) > 0 or len(arg_components["ARG1"]) > 0:
                    sentence_results.append(
                        {
                            "predicate": verb_entry["verb"],
                            "ARG0": " ".join(arg_components["ARG0"]),
                            "ARG1": " ".join(arg_components["ARG1"]),
                        }
                    )

            if len(sentence_results) == 0:
                sentence_results.append(
                    {
                        "predicate": "",
                        "ARG0": "",
                        "ARG1": "",
                    }
                )

            results.append(
                {
                    "article_id": record["article_id"],
                    "text": record["text"],
                    "srls": sentence_results,
                }
            )
        return results

    def _batch_process_srl_with_ids(self, df, predictor, batch_size=32):
        """
        Directly process text entries in batches to get SRLs and associate them with article_id and text.
        """
        srl_data = []
        # Use a range with step=batch_size to iterate over the DataFrame in batches
        for i in tqdm(
            range(0, len(df), batch_size),
            desc="Processing SRL Batches",
            total=len(df) // batch_size + (len(df) % batch_size > 0),
        ):
            # Slice the DataFrame to get the current batch
            batch_df = df.iloc[i : i + batch_size]

            batched_dict = batch_df.to_dict(orient="records")

            # Extract SRLs for the batch of records
            batch_results = self._extract_srl_batch(batched_dict, predictor)
            # Add the batch results to the main list
            srl_data.extend(batch_results)
        return pd.DataFrame(srl_data)
