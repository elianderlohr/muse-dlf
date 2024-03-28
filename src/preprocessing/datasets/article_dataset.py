import torch
from torch.utils.data import Dataset
import pandas as pd


class ArticleDataset(Dataset):
    def __init__(
        self,
        X,
        X_srl,
        X_frameaxis,
        tokenizer,
        labels=None,
        max_sentences_per_article=32,
        max_sentence_length=32,
        max_args_per_sentence=10,
        max_arg_length=16,
        frameaxis_dim=20,
    ):
        self.X = X
        self.X_srl = X_srl
        self.X_frameaxis = X_frameaxis
        self.labels = labels

        self.tokenizer = tokenizer
        self.max_sentences_per_article = max_sentences_per_article
        self.max_sentence_length = max_sentence_length
        self.max_args_per_sentence = max_args_per_sentence
        self.max_arg_length = max_arg_length

        self.frameaxis_dim = frameaxis_dim

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sentences = self.X.iloc[idx]
        srl_data = self.X_srl.iloc[idx]
        frameaxis_data = self.X_frameaxis.iloc[idx]

        # labels
        labels = self.labels.iloc[idx]

        # Tokenize sentences and get attention masks
        sentence_ids, sentence_attention_masks = [], []
        for sentence in sentences:
            encoded = self.tokenizer(
                sentence,
                add_special_tokens=True,
                max_length=self.max_sentence_length,
                truncation=True,
                padding="max_length",
                return_attention_mask=True,
            )
            sentence_ids.append(encoded["input_ids"])
            sentence_attention_masks.append(encoded["attention_mask"])

        # Padding for sentences if necessary
        while len(sentence_ids) < self.max_sentences_per_article:
            sentence_ids.append([0] * self.max_sentence_length)
            sentence_attention_masks.append([0] * self.max_sentence_length)

        sentence_ids = sentence_ids[: self.max_sentences_per_article]
        sentence_attention_masks = sentence_attention_masks[
            : self.max_sentences_per_article
        ]

        # frameaxis
        while len(frameaxis_data) < self.max_sentences_per_article:
            frameaxis_data.append([0] * self.frameaxis_dim)

        frameaxis_data = frameaxis_data[: self.max_sentences_per_article]

        # replace nan values in frameaxis with 0
        frameaxis_data = pd.DataFrame(frameaxis_data).fillna(0).values.tolist()

        # Process SRL data
        predicates, arg0s, arg1s = [], [], []
        predicate_attention_masks, arg0_attention_masks, arg1_attention_masks = (
            [],
            [],
            [],
        )
        for srl_items in srl_data:
            sentence_predicates, sentence_arg0s, sentence_arg1s = [], [], []
            sentence_predicate_masks, sentence_arg0_masks, sentence_arg1_masks = (
                [],
                [],
                [],
            )

            if not isinstance(srl_items, list):
                srl_items = [srl_items]

            for item in srl_items:
                encoded_predicate = self.tokenizer(
                    item["predicate"],
                    add_special_tokens=True,
                    max_length=self.max_arg_length,
                    truncation=True,
                    padding="max_length",
                    return_attention_mask=True,
                )
                encoded_arg0 = self.tokenizer(
                    item["ARG0"],
                    add_special_tokens=True,
                    max_length=self.max_arg_length,
                    truncation=True,
                    padding="max_length",
                    return_attention_mask=True,
                )
                encoded_arg1 = self.tokenizer(
                    item["ARG1"],
                    add_special_tokens=True,
                    max_length=self.max_arg_length,
                    truncation=True,
                    padding="max_length",
                    return_attention_mask=True,
                )

                sentence_predicates.append(encoded_predicate["input_ids"])
                sentence_arg0s.append(encoded_arg0["input_ids"])
                sentence_arg1s.append(encoded_arg1["input_ids"])

                sentence_predicate_masks.append(encoded_predicate["attention_mask"])
                sentence_arg0_masks.append(encoded_arg0["attention_mask"])
                sentence_arg1_masks.append(encoded_arg1["attention_mask"])

            # Padding for SRL elements
            for _ in range(self.max_args_per_sentence):
                sentence_predicates.append([0] * self.max_arg_length)
                sentence_arg0s.append([0] * self.max_arg_length)
                sentence_arg1s.append([0] * self.max_arg_length)

                sentence_predicate_masks.append([0] * self.max_arg_length)
                sentence_arg0_masks.append([0] * self.max_arg_length)
                sentence_arg1_masks.append([0] * self.max_arg_length)

            sentence_predicates = sentence_predicates[: self.max_args_per_sentence]
            sentence_arg0s = sentence_arg0s[: self.max_args_per_sentence]
            sentence_arg1s = sentence_arg1s[: self.max_args_per_sentence]

            sentence_predicate_masks = sentence_predicate_masks[
                : self.max_args_per_sentence
            ]
            sentence_arg0_masks = sentence_arg0_masks[: self.max_args_per_sentence]
            sentence_arg1_masks = sentence_arg1_masks[: self.max_args_per_sentence]

            predicates.append(sentence_predicates)
            arg0s.append(sentence_arg0s)
            arg1s.append(sentence_arg1s)

            predicate_attention_masks.append(sentence_predicate_masks)
            arg0_attention_masks.append(sentence_arg0_masks)
            arg1_attention_masks.append(sentence_arg1_masks)

        # Padding for SRL data
        srl_padding = [[0] * self.max_arg_length] * self.max_args_per_sentence
        mask_padding = [[0] * self.max_arg_length] * self.max_args_per_sentence

        predicates = (predicates + [srl_padding] * self.max_sentences_per_article)[
            : self.max_sentences_per_article
        ]
        arg0s = (arg0s + [srl_padding] * self.max_sentences_per_article)[
            : self.max_sentences_per_article
        ]
        arg1s = (arg1s + [srl_padding] * self.max_sentences_per_article)[
            : self.max_sentences_per_article
        ]

        predicate_attention_masks = (
            predicate_attention_masks + [mask_padding] * self.max_sentences_per_article
        )[: self.max_sentences_per_article]
        arg0_attention_masks = (
            arg0_attention_masks + [mask_padding] * self.max_sentences_per_article
        )[: self.max_sentences_per_article]
        arg1_attention_masks = (
            arg1_attention_masks + [mask_padding] * self.max_sentences_per_article
        )[: self.max_sentences_per_article]

        data = {
            "sentence_ids": torch.tensor(sentence_ids, dtype=torch.long),
            "sentence_attention_masks": torch.tensor(
                sentence_attention_masks, dtype=torch.long
            ),
            "predicate_ids": torch.tensor(predicates, dtype=torch.long),
            "predicate_attention_masks": torch.tensor(
                predicate_attention_masks, dtype=torch.long
            ),
            "arg0_ids": torch.tensor(arg0s, dtype=torch.long),
            "arg0_attention_masks": torch.tensor(
                arg0_attention_masks, dtype=torch.long
            ),
            "arg1_ids": torch.tensor(arg1s, dtype=torch.long),
            "arg1_attention_masks": torch.tensor(
                arg1_attention_masks, dtype=torch.long
            ),
            "frameaxis": torch.tensor(frameaxis_data, dtype=torch.float),
            "labels": torch.tensor(labels[0], dtype=torch.long),
        }

        return data


def custom_collate_fn(batch):
    # Extract individual lists from the batch
    sentence_ids = [item["sentence_ids"] for item in batch]
    sentence_attention_masks = [item["sentence_attention_masks"] for item in batch]
    predicate_ids = [item["predicate_ids"] for item in batch]
    predicate_attention_masks = [item["predicate_attention_masks"] for item in batch]
    arg0_ids = [item["arg0_ids"] for item in batch]
    arg0_attention_masks = [item["arg0_attention_masks"] for item in batch]
    arg1_ids = [item["arg1_ids"] for item in batch]
    arg1_attention_masks = [item["arg1_attention_masks"] for item in batch]
    frameaxis = [item["frameaxis"] for item in batch]
    labels = [item["labels"] for item in batch]

    # Pad each list
    sentence_ids = torch.nn.utils.rnn.pad_sequence(
        sentence_ids, batch_first=True, padding_value=0
    )
    sentence_attention_masks = torch.nn.utils.rnn.pad_sequence(
        sentence_attention_masks, batch_first=True, padding_value=0
    )
    predicate_ids = torch.nn.utils.rnn.pad_sequence(
        predicate_ids, batch_first=True, padding_value=0
    )
    predicate_attention_masks = torch.nn.utils.rnn.pad_sequence(
        predicate_attention_masks, batch_first=True, padding_value=0
    )
    arg0_ids = torch.nn.utils.rnn.pad_sequence(
        arg0_ids, batch_first=True, padding_value=0
    )
    arg0_attention_masks = torch.nn.utils.rnn.pad_sequence(
        arg0_attention_masks, batch_first=True, padding_value=0
    )
    arg1_ids = torch.nn.utils.rnn.pad_sequence(
        arg1_ids, batch_first=True, padding_value=0
    )
    arg1_attention_masks = torch.nn.utils.rnn.pad_sequence(
        arg1_attention_masks, batch_first=True, padding_value=0
    )
    frameaxis = torch.nn.utils.rnn.pad_sequence(
        frameaxis, batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)

    # Create the output dictionary
    output_dict = {
        "sentence_ids": sentence_ids,
        "sentence_attention_masks": sentence_attention_masks,
        "predicate_ids": predicate_ids,
        "predicate_attention_masks": predicate_attention_masks,
        "arg0_ids": arg0_ids,
        "arg0_attention_masks": arg0_attention_masks,
        "arg1_ids": arg1_ids,
        "arg1_attention_masks": arg1_attention_masks,
        "frameaxis": frameaxis,
        "labels": labels,
    }

    return output_dict
