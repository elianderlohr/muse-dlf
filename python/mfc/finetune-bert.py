# Imports
import argparse
import os
import json
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Downloading NLTK resources
nltk.download("all")

# Define paths to data
labeled_path = "data/mfc/immigration_labeled.json"
unlabeld_path = "data/mfc/immigration_unlabeled.json"
codes_path = "data/mfc/codes.json"

# Load data from path
def load_data():
    with open(labeled_path) as f:
        labeled = json.load(f)
    with open(unlabeld_path) as f:
        unlabeld = json.load(f)
    with open(codes_path) as f:
        codes = json.load(f)
    return labeled, unlabeld, codes

# Get labeled and unlabeled data as DataFrames
def get_labeled_data(labeled, codes):
    articles_list = []
    for article_id, article_data in labeled.items():
        text = article_data['text']
        primary_frame = article_data.get('primary_frame', "15.0")
        primary_frame = codes.get(str(primary_frame).split(".")[0] + ".0", "Unknown")
        sentences = sent_tokenize(text)
        for sentence in sentences:
            article = {
                'article_id': article_id,
                'text': sentence,
                'document_frame': primary_frame
            }
            articles_list.append(article)
    return pd.DataFrame(articles_list)

def get_unlabeled_data(unlabeld):
    articles_list = []
    for idx, article in enumerate(unlabeld):
        text = article['text']
        sentences = sent_tokenize(text)
        for sentence in sentences:
            article = {'article_id': f"unlabeled_{idx}", 'text': sentence}
            articles_list.append(article)
    return pd.DataFrame(articles_list)

# Preprocess labeled DataFrame
def preprocess_labeled_df(df):
    df = df[['article_id', 'text', 'document_frame']]
    return pd.concat([df, pd.get_dummies(df['document_frame'])], axis=1)

# Articles Dataset
class ArticlesDataset(Dataset):
    def __init__(self, articles, tokenizer):
        self.encodings = tokenizer(articles, max_length=512, truncation=True, padding='max_length', return_tensors='pt')

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# Train function
def train(epoch, model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'training_loss': f'{loss.item()/len(batch):.3f}'})

    # Calculate the average loss over all batches.
    avg_train_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch} - Average training loss: {avg_train_loss:.3f}")

# Test function
def test(model, test_loader, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(test_loader, desc="Evaluating")
    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()
            progress_bar.set_postfix({'evaluation_loss': f'{loss.item()/len(batch):.3f}'})

    # Calculate the average loss over all the batches.
    avg_test_loss = total_loss / len(test_loader)
    print(f"\nAverage test loss: {avg_test_loss:.3f}")

def train_model(model, train_loader, test_loader, epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train(epoch, model, train_loader, optimizer, device)
        test(model, test_loader, device)

        # Save the model after each epoch, overwriting the previous model
        model_save_path = './model_save/finetuned_model.pt'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a model on a text dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Input batch size for training (default: 32)")
    args = parser.parse_args()

    labeled, unlabeld, codes = load_data()
    df_labeled = get_labeled_data(labeled, codes)
    df_unlabeled = get_unlabeled_data(unlabeld)
    df_labeled = preprocess_labeled_df(df_labeled)
    df = pd.concat([df_labeled, df_unlabeled])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_df, test_df = train_test_split(df["text"].tolist(), test_size=0.2, random_state=42)
    train_dataset = ArticlesDataset(train_df, tokenizer)
    test_dataset = ArticlesDataset(test_df, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    train_model(model, train_loader, test_loader, batch_size=args.batch_size)

if __name__ == "__main__":
    main()