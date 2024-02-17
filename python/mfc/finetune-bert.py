# Imports
import argparse
import os
import json
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW, Trainer, TrainingArguments, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from pathlib import Path
from contextlib import redirect_stdout
import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths to data
labeled_path = "data/mfc/immigration_labeled.json"
unlabeld_path = "data/mfc/immigration_unlabeled.json"
codes_path = "data/mfc/codes.json"

def setup_logger(save_path):
    log_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_training.log"
    log_filepath = os.path.join(save_path, log_filename)
    file_handler = logging.FileHandler(log_filepath, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger, log_filepath

def download_nltk_resources():
    with open(os.devnull, 'w') as f:
        with redirect_stdout(f):
            nltk.download("punkt")

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
def train(epoch, model, train_loader, optimizer, device, scheduler, logger, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0.0
    model.zero_grad()

    progress_interval = len(train_loader) // 20

    for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} - Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            optimizer.step()
            scheduler.step()  # Update the learning rate.
            model.zero_grad()

        if step % progress_interval == 0:
            logger.info(f"Epoch {epoch} - Step {step+1}/{len(train_loader)} - Loss: {loss.item()}")

        total_loss += loss.item()

    # Logging
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Epoch {epoch} - Average training loss: {avg_loss:.4f}")


def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0

    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))

    return avg_loss, perplexity.item()


def train_model(model, train_loader, test_loader, epochs=3, save_path="models/finetuned-roberta/", logger=None, gradient_accumulation_steps=1):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Assuming 'train_dataset' is your dataset for training
    total_steps = len(train_loader) // gradient_accumulation_steps * epochs

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=0, 
                                                num_training_steps=total_steps)

    # Modified train loop with evaluation and logging
    for epoch in range(epochs):
        logger.info(f"\nEpoch {epoch+1}/{epochs}")
        train(epoch+1, model, train_loader, optimizer, device, scheduler, logger, gradient_accumulation_steps=gradient_accumulation_steps)

        avg_test_loss, test_perplexity = evaluate_model(model, test_loader, device)
        logger.info(f"Epoch {epoch+1} - Validation Loss: {avg_test_loss}, Perplexity: {test_perplexity}")

        # Save the model after each epoch
        model_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_roberta_finetuned_epoch_" + str(epoch+1) + ".pth"
        model_save_path = os.path.join(save_path, model_name)
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune a model on a text dataset.")
    parser.add_argument("--batch_size", type=int, default=32, help="Input batch size for training (default: 32)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train (default: 3)")
    # save_path = os.path.join(os.getcwd(), "models/finetuned-roberta/")
    parser.add_argument("--save_path", type=str, default="models/finetuned-roberta/", help="Path to save the finetuned model (default: models/finetuned-roberta/)")
    parser.add_argument("--load_model_path", type=str, default="", help="Path to load a pre-trained model (default: '')")
    args = parser.parse_args()

    logger, log_filepath = setup_logger(args.save_path)

    logger.info("Command-line arguments: " + str(args))

    logger.info(f"Training started. Logging to {log_filepath}")

    # Create the output directory if it doesn't exist
    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")

    labeled, unlabeld, codes = load_data()
    df_labeled = get_labeled_data(labeled, codes)
    df_unlabeled = get_unlabeled_data(unlabeld)
    df_labeled = preprocess_labeled_df(df_labeled)
    df = pd.concat([df_labeled, df_unlabeled])
    
    logger.info("Data loaded successfully.")

    logger.info("Loading model...")

    model = RobertaForMaskedLM.from_pretrained('roberta-base')

    if args.load_model_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        logger.info(f"Loaded model from {args.load_model_path}")
    
    logger.info("Model loaded successfully.")

    logger.info("Preprocessing data...")
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_df, test_df = train_test_split(df["text"].tolist(), test_size=0.2, random_state=42)
    train_dataset = ArticlesDataset(train_df, tokenizer)
    test_dataset = ArticlesDataset(test_df, tokenizer)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=data_collator)

    logger.info("Data preprocessed successfully.")

    train_model(model, train_loader, test_loader, epochs=args.epochs, save_path=args.save_path, logger=logger)

if __name__ == "__main__":
    welcome_message = """
    ############################################################################################################
    #                                                                                                          #
    #  Welcome to the fine-tuning script for RoBERTa!                                                          #
    #                                                                                                          #
    #  This script fine-tunes a RoBERTa model on a text dataset.                                               #
    #                                                                                                          #
    #  The script takes the following command-line arguments:                                                  #
    #                                                                                                          #
    #  --batch_size: Input batch size for training (default: 32)                                               #
    #  --epochs: Number of epochs to train (default: 3)                                                        #
    #  --save_path: Path to save the finetuned model (default: models/finetuned-roberta/)                      #
    #                                                                                                          #
    ############################################################################################################
    """

    print(welcome_message)

    print("Downloading NLTK resources...")

    download_nltk_resources()

    print("NLTK resources downloaded successfully.")

    main()