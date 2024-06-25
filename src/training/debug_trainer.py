# training/trainer.py

import torch
from tqdm import tqdm
import torch.nn.functional as F
import wandb


class DEBUGTrainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        loss_fn,
        optimizer,
        scheduler,
        accelerator,
        logger,
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accelerator = accelerator
        self.logger = logger

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc="Training"):
            print(f"Batch structure: {batch}")
            inputs = batch[0]
            labels = batch[1]
            self.optimizer.zero_grad()
            with self.accelerator.autocast():
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                inputs = batch[0]
                labels = batch[1]
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)
                correct_predictions += (preds == labels).sum().item()
                total_predictions += labels.size(0)
        accuracy = correct_predictions / total_predictions
        return total_loss / len(self.test_loader), accuracy

    def run_training(self, epochs, alpha):
        for epoch in range(epochs):
            train_loss = self.train_one_epoch()
            test_loss, accuracy = self.evaluate()
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )
            wandb.log(
                {
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "accuracy": accuracy,
                    "epoch": epoch + 1,
                }
            )
