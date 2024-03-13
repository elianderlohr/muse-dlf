import os
import numpy as np
import torch

from sklearn.metrics import f1_score, accuracy_score
import json
from tqdm import tqdm
from datetime import datetime
import math
import evaluate


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_function,
        scheduler,
        device="cuda",
        save_path="../notebooks/",
        accelerator=None,
        tau_min=1,
        tau_decay=0.95,
    ):
        """
        Initializes the Trainer.

        Args:
            model: The model to be trained.
            train_dataloader: The DataLoader for the training data.
            test_dataloader: The DataLoader for the testing data.
            optimizer: The optimizer to be used for training.
            loss_function: The loss function to be used for training.
            scheduler: The learning rate scheduler to be used for training.
            device: The device to be used for training.
            save_path: The path to save the model and metrics.
            accelerator: The accelerator to be used for training.
            tau_min: The minimum value of tau.
            tau_decay: The decay factor for tau.

        Returns:
            None
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler

        self.save_path = save_path

        self.accelerator = accelerator

        if device == "cuda":
            self.device = accelerator.device
        else:
            self.device = device

        self.tau_min = tau_min
        self.tau_decay = tau_decay

    def _train(self, epoch, model, train_dataloader, tau, alpha, device):
        model.train()
        total_loss, supervised_total_loss, unsupervised_total_loss = 0, 0, 0
        global global_steps

        local_steps = 0
        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc=f"Train - Epoch {epoch}")
        ):
            global_steps += 1
            if global_steps % 50 == 0:
                tau = max(self.tau_min, math.exp(-self.tau_decay * global_steps))

            local_steps += 1

            self.optimizer.zero_grad()

            sentence_ids = batch["sentence_ids"]
            sentence_attention_masks = batch["sentence_attention_masks"]

            predicate_ids = batch["predicate_ids"]
            predicate_attention_masks = batch["predicate_attention_masks"]

            arg0_ids = batch["arg0_ids"]
            arg0_attention_masks = batch["arg0_attention_masks"]

            arg1_ids = batch["arg1_ids"]
            arg1_attention_masks = batch["arg1_attention_masks"]

            frameaxis_data = batch["frameaxis"]

            labels = batch["labels"]

            unsupervised_loss, span_logits, sentence_logits, _ = model(
                sentence_ids,
                sentence_attention_masks,
                predicate_ids,
                predicate_attention_masks,
                arg0_ids,
                arg0_attention_masks,
                arg1_ids,
                arg1_attention_masks,
                frameaxis_data,
                tau,
            )

            span_loss = 0.0
            sentence_loss = 0.0

            span_loss = self.loss_function(span_logits, labels.float())
            sentence_loss = self.loss_function(sentence_logits, labels.float())

            supervised_loss = span_loss + sentence_loss

            sum_of_parameters = sum(p.sum() for p in model.parameters())

            zero_sum = sum_of_parameters * 0.0

            combined_loss = (
                alpha * supervised_loss + (1 - alpha) * unsupervised_loss
            ) + zero_sum

            self.accelerator.backward(combined_loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += combined_loss.item()
            supervised_total_loss += supervised_loss.item()
            unsupervised_total_loss += unsupervised_loss.item()
            # Log batch loss to wandb
            self.accelerator.log({"batch_loss": combined_loss.item(), "epoch": epoch})

            del (
                sentence_ids,
                predicate_ids,
                arg0_ids,
                arg1_ids,
                labels,
                unsupervised_loss,
            )
            torch.cuda.empty_cache()

        avg_total_loss = total_loss / len(train_dataloader)
        avg_supervised_loss = supervised_total_loss / len(train_dataloader)
        avg_unsupervised_loss = unsupervised_total_loss / len(train_dataloader)

        self.accelerator.print(
            f"Epoch {epoch}, Avg Total Loss: {avg_total_loss}, Avg Supervised Loss: {avg_supervised_loss}, Avg Unsupervised Loss: {avg_unsupervised_loss}"
        )
        self.accelerator.log(
            {
                "epoch_avg_total_loss": avg_total_loss,
                "epoch_avg_supervised_loss": avg_supervised_loss,
                "epoch_avg_unsupervised_loss": avg_unsupervised_loss,
                "epoch": epoch,
            }
        )

    def _evaluate(self, epoch, model, test_dataloader, device, tau):
        model.eval()

        span_preds = []
        sentence_preds = []
        combined_preds = []
        all_labels = []

        # Load the evaluate metrics
        f1_metric_micro = evaluate.load("f1", config_name="micro")
        f1_metric_macro = evaluate.load("f1", config_name="macro")
        accuracy_metric = evaluate.load("accuracy")

        for batch_idx, batch in enumerate(
            tqdm(test_dataloader, desc=f"Evaluate - Epoch {epoch}")
        ):
            sentence_ids = batch["sentence_ids"]
            sentence_attention_masks = batch["sentence_attention_masks"]

            predicate_ids = batch["predicate_ids"]
            predicate_attention_masks = batch["predicate_attention_masks"].to(device)

            arg0_ids = batch["arg0_ids"]
            arg0_attention_masks = batch["arg0_attention_masks"]

            arg1_ids = batch["arg1_ids"]
            arg1_attention_masks = batch["arg1_attention_masks"]

            frameaxis_data = batch["frameaxis"]

            labels = batch["labels"]

            with torch.no_grad():
                _, span_logits, sentence_logits, combined_logits = model(
                    sentence_ids,
                    sentence_attention_masks,
                    predicate_ids,
                    predicate_attention_masks,
                    arg0_ids,
                    arg0_attention_masks,
                    arg1_ids,
                    arg1_attention_masks,
                    frameaxis_data,
                    tau,
                )

            span_pred = (torch.softmax(span_logits, dim=1) > 0.5).int()
            sentence_pred = (torch.softmax(sentence_logits, dim=1) > 0.5).int()
            combined_pred = (torch.softmax(combined_logits, dim=1) > 0.5).int()

            span_logits = self.accelerator.gather_for_metrics((span_logits, labels))
            sentence_logits = self.accelerator.gather_for_metrics(
                (sentence_logits, labels)
            )
            combined_logits = self.accelerator.gather_for_metrics(
                (combined_logits, labels)
            )

            span_preds.append(span_pred.cpu().numpy())
            sentence_preds.append(sentence_pred.cpu().numpy())
            combined_preds.append(combined_pred.cpu().numpy())

            all_labels.append(labels.cpu().numpy())

            # Explicitly delete tensors to free up memory
            del (
                sentence_ids,
                predicate_ids,
                arg0_ids,
                arg1_ids,
                labels,
                span_logits,
                sentence_logits,
                sentence_pred,
            )
            torch.cuda.empty_cache()

        # Calculate metrics for span predictions
        span_results_micro = f1_metric_micro.compute(
            predictions=span_preds, references=all_labels
        )
        span_results_macro = f1_metric_macro.compute(
            predictions=span_preds, references=all_labels
        )
        span_accuracy = accuracy_metric.compute(
            predictions=span_preds, references=all_labels
        )

        # Calculate metrics for sentence predictions
        sentence_results_micro = f1_metric_micro.compute(
            predictions=sentence_preds, references=all_labels
        )
        sentence_results_macro = f1_metric_macro.compute(
            predictions=sentence_preds, references=all_labels
        )
        sentence_accuracy = accuracy_metric.compute(
            predictions=sentence_preds, references=all_labels
        )

        # Calculate metrics for combined predictions
        combined_results_micro = f1_metric_micro.compute(
            predictions=combined_preds, references=all_labels
        )
        combined_results_macro = f1_metric_macro.compute(
            predictions=combined_preds, references=all_labels
        )
        combined_accuracy = accuracy_metric.compute(
            predictions=combined_preds, references=all_labels
        )

        self.accelerator.print("Span Metrics:")
        self.accelerator.print(
            f"F1 Micro: {span_results_micro['f1']:.4f}, F1 Macro: {span_results_macro['f1']:.4f}, Accuracy: {span_accuracy['accuracy']:.4f}"
        )

        self.accelerator.print("Sentence Metrics:")
        self.accelerator.print(
            f"F1 Micro: {sentence_results_micro['f1']:.4f}, F1 Macro: {sentence_results_macro['f1']:.4f}, Accuracy: {sentence_accuracy['accuracy']:.4f}"
        )

        self.accelerator.print("Combined Metrics:")
        self.accelerator.print(
            f"F1 Micro: {combined_results_micro['f1']:.4f}, F1 Macro: {combined_results_macro['f1']:.4f}, Accuracy: {combined_accuracy['accuracy']:.4f}"
        )

        metrics = {
            "span_results_micro": span_results_micro,
            "span_results_macro": span_results_macro,
            "span_accuracy": span_accuracy,
            "sentence_results_micro": sentence_results_micro,
            "sentence_results_macro": sentence_results_macro,
            "sentence_accuracy": sentence_accuracy,
            "combined_results_micro": combined_results_micro,
            "combined_results_macro": combined_results_macro,
            "combined_accuracy": combined_accuracy,
            "epoch": epoch,
        }

        self.accelerator.log(metrics)

        return metrics

    def _save_model(self, epoch, model, metrics):
        save_dir = os.path.join(self.save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        torch.save(
            model.state_dict(),
            model_save_path,
        )

        metrics_save_path = os.path.join(save_dir, f"metrics_epoch_{epoch}.json")
        with open(metrics_save_path, "w") as f:
            json.dump(metrics, f)

    def run_training(self, epochs, alpha=0.5):
        tau = 1

        global global_steps
        global_steps = 0

        for epoch in range(epochs):
            self._train(
                epoch + 1,
                self.model,
                self.train_dataloader,
                tau,
                alpha,
                self.device,
            )
            metrics = self._evaluate(
                epoch, self.model, self.test_dataloader, self.device, tau
            )

            self.scheduler.step()

            self._save_model(epoch, self.model, metrics)
