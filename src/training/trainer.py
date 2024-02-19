import os
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, accuracy_score
import json
from tqdm import tqdm
import datetime
import math
import wandb


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_function,
        device="cuda",
        save_path="../notebooks/",
        wandb_project_name="your_project_name",
        wandb_api_key="your_api_key",
        tau_min=1,
        tau_decay=0.95,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.save_path = save_path

        # Initialize Weights & Biases
        wandb.login(key=wandb_api_key)

        self.run = wandb.init(
            project=wandb_project_name,
            config={
                "learning_rate": optimizer.param_groups[0]["lr"],
                "tau_min": tau_min,
                "tau_decay": tau_decay,
            },
        )

        self.tau_min = tau_min
        self.tau_decay = tau_decay

    def _train(self, epoch, model, train_dataloader, device, tau, alpha):
        model.train()
        total_loss, supervised_total_loss, unsupervised_total_loss = 0, 0, 0
        global global_steps

        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch}")
        ):
            global_steps += 1
            if global_steps % 50 == 0:
                tau = max(self.tau_min, math.exp(-self.tau_decay * global_steps))

            local_steps += 1

            self.optimizer.zero_grad()

            sentence_ids = batch["sentence_ids"].to(device)
            sentence_attention_masks = batch["sentence_attention_masks"].to(device)

            predicate_ids = batch["predicate_ids"].to(device)
            predicate_attention_masks = batch["predicate_attention_masks"].to(device)

            arg0_ids = batch["arg0_ids"].to(device)
            arg0_attention_masks = batch["arg0_attention_masks"].to(device)

            arg1_ids = batch["arg1_ids"].to(device)
            arg1_attention_masks = batch["arg1_attention_masks"].to(device)

            frameaxis_data = batch["frameaxis"].to(device)

            labels = batch["labels"].to(device)

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

            combined_loss = alpha * supervised_loss + (1 - alpha) * unsupervised_loss

            if torch.isnan(combined_loss):
                print(
                    f"NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}. Stopping..."
                )
                return

            combined_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # After the backward pass
            if any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in model.parameters()
            ):
                print(
                    f"NaN gradients detected at epoch {epoch+1}, batch {batch_idx+1}. Stopping..."
                )
                return

            self.optimizer.step()

            total_loss += combined_loss.item()
            supervised_total_loss += supervised_loss.item()
            unsupervised_total_loss += unsupervised_loss.item()
            # Log batch loss to wandb
            wandb.log({"batch_loss": combined_loss.item(), "epoch": epoch})

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

        print(
            f"Epoch {epoch}, Avg Total Loss: {avg_total_loss}, Avg Supervised Loss: {avg_supervised_loss}, Avg Unsupervised Loss: {avg_unsupervised_loss}"
        )
        wandb.log(
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

        with torch.no_grad():
            for batch in test_dataloader:
                sentence_ids = batch["sentence_ids"].to(device)
                sentence_attention_masks = batch["sentence_attention_masks"].to(device)

                predicate_ids = batch["predicate_ids"].to(device)
                predicate_attention_masks = batch["predicate_attention_masks"].to(
                    device
                )

                arg0_ids = batch["arg0_ids"].to(device)
                arg0_attention_masks = batch["arg0_attention_masks"].to(device)

                arg1_ids = batch["arg1_ids"].to(device)
                arg1_attention_masks = batch["arg1_attention_masks"].to(device)

                frameaxis_data = batch["frameaxis"].to(device)

                labels = batch["labels"].to(device)

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

        all_span_preds = np.vstack(span_preds)
        all_sentence_preds = np.vstack(sentence_preds)
        all_combined_preds = np.vstack(combined_preds)
        all_labels = np.vstack(all_labels)

        # Compute metrics
        f1_span_micro = f1_score(
            all_labels, all_span_preds, average="micro", zero_division=0
        )
        f1_sentence_micro = f1_score(
            all_labels, all_sentence_preds, average="micro", zero_division=0
        )
        f1_combined_micro = f1_score(
            all_labels, all_combined_preds, average="micro", zero_division=0
        )

        f1_span_macro = f1_score(
            all_labels, all_span_preds, average="macro", zero_division=0
        )
        f1_sentence_macro = f1_score(
            all_labels, all_sentence_preds, average="macro", zero_division=0
        )
        f1_combined_macro = f1_score(
            all_labels, all_combined_preds, average="macro", zero_division=0
        )

        accuracy_span = accuracy_score(all_labels, all_span_preds)
        accuracy_sentence = accuracy_score(all_labels, all_sentence_preds)
        accuracy_combined = accuracy_score(all_labels, all_combined_preds)

        print(
            f"Validation Metrics - micro F1 - Span: {f1_span_micro:.2f}, Sentence: {f1_sentence_micro:.2f}, Combined: {f1_combined_micro:.2f}, macro F1 - Span: {f1_span_macro:.2f}, Sentence: {f1_sentence_macro:.2f}, Combined: {f1_combined_macro:.2f}"
        )

        metrics = {
            "epoch": epoch,
            "accuracy_span": accuracy_span,
            "accuracy_sentence": accuracy_sentence,
            "accuracy_combined": accuracy_combined,
            "f1_span_micro": f1_span_micro,
            "f1_sentence_micro": f1_sentence_micro,
            "f1_combined_micro": f1_combined_micro,
            "f1_span_macro": f1_span_macro,
            "f1_sentence_macro": f1_sentence_macro,
            "f1_combined_macro": f1_combined_macro,
        }

        wandb.log(metrics)

        return metrics

    def _save_model(self, epoch, model, metrics):
        save_dir = os.path.join(self.save_path, "saved_models")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
        torch.save(
            model.state_dict(),
            model_save_path,
        )

        metrics_save_path = os.path.join(save_dir, f"metrics_epoch_{epoch}.json")
        with open(metrics_save_path, "w") as f:
            json.dump(metrics, f)

    def run_training(self, epochs, alpha=0.5):
        tau = 1
        scheduler = StepLR(self.optimizer, step_size=2, gamma=0.1)

        for epoch in range(epochs):
            self._train(
                epoch,
                self.model,
                self.train_dataloader,
                self.optimizer,
                self.device,
                tau,
                alpha,
            )
            metrics = self._evaluate(
                epoch, self.model, self.test_dataloader, self.device, tau
            )

            scheduler.step()

            self._save_model(epoch, self.model, metrics)
