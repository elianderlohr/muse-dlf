from logging import config
import os
import time
import torch
import json
from tqdm import tqdm
import math
import evaluate
from wandb import AlertLevel
from utils.logging_manager import LoggerManager
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report

from typing import Dict, List, Literal, Tuple

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

logger = LoggerManager.get_logger(__name__)

import wandb

wandb.require("core")


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        loss_function,
        scheduler,
        model_type="muse-dlf",  # muse or slmuse
        device="cuda",
        save_path="../notebooks/",
        run_name="",
        training_management: Literal[
            "accelerate", "wandb", None
        ] = None,  # 'accelerate', 'wandb', or None
        tau_min=1,
        tau_decay=0.95,
        early_stopping_patience=30,
        mixed_precision="fp16",  # "fp16"
        clip_value=1.0,
        accumulation_steps=1,
        test_every_n_batches=50,
        save_threshold=0.5,
        save_metric="accuracy",
        model_config={},
        save_model=True,
        **kwargs,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.scheduler: CosineAnnealingWarmRestarts = scheduler

        self.device = device
        self.save_path = save_path
        self.run_name = run_name
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.early_stopping_patience = early_stopping_patience

        self.test_every_n_batches = test_every_n_batches
        self.save_threshold = save_threshold
        self.save_metric = save_metric

        self.model_config = model_config

        self.save_model = save_model

        # Initialize the mixed precision
        logger.info(
            f"Mixed precision is enabled: {mixed_precision in ['fp16', 'bf16', 'fp32']}, therefore set mixed precision to: {mixed_precision}"
        )

        self.scaler = (
            GradScaler() if mixed_precision in ["fp16", "bf16", "fp32"] else None
        )

        self.training_management = training_management
        if self.training_management == "accelerate":
            logger.info("Using Accelerate for training.")
            from accelerate import Accelerator

            if "accelerator_instance" in kwargs:
                self.accelerator: Accelerator = kwargs["accelerator_instance"]
                self.wandb = self.accelerator.get_tracker("wandb")
            else:
                raise ValueError(
                    "You must provide an accelerator instance if you want to use Accelerate for training."
                )
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.test_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.test_dataloader,
                self.scheduler,
            )
        elif self.training_management == "wandb":
            logger.info("Using Weights and Biases for training.")
            import wandb

            if "wandb_instance" in kwargs:
                self.wandb: wandb = kwargs["wandb_instance"]
            else:
                raise ValueError(
                    "You must provide a wandb instance if you want to use wandb for training."
                )
        else:
            logger.info("Using standard PyTorch for training.")
            self.accelerator = None

        # Get the model type
        self.model_type = model_type

        logger.info(f"Model type: {self.model_type}")

        self.mixed_precision = mixed_precision
        self.clip_value = clip_value
        self.accumulation_steps = accumulation_steps

    def _log_metrics(self, metrics):
        if self.training_management == "wandb":
            self.wandb.log(metrics)
        elif self.training_management == "accelerate":
            self.accelerator.log(metrics)
        else:
            logger.info(metrics)

    def _log_alert(self, title, text):
        if self.training_management == "wandb":
            self.wandb.alert(title=title, text=text, level=AlertLevel.INFO)
        else:
            logger.warning(f"{title} - {text}")

    def check_for_nans(self, tensor, tensor_name):
        if torch.isnan(tensor).any():
            logger.error(f"NaNs found in {tensor_name}")
            return True
        return False

    def get_activation_function(self, logits):
        if self.model_type == "muse-dlf":
            return (torch.sigmoid(logits) > 0.5).int()
        elif self.model_type == "slmuse-dlf":
            return (torch.softmax(logits, dim=1) > 0.5).int()
        else:
            raise ValueError(
                f"Model type {self.model_type} not supported: only muse-dlf and slmuse-dlf are supported."
            )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def prepare_predictions_and_labels(
        self,
        model_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        prediction_types = [
            "supervised",
            "span",
            "sent",
            "predicate",
            "arg0",
            "arg1",
            "frameaxis",
        ]
        results = {}

        for pred_type in prediction_types:
            if pred_type == "supervised":
                preds = model_outputs["supervised_logits"]
            elif pred_type in ["span", "sent"]:
                preds = model_outputs[f"{pred_type}_logits"]
            else:
                preds = model_outputs["other_outputs"][f"{pred_type}_logits"]

            # Apply activation function
            preds = self.get_activation_function(preds)

            # Gather predictions if using Accelerate
            if self.training_management == "accelerate":
                preds, labels_gathered = self.accelerator.gather_for_metrics(
                    (preds, labels)
                )
            else:
                labels_gathered = labels

            results[pred_type] = (preds, labels_gathered)

        return results

    def add_batch_to_metrics(
        self,
        metrics: Dict[str, object],
        predictions: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ):
        for pred_type, (preds, labels) in predictions.items():
            if self.model_type == "muse-dlf":
                # Multi-label case
                metrics[f"f1_metric_micro_{pred_type}"].add_batch(
                    predictions=preds, references=labels
                )
                metrics[f"f1_metric_macro_{pred_type}"].add_batch(
                    predictions=preds, references=labels
                )
                metrics[f"accuracy_metric_{pred_type}"].add_batch(
                    predictions=preds, references=labels
                )
            elif self.model_type == "slmuse-dlf":
                preds = preds.argmax(dim=1)
                labels = labels.argmax(dim=1)

                # Single-label case
                metrics[f"f1_metric_micro_{pred_type}"].add_batch(
                    predictions=preds, references=labels
                )
                metrics[f"f1_metric_macro_{pred_type}"].add_batch(
                    predictions=preds, references=labels
                )
                metrics[f"accuracy_metric_{pred_type}"].add_batch(
                    predictions=preds, references=labels
                )

    def compute_metrics(self, metrics: Dict[str, object]) -> Dict[str, float]:
        results = {}
        for metric_name, metric in metrics.items():
            if "f1_metric_micro" in metric_name:
                results[f"{metric_name.replace("f1_metric_micro_", "")}_micro_f1"] = metric.compute(
                    average="micro"
                )["f1"]
            elif "f1_metric_macro" in metric_name:
                results[f"{metric_name.replace("f1_metric_macro_", "")}_macro_f1"] = metric.compute(
                    average="macro"
                )["f1"]
            elif "accuracy_metric" in metric_name:
                results[f"{metric_name.replace("accuracy_metric_", "")}_accuracy"] = metric.compute()["accuracy"]
        return results

    def log_per_class_metrics(self, class_report: Dict, prefix="train"):
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict):
                self._log_metrics(
                    {
                        f"{prefix}_precision_class_{class_name}": metrics["precision"],
                        f"{prefix}_recall_class_{class_name}": metrics["recall"],
                        f"{prefix}_f1_class_{class_name}": metrics["f1-score"],
                    }
                )

    def calculate_loss(self, outputs, labels, alpha):
        unsupervised_loss = outputs["unsupervised_loss"]
        span_loss = self.loss_function(
            outputs["span_logits"], labels, input_type="logits"
        )
        sentence_loss = self.loss_function(
            outputs["sent_logits"], labels, input_type="logits"
        )

        supervised_loss = span_loss + sentence_loss
        combined_loss = alpha * supervised_loss + (1 - alpha) * unsupervised_loss

        return combined_loss, {
            "unsupervised_loss": unsupervised_loss,
            "span_loss": span_loss,
            "sentence_loss": sentence_loss,
            "supervised_loss": supervised_loss,
            "combined_loss": combined_loss,
        }

    def _train(
        self,
        epoch,
        train_dataloader,
        tau,
        alpha,
        experiment_id,
        device,
        early_stopping={
            "best_accuracy": 0,
            "best_micro_f1": 0,
            "best_macro_f1": 0,
            "early_stop": 0,
            "early_stopped": False,
        },
    ):
        self.model.train()
        total_loss, supervised_total_loss, unsupervised_total_loss = 0, 0, 0

        # Initialize metrics
        metric_types = ["f1_metric_micro", "f1_metric_macro", "accuracy_metric"]
        prediction_types = [
            "supervised",
            "span",
            "sentence",
            "predicate",
            "arg0",
            "arg1",
            "frameaxis",
        ]
        metrics = {
            f"{metric}_{pred_type}": evaluate.load(
                metric.split("_")[0],
                config_name=(
                    metric.split("_")[2] if len(metric.split("_")) > 2 else None
                ),
                experiment_id=experiment_id,
            )
            for metric in metric_types
            for pred_type in prediction_types
        }

        precision_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16 if self.mixed_precision == "bf16" else torch.float32
        )

        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc=f"Train - Epoch {epoch}")
        ):
            global global_steps
            global_steps += 1

            # Update tau every 50 steps
            if global_steps % 50 == 0:
                tau = max(self.tau_min, math.exp(-self.tau_decay * global_steps))

            # Prepare inputs
            inputs = {
                k: v.to(device) if self.training_management != "accelerate" else v
                for k, v in batch.items()
                if k != "labels"
            }

            # Extract necessary items
            model_inputs = {
                "sentence_ids": inputs["sentence_ids"],
                "sentence_attention_masks": inputs["sentence_attention_masks"],
                "predicate_ids": inputs["predicate_ids"],
                "arg0_ids": inputs["arg0_ids"],
                "arg1_ids": inputs["arg1_ids"],
                "frameaxis_data": inputs["frameaxis"],
                "tau": tau,
            }

            labels = (
                batch["labels"].to(device)
                if self.training_management != "accelerate"
                else batch["labels"]
            )

            if self.model_type == "muse-dlf":
                prepared_labels = labels.float()
            elif self.model_type == "slmuse-dlf":
                if labels.dim() == 2:
                    logger.debug(
                        "Labels are one-hot encoded, converting to class index."
                    )
                    prepared_labels = torch.argmax(labels, dim=1).long()

            # Forward pass
            if self.training_management == "accelerate":
                with self.accelerator.autocast():
                    outputs = self.model(**model_inputs)
                    total_loss, loss_dict = self.calculate_loss(
                        outputs, prepared_labels, alpha
                    )
            else:
                with autocast(
                    enabled=self.mixed_precision in ["fp16", "bf16", "fp32"],
                    dtype=precision_dtype,
                ):
                    outputs = self.model(**model_inputs)
                    total_loss, loss_dict = self.calculate_loss(
                        outputs, prepared_labels, alpha
                    )

            # Backward pass and optimization
            if self.training_management == "accelerate":
                with self.accelerator.accumulate(self.model):
                    self.accelerator.backward(total_loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.clip_value
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                if self.scaler is not None:
                    self.scaler.scale(total_loss / self.accumulation_steps).backward()
                    if (batch_idx + 1) % self.accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(train_dataloader):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_value
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                else:
                    (total_loss / self.accumulation_steps).backward()
                    if (batch_idx + 1) % self.accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(train_dataloader):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_value
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

            total_loss += loss_dict["combined_loss"].item()
            supervised_total_loss += loss_dict["supervised_loss"].item()
            unsupervised_total_loss += loss_dict["unsupervised_loss"].item()

            current_lr_scheduler = self.scheduler.get_last_lr()[0]
            current_lr_model = self.get_lr()
            self._log_metrics(
                {
                    "batch_combined_loss": loss_dict["combined_loss"].item(),
                    "batch_supervised_loss": loss_dict["supervised_loss"].item(),
                    "batch_span_loss": loss_dict["span_loss"].item(),
                    "batch_sentence_loss": loss_dict["sentence_loss"].item(),
                    "batch_unsupervised_loss": loss_dict["unsupervised_loss"].item(),
                    "batch": batch_idx,
                    "global_steps": global_steps,
                    "tau": tau,
                    "epoch": epoch,
                    "learning_rate_scheduler": current_lr_scheduler,
                    "learning_rate_model": current_lr_model,
                }
            )

            # Evaluate and log train metrics every test_every_n_batches steps
            if global_steps % self.test_every_n_batches == 0:
                logger.info(
                    f"[TRAIN] Starting to evaluate the model at epoch {epoch}, batch {global_steps}"
                )

                with torch.no_grad():
                    predictions = self.prepare_predictions_and_labels(
                        outputs,
                        labels,
                    )
                    self.add_batch_to_metrics(metrics, predictions)

                    eval_results = self.compute_metrics(metrics)
                    self._log_metrics(
                        {f"train_{k}": v for k, v in eval_results.items()}
                    )

                    # Generate and log per-class metrics for combined predictions
                    combined_pred, combined_labels = predictions["supervised"]
                    class_report = classification_report(
                        combined_labels.cpu().numpy(),
                        combined_pred.cpu().numpy(),
                        output_dict=True,
                    )
                    self.log_per_class_metrics(class_report, prefix="train")

                    logger.info(
                        f"[TRAIN] Epoch {epoch}, Step {global_steps}: "
                        f"Micro F1: {eval_results['combined_micro_f1']}, "
                        f"Macro F1: {eval_results['combined_macro_f1']}, "
                        f"Accuracy: {eval_results['combined_accuracy']}"
                    )

                    if (
                        eval_results[f"combined_{self.save_metric}"]
                        >= early_stopping[f"best_{self.save_metric}"]
                    ):
                        early_stopping["best_accuracy"] = eval_results[
                            "combined_accuracy"
                        ]
                        early_stopping["best_micro_f1"] = eval_results[
                            "combined_micro_f1"
                        ]
                        early_stopping["best_macro_f1"] = eval_results[
                            "combined_macro_f1"
                        ]
                        early_stopping["early_stop"] = 0

                        if (
                            eval_results[f"combined_{self.save_metric}"]
                            > self.save_threshold
                        ):
                            self._save_model(f"step_{global_steps}")
                    else:
                        early_stopping["early_stop"] += 1

                        if early_stopping["early_stop"] >= self.early_stopping_patience:
                            logger.info("Early stopping triggered.")
                            early_stopping["early_stopped"] = True
                            return tau, early_stopping

                self.model.train()

        avg_total_loss = total_loss / len(train_dataloader)
        avg_supervised_loss = supervised_total_loss / len(train_dataloader)
        avg_unsupervised_loss = unsupervised_total_loss / len(train_dataloader)

        logger.info(
            f"[TRAIN] Epoch {epoch}, Step {global_steps}: "
            f"Avg Total Loss: {avg_total_loss}, "
            f"Avg Supervised Loss: {avg_supervised_loss}, "
            f"Avg Unsupervised Loss: {avg_unsupervised_loss}"
        )

        self._log_metrics(
            {
                "epoch_combined_loss": avg_total_loss,
                "epoch_supervised_loss": avg_supervised_loss,
                "epoch_unsupervised_loss": avg_unsupervised_loss,
                "epoch": epoch,
            }
        )

        self._save_model(f"epoch_{epoch}")

        return tau, early_stopping

    def _evaluate(self, epoch, test_dataloader, device, tau, alpha, experiment_id):
        self.model.eval()
        total_val_loss = 0.0

        # Initialize metrics
        metric_types = ["f1_metric_micro", "f1_metric_macro", "accuracy_metric"]
        prediction_types = [
            "supervised",
            "span",
            "sent",
            "predicate",
            "arg0",
            "arg1",
            "frameaxis",
        ]
        metrics = {
            f"{metric}_{pred_type}": evaluate.load(
                metric.split("_")[0],
                config_name=(
                    metric.split("_")[2] if len(metric.split("_")) > 2 else None
                ),
                experiment_id=experiment_id,
            )
            for metric in metric_types
            for pred_type in prediction_types
        }

        precision_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16 if self.mixed_precision == "bf16" else torch.float32
        )

        all_combined_preds = []
        all_combined_labels = []

        for batch in tqdm(test_dataloader, desc=f"Evaluate - Epoch {epoch}"):
            inputs = {
                k: v.to(device) if self.training_management != "accelerate" else v
                for k, v in batch.items()
                if k != "labels"
            }

            model_inputs = {
                "sentence_ids": inputs["sentence_ids"],
                "sentence_attention_masks": inputs["sentence_attention_masks"],
                "predicate_ids": inputs["predicate_ids"],
                "arg0_ids": inputs["arg0_ids"],
                "arg1_ids": inputs["arg1_ids"],
                "frameaxis_data": inputs["frameaxis"],
                "tau": tau,
            }

            labels = (
                batch["labels"].to(device)
                if self.training_management != "accelerate"
                else batch["labels"]
            )

            if self.model_type == "muse-dlf":
                prepared_labels = labels.float()
            elif self.model_type == "slmuse-dlf":
                if labels.dim() == 2:
                    logger.debug(
                        "Labels are one-hot encoded, converting to class index."
                    )
                    prepared_labels = torch.argmax(labels, dim=1).long()

            with torch.no_grad():
                if self.training_management == "accelerate":
                    with self.accelerator.autocast():
                        outputs = self.model(**model_inputs)

                        # Calculate losses
                        unsupervised_loss = outputs["unsupervised_loss"]
                        span_loss = self.loss_function(
                            outputs["span_logits"], prepared_labels, input_type="logits"
                        )
                        sentence_loss = self.loss_function(
                            outputs["sent_logits"], prepared_labels, input_type="logits"
                        )
                        supervised_loss = span_loss + sentence_loss
                        combined_loss = (
                            alpha * supervised_loss + (1 - alpha) * unsupervised_loss
                        )
                else:
                    with autocast(
                        enabled=self.mixed_precision in ["fp16", "bf16", "fp32"],
                        dtype=precision_dtype,
                    ):
                        outputs = self.model(**model_inputs)

                        # Calculate losses
                        unsupervised_loss = outputs["unsupervised_loss"]
                        span_loss = self.loss_function(
                            outputs["span_logits"], prepared_labels, input_type="logits"
                        )
                        sentence_loss = self.loss_function(
                            outputs["sent_logits"], prepared_labels, input_type="logits"
                        )
                        supervised_loss = span_loss + sentence_loss
                        combined_loss = (
                            alpha * supervised_loss + (1 - alpha) * unsupervised_loss
                        )

                total_val_loss += combined_loss.item()

                predictions = self.prepare_predictions_and_labels(
                    outputs,
                    labels,
                )
                self.add_batch_to_metrics(metrics, predictions)

                combined_pred, combined_labels = predictions["supervised"]
                all_combined_preds.append(combined_pred.cpu())
                all_combined_labels.append(combined_labels.cpu())

        avg_val_loss = total_val_loss / len(test_dataloader)

        # Concatenate all predictions and labels
        all_combined_preds = torch.cat(all_combined_preds).numpy()
        all_combined_labels = torch.cat(all_combined_labels).numpy()

        # Generate classification report
        class_report = classification_report(
            all_combined_labels, all_combined_preds, output_dict=True
        )

        if (
            self.training_management == "accelerate"
            and self.accelerator.is_main_process
        ) or (self.training_management != "accelerate"):
            logger.info("\nPer-class metrics for evaluation data:")
            logger.info(classification_report(all_combined_labels, all_combined_preds))

        # Compute all metrics
        eval_results = self.compute_metrics(metrics)

        logger.info(
            f"[EVALUATE] Epoch {epoch}: "
            f"Micro F1: {eval_results['combined_micro_f1']}, "
            f"Macro F1: {eval_results['combined_macro_f1']}, "
            f"Accuracy: {eval_results['combined_accuracy']}"
        )

        # Prepare metrics dictionary
        metrics_dict = {
            **{f"{k}": v for k, v in eval_results.items()},
            "epoch": epoch,
            "val_loss": avg_val_loss,
        }

        self._log_metrics(metrics_dict)

        # Log per-class metrics
        self.log_per_class_metrics(class_report, prefix="eval")

        return metrics_dict

    def _save_model(self, epoch_step=None):
        logger.info("Starting to save the model.")

        if self.save_model == False:
            logger.info("Model saving is disabled.")
            return

        if (
            self.training_management == "accelerate"
            and self.accelerator.is_main_process
        ) or self.training_management == "wandb":

            # save dir path
            save_dir = os.path.join(self.save_path)
            try:
                os.makedirs(save_dir, exist_ok=True)
                logger.info(f"Created directory: {save_dir}")
            except PermissionError as e:
                logger.error(
                    f"Permission denied: Cannot create directory {save_dir}. Exception: {e}"
                )
                return
            except Exception as e:
                logger.error(
                    f"Warning: Could not create directory {save_dir}. Exception: {e}"
                )
                return

            # save model
            if (
                self.training_management == "accelerate"
                and self.accelerator.is_main_process
            ):
                try:
                    # Save the model using accelerator.save_model
                    self.accelerator.save_model(
                        self.model,
                        save_dir,
                        safe_serialization=False,
                    )
                    logger.info(f"Model saved at {save_dir}")
                except Exception as e:
                    logger.error(
                        f"Warning: Failed to save model at {save_dir}. Exception: {e}"
                    )
                    return
            elif self.training_management != "accelerate":
                try:
                    # Validate model state dict
                    state_dict = self.model.state_dict()
                    torch.save(state_dict, save_dir)
                    logger.info(f"Model saved at {save_dir}")
                except Exception as e:
                    logger.error(
                        f"Warning: Failed to save model at {save_dir}. Exception: {e}"
                    )
                    return

            # save config file
            config_save_path = os.path.join(save_dir, "config.json")
            if (
                self.training_management == "accelerate"
                and self.accelerator.is_main_process
            ):
                try:
                    self.accelerator.save(self.model_config, config_save_path)
                    logger.info(f"Config saved at {config_save_path}")
                except Exception as e:
                    logger.error(
                        f"Warning: Failed to save config at {config_save_path}. Exception: {e}"
                    )
            elif self.training_management != "accelerate":
                try:
                    with open(config_save_path, "w") as f:
                        json.dump(self.model_config, f)
                    logger.info(f"Config saved at {config_save_path}")
                except Exception as e:
                    logger.error(
                        f"Warning: Failed to save config at {config_save_path}. Exception: {e}"
                    )

            # save model artifact
            model_artifact = wandb.Artifact(
                name=f"{self.run_name.replace('-', '_')}_model",
                type="model",
                metadata=self.model_config,
            )

            model_artifact.add_dir(save_dir)

            # Save to wandb
            if self.training_management == "wandb":
                logger.info(
                    "Use wandb object to save artifacts as training_mode='wandb'"
                )
                try:
                    logged_artifact = self.wandb.log_artifact(model_artifact)
                    logged_artifact.wait()
                    logger.info(f"Model artifact logged to Weights and Biases.")
                except Exception as e:
                    logger.error(
                        f"Failed to log artifact to Weights and Biases. Exception: {e}"
                    )

            if self.training_management == "accelerate":
                logger.info(
                    "Use wandb accelerate tracker object to save artifacts as training_mode='accelerate'"
                )
                try:
                    wandb_tracker = self.accelerator.get_tracker("wandb", unwrap=True)

                    if self.accelerator.is_main_process:

                        logger.info(f"Logging model to W&B")
                        logged_artifact = wandb_tracker.log_artifact(model_artifact)
                        logged_artifact.wait()
                        logger.info(
                            f"Model artifact logged to Weights and Biases through Accelerate."
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to log artifact to Weights and Biases through Accelerate. Exception: {e}"
                    )

            # Remove the save_dir dir
            try:
                if os.path.exists(save_dir):
                    for root, dirs, files in os.walk(save_dir, topdown=False):
                        for file in files:
                            os.remove(os.path.join(root, file))
                        for dir in dirs:
                            os.rmdir(os.path.join(root, dir))
                    os.rmdir(save_dir)
                logger.info(f"Removed local files: {save_dir}")
            except Exception as e:
                logger.error(
                    f"Failed to remove local files: {save_dir}. Exception: {e}"
                )

    def run_training(self, epochs, alpha=0.5):
        tau = 1
        if self.training_management != "accelerate":
            self.model = self.model.to(self.device)

        global global_steps
        global_steps = 0

        early_stopping = {
            "best_accuracy": 0,
            "best_micro_f1": 0,
            "best_macro_f1": 0,
            "early_stop": 0,
            "early_stopped": False,
            "stopping_code": 0,
        }

        experiment_id = f"experiment_{int(time.time())}"

        for epoch in range(1, epochs + 1):
            try:
                tau, early_stopping = self._train(
                    epoch,
                    self.train_dataloader,
                    tau,
                    alpha,
                    experiment_id,
                    self.device,
                    early_stopping,
                )
            except Exception as e:
                logger.error(
                    f"Training failed at epoch {epoch} with exception: {e}",
                    exc_info=True,
                )
                self._log_alert(
                    title=f"Training failed at epoch {epoch}",
                    text=f"Exception: {str(e)}",
                )
                early_stopping["early_stopped"] = True
                early_stopping["stopping_code"] = 103
                break

            if early_stopping["early_stopped"]:
                early_stopping["stopping_code"] = 101
                self._log_alert(
                    title="Early stopping triggered.",
                    text="The model has been early stopped.",
                )
                break

            try:
                metrics = self._evaluate(
                    epoch, self.test_dataloader, self.device, tau, alpha, experiment_id
                )
            except Exception as e:
                logger.error(
                    f"Evaluation failed at epoch {epoch} with exception: {e}",
                    exc_info=True,
                )
                self._log_alert(
                    title=f"Evaluation failed at epoch {epoch}",
                    text=f"Exception: {str(e)}",
                )
                early_stopping["early_stopped"] = True
                early_stopping["stopping_code"] = 104
                break

            if epoch >= 1 and metrics["accuracy"] < 0.2:
                logger.info("Accuracy is below 0.2. Stopping training.")
                early_stopping["early_stopped"] = True
                early_stopping["stopping_code"] = 102
                self._log_alert(
                    title="Accuracy is below 0.2.",
                    text="The model never surpassed 0.2 accuracy.",
                )
                break

            if epoch >= 2 and metrics["accuracy"] < 0.3:
                logger.info("Accuracy is below 0.3. Stopping training.")
                early_stopping["early_stopped"] = True
                early_stopping["stopping_code"] = 102
                self._log_alert(
                    title="Accuracy is below 0.3.",
                    text="The model never surpassed 0.3 accuracy.",
                )
                break

        return early_stopping