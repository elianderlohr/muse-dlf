from ast import Dict
from logging import config
import os
import time
from typing import Literal
import torch
import json
from tqdm import tqdm
import math
import evaluate
from wandb import AlertLevel
from utils.logging_manager import LoggerManager
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report

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
        training_management=None,  # 'accelerate', 'wandb', or None
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

    def calculate_loss(self, outputs, labels, alpha):
        unsupervised_loss = outputs["unsupervised_loss"]
        span_loss = self.loss_function(
            outputs["span_logits"], labels, input_type="logits"
        )
        sentence_loss = self.loss_function(
            outputs["sent_logits"], labels, input_type="logits"
        )

        # predicate
        predicate_loss = self.loss_function(
            outputs["predicate_logits"], labels, input_type="logits"
        )
        arg0_loss = self.loss_function(
            outputs["arg0_logits"], labels, input_type="logits"
        )
        arg1_loss = self.loss_function(
            outputs["arg1_logits"], labels, input_type="logits"
        )
        frameaxis_loss = self.loss_function(
            outputs["frameaxis_logits"], labels, input_type="logits"
        )

        supervised_loss = span_loss + sentence_loss
        combined_loss = alpha * supervised_loss + (1 - alpha) * unsupervised_loss

        return combined_loss, {
            "unsupervised_loss": unsupervised_loss,
            "span_loss": span_loss,
            "sentence_loss": sentence_loss,
            "supervised_loss": supervised_loss,
            "combined_loss": combined_loss,
            "predicate_loss": predicate_loss,
            "arg0_loss": arg0_loss,
            "arg1_loss": arg1_loss,
            "frameaxis_loss": frameaxis_loss,
        }

    def _log_classification_report(self, logits, labels):
        combined_pred_np = logits.cpu().numpy()
        combined_labels_np = labels.cpu().numpy()

        # Generate classification report
        class_report = classification_report(
            combined_labels_np, combined_pred_np, output_dict=True
        )

        if (
            self.training_management == "accelerate"
            and self.accelerator.is_main_process
        ) or (self.training_management != "accelerate"):
            # Print the classification report
            logger.info("\nPer-class metrics for training data:")
            logger.info(classification_report(combined_labels_np, combined_pred_np))

        # Log per-class metrics
        for class_name, metrics in class_report.items():
            if isinstance(
                metrics, dict
            ):  # Skip 'accuracy', 'macro avg', 'weighted avg'
                self._log_metrics(
                    {
                        f"train_precision_class_{class_name}": metrics["precision"],
                        f"train_recall_class_{class_name}": metrics["recall"],
                        f"train_f1_class_{class_name}": metrics["f1-score"],
                    }
                )

    def _create_metrics_dict(self, vars_to_log, experiment_id):
        metrics_dict = {}

        metrics_dict["accuracy"] = {}
        metrics_dict["f1_micro"] = {}
        metrics_dict["f1_macro"] = {}

        for metric in ["accuracy", "f1_micro", "f1_macro"]:
            name = metric if "_" not in metric else metric.split("_")[0]
            config = None if "_" not in metric else metric.split("_")[1]
            for var in vars_to_log:
                metrics_dict[metric][var] = evaluate.load(
                    name, config_name=config, experiment_id=experiment_id
                )

        return metrics_dict

    def _prepare_logits(self, outputs: Dict, labels, keys=[]):
        logits = {}
        for key in keys:
            logger.info(f"1. _prepare_logits: {outputs[key].shape}")
            pred = self.get_activation_function(outputs[key])
            logger.info(f"3. pred: {pred.shape}")
            logger.info(f"3. labels: {labels.shape}")
            logits[key] = (pred, labels)

        return logits

    def _metrics_add_batch(self, metrics, logits):
        for metric in metrics.keys():
            for key, value in logits.items():
                metrics_name = key.split("_")[0]

                preds, labels = value

                # print shapes:

                logger.info(f"before: {metrics_name} - preds shape: {preds.shape}")
                logger.info(f"before: {metrics_name} - labels shape: {labels.shape}")

                if self.model_type == "muse-dlf":
                    preds = preds.float()
                    labels = labels.int()
                elif self.model_type == "slmuse-dlf":
                    preds = preds.argmax(dim=1)
                    labels = labels.argmax(dim=1)

                logger.info(f"after: {metrics_name} - preds shape: {preds.shape}")
                logger.info(f"after: {metrics_name} - labels shape: {labels.shape}")

                metrics[metric][metrics_name].add_batch(
                    predictions=preds, references=labels
                )

        return metrics

    def _metrics_calculate(self, metrics, prefix="train"):
        results = {}

        for metric, value in metrics.items():
            for key, evaluator in value.items():
                if metric == "accuracy":
                    result = evaluator.compute()
                elif metric == "f1_micro":
                    result = evaluator.compute(average="micro")
                elif metric == "f1_macro":
                    result = evaluator.compute(average="macro")

                if key == "supervised":
                    results[f"{prefix}_{metric}"] = result
                else:
                    results[f"{prefix}_{metric}_{key}"] = result

        return results

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

        metrics_dict = self._create_metrics_dict(
            [
                "supervised",
                "span",
                "sent",
                "predicate",
                "arg0",
                "arg1",
                "frameaxis",
            ],
            experiment_id,
        )

        precision_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16 if self.mixed_precision == "bf16" else torch.float32
        )

        for batch_idx, batch in enumerate(
            tqdm(
                train_dataloader,
                desc=f"Train - Epoch {epoch}",
                disable=not self.accelerator.is_main_process,
            )
        ):
            global global_steps
            global_steps += 1

            # Update tau every 50 steps
            if global_steps % 50 == 0:
                tau = max(self.tau_min, math.exp(-self.tau_decay * global_steps))

            # Prepare inputs
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]

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

            if self.model_type == "muse-dlf":
                prepared_labels = labels.float()
            elif self.model_type == "slmuse-dlf":
                if labels.dim() == 2:
                    logger.debug(
                        "Labels are one-hot encoded, converting to class index."
                    )
                    prepared_labels = torch.argmax(labels, dim=1).long()

            # Forward pass and loss calculation
            if self.training_management == "accelerate":
                with self.accelerator.autocast():
                    outputs = self.model(**model_inputs)
                    current_total_loss, loss_dict = self.calculate_loss(
                        outputs, prepared_labels, alpha
                    )
            else:
                with autocast(
                    enabled=self.mixed_precision in ["fp16", "bf16", "fp32"],
                    dtype=precision_dtype,
                ):
                    outputs = self.model(**model_inputs)
                    current_total_loss, loss_dict = self.calculate_loss(
                        outputs, prepared_labels, alpha
                    )

            # Backward pass
            if self.training_management == "accelerate":
                with self.accelerator.accumulate(self.model):
                    self.accelerator.backward(current_total_loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.clip_value
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
            else:
                if self.scaler is not None:
                    self.scaler.scale(
                        current_total_loss / self.accumulation_steps
                    ).backward()
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
                    (current_total_loss / self.accumulation_steps).backward()
                    if (batch_idx + 1) % self.accumulation_steps == 0 or (
                        batch_idx + 1
                    ) == len(train_dataloader):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_value
                        )
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

            # Update loss statistics
            if self.training_management == "accelerate":
                total_loss += (
                    self.accelerator.gather(loss_dict["combined_loss"]).sum().item()
                )
                supervised_total_loss += (
                    self.accelerator.gather(loss_dict["supervised_loss"]).sum().item()
                )
                unsupervised_total_loss += (
                    self.accelerator.gather(loss_dict["unsupervised_loss"]).sum().item()
                )
            else:
                total_loss += loss_dict["combined_loss"].item()
                supervised_total_loss += loss_dict["supervised_loss"].item()
                unsupervised_total_loss += loss_dict["unsupervised_loss"].item()

            # Log metrics
            if self.accelerator.is_main_process:
                current_lr_scheduler = self.scheduler.get_last_lr()[0]
                current_lr_model = self.get_lr()
                self._log_metrics(
                    {
                        "batch_combined_loss": loss_dict["combined_loss"].item(),
                        "batch_supervised_loss": loss_dict["supervised_loss"].item(),
                        "batch_span_loss": loss_dict["span_loss"].item(),
                        "batch_sentence_loss": loss_dict["sentence_loss"].item(),
                        "batch_unsupervised_loss": loss_dict[
                            "unsupervised_loss"
                        ].item(),
                        "batch_predicate_loss": loss_dict["predicate_loss"].item(),
                        "batch_arg0_loss": loss_dict["arg0_loss"].item(),
                        "batch_arg1_loss": loss_dict["arg1_loss"].item(),
                        "batch_frameaxis_loss": loss_dict["frameaxis_loss"].item(),
                        "batch": batch_idx,
                        "global_steps": global_steps,
                        "tau": tau,
                        "epoch": epoch,
                        "learning_rate_scheduler": current_lr_scheduler,
                        "learning_rate_model": current_lr_model,
                    }
                )

            # Evaluation and early stopping check
            if global_steps % self.test_every_n_batches == 0:
                with torch.no_grad():
                    if self.training_management == "accelerate":
                        outputs = self.accelerator.gather(outputs)
                        labels = self.accelerator.gather(labels)

                    if self.accelerator.is_main_process:
                        logger.info(
                            f"[TRAIN] Starting to evaluate the model at epoch {epoch}, batch {global_steps}"
                        )

                        prepared_logits = self._prepare_logits(
                            outputs,
                            labels,
                            keys=[
                                "span_logits",
                                "sent_logits",
                                "supervised_logits",
                                "predicate_logits",
                                "arg0_logits",
                                "arg1_logits",
                                "frameaxis_logits",
                            ],
                        )

                        metrics_dict = self._metrics_add_batch(
                            metrics_dict, prepared_logits
                        )
                        metrics = self._metrics_calculate(metrics_dict, prefix="train")
                        self._log_metrics(metrics)

                        supervised_pred, supervised_labels = prepared_logits[
                            "supervised"
                        ]
                        self._log_classification_report(
                            supervised_pred, supervised_labels
                        )

                        logger.info(
                            f"[TRAIN] Epoch {epoch}, Step {global_steps}: Micro F1: {metrics['train_f1_micro']}, "
                            f"Macro F1: {metrics['train_f1_macro']}, Accuracy: {metrics['train_accuracy']}"
                        )

                        if (
                            metrics[f"train_{self.save_metric}"]
                            >= early_stopping[f"best_{self.save_metric}"]
                        ):
                            early_stopping["best_accuracy"] = metrics["train_accuracy"]
                            early_stopping["best_micro_f1"] = metrics["train_f1_micro"]
                            early_stopping["best_macro_f1"] = metrics["train_f1_macro"]
                            early_stopping["early_stop"] = 0

                            if metrics[self.save_metric] > self.save_threshold:
                                self._save_model()
                        else:
                            early_stopping["early_stop"] += 1

                            if (
                                early_stopping["early_stop"]
                                >= self.early_stopping_patience
                            ):
                                logger.info("Early stopping triggered.")
                                early_stopping["early_stopped"] = True

                        metrics_dict = self._create_metrics_dict(
                            [
                                "supervised",
                                "span",
                                "sent",
                                "predicate",
                                "arg0",
                                "arg1",
                                "frameaxis",
                            ],
                            experiment_id,
                        )

                if early_stopping["early_stopped"]:
                    return tau, early_stopping

            # Clean up
            del outputs, labels, prepared_labels, loss_dict
            torch.cuda.empty_cache()

        # End of epoch logging
        if self.accelerator.is_main_process:
            avg_total_loss = total_loss / len(train_dataloader)
            avg_supervised_loss = supervised_total_loss / len(train_dataloader)
            avg_unsupervised_loss = unsupervised_total_loss / len(train_dataloader)

            logger.info(
                f"[TRAIN] Epoch {epoch}, Step {global_steps}: Avg Total Loss: {avg_total_loss}, "
                f"Avg Supervised Loss: {avg_supervised_loss}, Avg Unsupervised Loss: {avg_unsupervised_loss}"
            )

            self._log_metrics(
                {
                    "epoch_combined_loss": avg_total_loss,
                    "epoch_supervised_loss": avg_supervised_loss,
                    "epoch_unsupervised_loss": avg_unsupervised_loss,
                    "epoch": epoch,
                }
            )

            self._save_model()

        return tau, early_stopping

    def _evaluate(self, epoch, test_dataloader, device, tau, alpha, experiment_id):
        self.model.eval()

        metrics_dict = self._create_metrics_dict(
            [
                "supervised",
                "span",
                "sent",
                "predicate",
                "arg0",
                "arg1",
                "frameaxis",
            ],
            experiment_id,
        )

        precision_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16 if self.mixed_precision == "bf16" else torch.float32
        )

        all_supervised_preds = []
        all_supervised_labels = []

        for batch_idx, batch in enumerate(
            tqdm(
                test_dataloader,
                desc=f"Evaluate - Epoch {epoch}",
                disable=not self.accelerator.is_main_process,
            )
        ):
            # Prepare inputs
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]

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

            with torch.no_grad():
                if self.training_management == "accelerate":
                    with self.accelerator.autocast():
                        outputs = self.model(**model_inputs)
                else:
                    with autocast(
                        enabled=self.mixed_precision in ["fp16", "bf16", "fp32"],
                        dtype=precision_dtype,
                    ):
                        outputs = self.model(**model_inputs)

                prepared_logits = self._prepare_logits(
                    outputs,
                    labels,
                    keys=[
                        "span_logits",
                        "sent_logits",
                        "supervised_logits",
                        "predicate_logits",
                        "arg0_logits",
                        "arg1_logits",
                        "frameaxis_logits",
                    ],
                )

                # Gather predictions and labels from all processes
                if self.training_management == "accelerate":
                    prepared_logits = self.accelerator.gather(prepared_logits)
                    labels = self.accelerator.gather(labels)

                # Process metrics only on the main process
                if (
                    self.training_management != "accelerate"
                    or self.accelerator.is_main_process
                ):
                    all_supervised_preds.append(
                        prepared_logits["supervised_logits"][0]
                    )  # predictions
                    all_supervised_labels.append(
                        prepared_logits["supervised_logits"][1]
                    )  # labels

                    # Add batch to metrics
                    metrics_dict = self._metrics_add_batch(
                        metrics_dict, prepared_logits
                    )

                del labels, prepared_logits, outputs
                torch.cuda.empty_cache()

        # After the loop, process the accumulated data on the main process
        if self.training_management != "accelerate" or self.accelerator.is_main_process:
            all_supervised_preds = torch.cat(all_supervised_preds, dim=0)
            all_supervised_labels = torch.cat(all_supervised_labels, dim=0)

            # Calculate final metrics
            metrics = self._metrics_calculate(metrics_dict, prefix="")
            self._log_metrics(metrics)

            # Add per-class evaluation
            self._log_classification_report(all_supervised_preds, all_supervised_labels)

            logger.info(
                f"[EVALUATE] Epoch {epoch}: Micro F1: {metrics['f1_micro']}, Macro F1: {metrics['f1_macro']}, Accuracy: {metrics['accuracy']}"
            )
        else:
            metrics = None

        return metrics

    def _save_model(self):
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
