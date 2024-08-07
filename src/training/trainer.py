from ast import Dict
import os
import time
from typing import Literal
import numpy as np
from sklearn.calibration import label_binarize
import torch
import json
from tqdm import tqdm
import math

from sklearn.metrics import accuracy_score, f1_score
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
        model_type: Literal["slmuse-dlf", "muse-dlf"] = "muse-dlf",  # muse or slmuse
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
        save_metric: Literal["accuracy", "f1_micro", "f1_macro"] = "accuracy",
        model_config={},
        class_column_names=[],
        save_model=True,
        _debug=False,
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

        self.class_column_names = class_column_names

        self.model_config = model_config

        self.save_model = save_model

        self._debug = _debug

        # Initialize the mixed precision
        logger.info(
            f"Mixed precision is enabled: {mixed_precision in ['fp16', 'bf16', 'fp32']}, therefore set mixed precision to: {mixed_precision}"
        )

        self.scaler = (
            GradScaler() if mixed_precision in ["fp16", "bf16", "fp32"] else None
        )

        self.training_management: Literal["accelerate", "wandb"] = training_management
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
        self.model_type: Literal["slmuse-dlf", "muse-dlf"] = model_type

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
            return torch.sigmoid(logits)
        elif self.model_type == "slmuse-dlf":
            return logits
        else:
            raise ValueError(
                f"Model type {self.model_type} not supported: only muse-dlf and slmuse-dlf are supported."
            )

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def _print(self, *args):
        if self.training_management == "accelerate":
            if self.accelerator.is_main_process:
                self.accelerator.print(*args)
        else:
            print(*args)

    def calculate_loss(self, outputs, labels, alpha):
        # Calculate losses
        unsupervised_loss = outputs["unsupervised_loss"]
        span_loss = self.loss_function(outputs["span_logits"], labels)
        sentence_loss = self.loss_function(outputs["sent_logits"], labels)

        # Additional losses for logging only
        with torch.no_grad():
            predicate_loss = self.loss_function(outputs["predicate_logits"], labels)
            arg0_loss = self.loss_function(outputs["arg0_logits"], labels)
            arg1_loss = self.loss_function(outputs["arg1_logits"], labels)
            frameaxis_loss = self.loss_function(outputs["frameaxis_logits"], labels)

        supervised_loss = span_loss + sentence_loss
        sum_of_parameters = sum(p.sum() for p in self.model.parameters())
        zero_sum = sum_of_parameters * 0.0
        combined_loss = (
            alpha * supervised_loss + (1 - alpha) * unsupervised_loss
        ) + zero_sum

        return combined_loss, {
            "combined_loss": combined_loss.detach(),
            "unsupervised_loss": unsupervised_loss.detach(),
            "span_loss": span_loss.detach(),
            "sentence_loss": sentence_loss.detach(),
            "supervised_loss": supervised_loss.detach(),
            "predicate_loss": predicate_loss.detach(),
            "arg0_loss": arg0_loss.detach(),
            "arg1_loss": arg1_loss.detach(),
            "frameaxis_loss": frameaxis_loss.detach(),
        }

    def _log_classification_report(self, logits, labels, prefix="train"):
        combined_pred_np = logits.cpu().numpy()
        combined_labels_np = labels.cpu().numpy()

        if self.model_type == "muse-dlf":
            # Convert continuous predictions to binary for muse-dlf
            threshold = 0.5  # You may need to adjust this threshold
            binary_predictions = (combined_pred_np > threshold).astype(int)

            y_true = combined_pred_np
            y_pred = binary_predictions
        else:
            # For other model types, assume the predictions are already in the correct format
            binary_predictions = combined_pred_np

            # Convert to one-hot encoding
            y_true = label_binarize(combined_labels_np, classes=all_classes)
            y_pred = label_binarize(combined_pred_np, classes=all_classes)

        all_classes = list(range(len(self.class_column_names)))

        # Generate classification report
        class_report = classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
            target_names=self.class_column_names,
            labels=all_classes,
        )

        if (
            self.training_management == "accelerate"
            and self.accelerator.is_main_process
        ) or (self.training_management != "accelerate"):
            # Print the classification report
            logger.info("\nPer-class metrics for training data:")
            logger.info(
                classification_report(
                    y_true,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                    target_names=self.class_column_names,
                    labels=all_classes,
                )
            )

        prefix = f"{prefix}_" if len(prefix) > 0 else ""

        # Log per-class metrics
        for class_name, metrics in class_report.items():
            if isinstance(
                metrics, dict
            ):  # Skip 'accuracy', 'macro avg', 'weighted avg'
                self._log_metrics(
                    {
                        f"{prefix}precision_class_{class_name}": metrics["precision"],
                        f"{prefix}recall_class_{class_name}": metrics["recall"],
                        f"{prefix}f1_class_{class_name}": metrics["f1-score"],
                    }
                )

    def _create_metrics_dict(self, vars_to_log, experiment_id):
        metrics_dict = {}
        metrics_dict["accuracy"] = {}
        metrics_dict["f1_micro"] = {}
        metrics_dict["f1_macro"] = {}

        if self.model_type == "slmuse-dlf":
            for metric in ["accuracy", "f1_micro", "f1_macro"]:
                name = metric if "_" not in metric else metric.split("_")[0]
                config = None if "_" not in metric else metric.split("_")[1]
                for var in vars_to_log:
                    metrics_dict[metric][var] = evaluate.load(
                        name, config_name=config, experiment_id=experiment_id
                    )
        elif self.model_type == "muse-dlf":
            for metric in ["accuracy", "f1_micro", "f1_macro"]:
                for var in vars_to_log:
                    metrics_dict[metric][var] = []

        return metrics_dict

    def _prepare_logits(self, outputs: Dict, labels: torch.Tensor, keys=[]):

        if self.model_type == "muse-dlf":
            labels = labels.detach().cpu()
        elif self.model_type == "slmuse-dlf":
            labels = labels.argmax(dim=1).long().detach().cpu()

        logits = {}
        for key in keys:
            preds = self.get_activation_function(outputs[key])

            if self.model_type == "muse-dlf":
                preds = preds.float().detach().cpu()
            elif self.model_type == "slmuse-dlf":
                preds = preds.argmax(dim=1).long().detach().cpu()

            logits[key] = (preds, labels)

        return logits

    def _metrics_add_batch(self, metrics, logits):
        if self.model_type == "slmuse-dlf":
            for metric in metrics.keys():
                for key, value in logits.items():
                    metrics_name = key.split("_")[0]
                    preds, labels = value
                    metrics[metric][metrics_name].add_batch(
                        predictions=preds, references=labels
                    )
        elif self.model_type == "muse-dlf":
            for metric in metrics.keys():
                for key, value in logits.items():
                    metrics_name = key.split("_")[0]
                    preds, labels = value
                    metrics[metric][metrics_name].append((preds, labels))

        return metrics

    def _metrics_calculate(self, metrics, prefix="train"):
        results = {}

        if self.model_type == "slmuse-dlf":
            for metric, value in metrics.items():
                for key, evaluator in value.items():
                    if metric == "accuracy":
                        result = evaluator.compute()["accuracy"]
                    elif metric == "f1_micro":
                        result = evaluator.compute(average="micro")["f1"]
                    elif metric == "f1_macro":
                        result = evaluator.compute(average="macro")["f1"]

                    prefix_name = f"{prefix}_{metric}" if len(prefix) > 0 else metric

                    if key == "supervised":
                        results[prefix_name] = result
                    else:
                        results[f"{prefix_name}_{key}"] = result

        elif self.model_type == "muse-dlf":
            for metric, value in metrics.items():
                for key, data_list in value.items():
                    all_preds = []
                    all_labels = []
                    for preds, labels in data_list:
                        all_preds.extend(preds.numpy())
                        all_labels.extend(labels.numpy())

                    all_preds = np.array(all_preds)
                    all_labels = np.array(all_labels)

                    # Check if we're dealing with multi-label data
                    is_multilabel = (
                        all_labels.ndim > 1 and all_labels.shape[1] > 1
                    ) or (all_labels.dtype == bool)

                    if is_multilabel:
                        # For multi-label, we need to binarize the predictions
                        threshold = 0.5  # You might want to adjust this
                        all_preds_binary = (all_preds > threshold).astype(int)
                        all_labels_binary = all_labels
                    else:
                        # For single-label, we take the argmax of predictions
                        all_preds_binary = np.argmax(all_preds, axis=1)
                        all_labels_binary = all_labels

                    if metric == "accuracy":
                        result = accuracy_score(all_labels_binary, all_preds_binary)
                    elif metric == "f1_micro":
                        result = f1_score(
                            all_labels_binary, all_preds_binary, average="micro"
                        )
                    elif metric == "f1_macro":
                        result = f1_score(
                            all_labels_binary, all_preds_binary, average="macro"
                        )

                    prefix_name = f"{prefix}_{metric}" if len(prefix) > 0 else metric

                    if key == "supervised":
                        results[prefix_name] = result
                    else:
                        results[f"{prefix_name}_{key}"] = result

        return results

    def _train(
        self,
        epoch,
        train_dataloader,
        tau,
        alpha,
        experiment_id,
        early_stopping={
            "best_accuracy": 0,
            "best_f1_micro": 0,
            "best_f1_macro": 0,
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
            if global_steps % self.test_every_n_batches == 0:
                tau = max(self.tau_min, math.exp(-self.tau_decay * global_steps))
                self.accelerator.wait_for_everyone()

            # Prepare inputs
            inputs = {
                k: v.to(self.accelerator.device)
                for k, v in batch.items()
                if k != "labels"
            }
            labels = batch["labels"].to(self.accelerator.device)

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

            if self.training_management == "accelerate":
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        # log for gpu num
                        logger.debug(
                            f"Process Index: {self.accelerator.process_index} - Start forward pass"
                        )
                        self.accelerator.wait_for_everyone()
                        # Forward pass
                        outputs = self.model(**model_inputs)

                        logger.debug(
                            f"Process Index: {self.accelerator.process_index} - End forward pass"
                        )

                        self.accelerator.wait_for_everyone()
                        combined_loss, loss_dict = self.calculate_loss(
                            outputs, prepared_labels, alpha
                        )

                        # print shape with process index
                        logger.info(
                            f"Process Index: {self.accelerator.process_index} - {combined_loss.item()}"
                        )

                        self.accelerator.wait_for_everyone()
                        logger.debug(
                            f"Process Index: {self.accelerator.process_index} - Loss calculated, started backwards pass"
                        )
                    # Backward pass
                    self.accelerator.backward(combined_loss)

                    logger.debug(
                        f"Process Index: {self.accelerator.process_index} - End backwards pass"
                    )

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.clip_value
                        )

                    logger.debug(
                        f"Process Index: {self.accelerator.process_index} - Start optimizer step"
                    )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    logger.debug(
                        f"Process Index: {self.accelerator.process_index} - End optimizer step"
                    )

                self.accelerator.wait_for_everyone()
            else:
                with autocast(
                    enabled=self.mixed_precision in ["fp16", "bf16", "fp32"],
                    dtype=precision_dtype,
                ):
                    outputs = self.model(**model_inputs)
                    combined_loss, loss_dict = self.calculate_loss(
                        outputs, prepared_labels, alpha
                    )

                    if self.scaler is not None:
                        self.scaler.scale(
                            combined_loss / self.accumulation_steps
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
                        (combined_loss / self.accumulation_steps).backward()
                        if (batch_idx + 1) % self.accumulation_steps == 0 or (
                            batch_idx + 1
                        ) == len(train_dataloader):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.clip_value
                            )
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()

            logger.debug(
                f"Process Index: {self.accelerator.process_index} - Update loss statistics"
            )

            self.accelerator.wait_for_everyone()

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

            logger.debug(
                f"Process Index: {self.accelerator.process_index} - Update metrics"
            )

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
                self.accelerator.wait_for_everyone()
                with torch.no_grad():
                    if self.training_management == "accelerate":
                        gathered_outputs = self.accelerator.gather(outputs)
                        gathered_labels = self.accelerator.gather(labels)

                    if self.accelerator.is_main_process:
                        logger.info(
                            f"[TRAIN] Starting to evaluate the model at epoch {epoch}, batch {global_steps}"
                        )

                        # Prepare logits
                        prepared_logits = self._prepare_logits(
                            gathered_outputs,
                            gathered_labels,
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
                            "supervised_logits"
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
                            early_stopping["best_f1_micro"] = metrics["train_f1_micro"]
                            early_stopping["best_f1_macro"] = metrics["train_f1_macro"]
                            early_stopping["early_stop"] = 0

                            if (
                                metrics[f"train_{self.save_metric}"]
                                > self.save_threshold
                            ):
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

                self.accelerator.wait_for_everyone()

            # Clean up
            del outputs, labels, prepared_labels, loss_dict
            torch.cuda.empty_cache()

        # End of epoch logging
        self.accelerator.wait_for_everyone()
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

    def _evaluate(self, epoch, test_dataloader, tau, experiment_id):
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
                    with self.accelerator.accumulate(self.model):
                        with self.accelerator.autocast():
                            outputs = self.model(**model_inputs)
                else:
                    with autocast(
                        enabled=self.mixed_precision in ["fp16", "bf16", "fp32"],
                        dtype=precision_dtype,
                    ):
                        outputs = self.model(**model_inputs)

                # Synchronize all processes here
                logger.debug("Waiting for all processes to synchronize.")
                self.accelerator.wait_for_everyone()

                if self.training_management == "accelerate":
                    gathered_outputs = self.accelerator.gather(outputs)
                    gathered_labels = self.accelerator.gather(labels)
                else:
                    gathered_outputs = outputs
                    gathered_labels = labels

                if self.accelerator.is_main_process:
                    logger.info(
                        f"[EVALUATE] Starting to evaluate the model at epoch {epoch}, batch {global_steps}"
                    )

                    # Prepare logits
                    prepared_logits = self._prepare_logits(
                        gathered_outputs,
                        gathered_labels,
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

                    del prepared_logits
                    torch.cuda.empty_cache()

                del labels, outputs, gathered_outputs, gathered_labels

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
            "best_f1_micro": 0,
            "best_f1_macro": 0,
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
                    epoch, self.test_dataloader, tau, experiment_id
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

            # wait
            self.accelerator.wait_for_everyone()

            if (
                self.training_management != "accelerate"
                or self.accelerator.is_main_process
            ):
                if epoch >= 1 and metrics[self.save_metric] < 0.2:
                    logger.info("Accuracy is below 0.2. Stopping training.")
                    early_stopping["early_stopped"] = True
                    early_stopping["stopping_code"] = 102
                    self._log_alert(
                        title="Accuracy is below 0.2.",
                        text="The model never surpassed 0.2 accuracy.",
                    )
                    break

                if epoch >= 2 and metrics[self.save_metric] < 0.3:
                    logger.info("Accuracy is below 0.3. Stopping training.")
                    early_stopping["early_stopped"] = True
                    early_stopping["stopping_code"] = 102
                    self._log_alert(
                        title="Accuracy is below 0.3.",
                        text="The model never surpassed 0.3 accuracy.",
                    )
                    break

            # wait
            self.accelerator.wait_for_everyone()

        return early_stopping
