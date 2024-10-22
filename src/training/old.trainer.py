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

        # Load the evaluate metrics
        f1_metric_micro = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric = evaluate.load("accuracy", experiment_id=experiment_id)

        # metrics for span
        f1_metric_micro_span = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_span = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_span = evaluate.load("accuracy", experiment_id=experiment_id)

        # metrics for sentence
        f1_metric_micro_sentence = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_sentence = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_sentence = evaluate.load(
            "accuracy", experiment_id=experiment_id
        )

        # predicate
        f1_metric_micro_predicate = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_predicate = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_predicate = evaluate.load(
            "accuracy", experiment_id=experiment_id
        )

        # arg0
        f1_metric_micro_arg0 = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_arg0 = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_arg0 = evaluate.load("accuracy", experiment_id=experiment_id)

        # arg1
        f1_metric_micro_arg1 = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_arg1 = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_arg1 = evaluate.load("accuracy", experiment_id=experiment_id)

        # frameaxis
        f1_metric_micro_frameaxis = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_frameaxis = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_frameaxis = evaluate.load(
            "accuracy", experiment_id=experiment_id
        )

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

            sentence_ids = (
                batch["sentence_ids"]
                if self.training_management == "accelerate"
                else batch["sentence_ids"].to(device)
            )
            sentence_attention_masks = (
                batch["sentence_attention_masks"]
                if self.training_management == "accelerate"
                else batch["sentence_attention_masks"].to(device)
            )

            predicate_ids = (
                batch["predicate_ids"]
                if self.training_management == "accelerate"
                else batch["predicate_ids"].to(device)
            )

            arg0_ids = (
                batch["arg0_ids"]
                if self.training_management == "accelerate"
                else batch["arg0_ids"].to(device)
            )

            arg1_ids = (
                batch["arg1_ids"]
                if self.training_management == "accelerate"
                else batch["arg1_ids"].to(device)
            )
            frameaxis_data = (
                batch["frameaxis"]
                if self.training_management == "accelerate"
                else batch["frameaxis"].to(device)
            )

            labels = (
                batch["labels"]
                if self.training_management == "accelerate"
                else batch["labels"].to(device)
            )

            if self.model_type == "muse-dlf":
                arg_max_labels = labels.float()
            elif self.model_type == "slmuse-dlf":
                if labels.dim() == 2:
                    logger.debug(
                        "Labels are one-hot encoded, converting to class index."
                    )
                    arg_max_labels = torch.argmax(labels, dim=1).long()

            if self.training_management == "accelerate":
                with self.accelerator.autocast():
                    (
                        unsupervised_loss,
                        span_probs,
                        sent_probs,
                        combined_probs,
                        other,
                    ) = self.model(
                        sentence_ids,
                        sentence_attention_masks,
                        predicate_ids,
                        arg0_ids,
                        arg1_ids,
                        frameaxis_data,
                        tau,
                    )

                    span_loss = self.loss_function(
                        span_probs, arg_max_labels, input_type="logits"
                    )
                    sentence_loss = self.loss_function(
                        sent_probs, arg_max_labels, input_type="logits"
                    )
                    supervised_loss = span_loss + sentence_loss

                    # supervised_loss = self.loss_function(
                    #    combined_probs, arg_max_labels, input_type="logits"
                    # )

                    sum_of_parameters = sum(p.sum() for p in self.model.parameters())
                    zero_sum = sum_of_parameters * 0.0
                    combined_loss = (
                        alpha * supervised_loss + (1 - alpha) * unsupervised_loss
                    ) + zero_sum
            else:
                with autocast(
                    enabled=self.mixed_precision in ["fp16", "bf16", "fp32"],
                    dtype=precision_dtype,
                ):
                    (
                        unsupervised_loss,
                        span_probs,
                        sent_probs,
                        combined_probs,
                        other,
                    ) = self.model(
                        sentence_ids,
                        sentence_attention_masks,
                        predicate_ids,
                        arg0_ids,
                        arg1_ids,
                        frameaxis_data,
                        tau,
                    )

                    span_loss = self.loss_function(
                        span_probs, arg_max_labels, input_type="logits"
                    )
                    sentence_loss = self.loss_function(
                        sent_probs, arg_max_labels, input_type="logits"
                    )
                    supervised_loss = span_loss + sentence_loss

                    sum_of_parameters = sum(p.sum() for p in self.model.parameters())
                    zero_sum = sum_of_parameters * 0.0
                    combined_loss = (
                        alpha * supervised_loss + (1 - alpha) * unsupervised_loss
                    ) + zero_sum

            # Delete tensors immediately after use
            del (
                sentence_ids,
                sentence_attention_masks,
                predicate_ids,
                arg0_ids,
                arg1_ids,
                frameaxis_data,
            )
            torch.cuda.empty_cache()

            # Check for NaNs in model outputs
            if (
                self.check_for_nans(unsupervised_loss, "unsupervised_loss")
                or self.check_for_nans(span_probs, "span_probs")
                or self.check_for_nans(sent_probs, "sent_probs")
                or self.check_for_nans(combined_probs, "combined_probs")
                or self.check_for_nans(other["predicate"], "other['predicate']")
                or self.check_for_nans(other["arg0"], "other['arg0']")
                or self.check_for_nans(other["arg1"], "other['arg1']")
                or self.check_for_nans(other["frameaxis"], "other['frameaxis']")
            ):
                logger.error(
                    f"{experiment_id} - {batch_idx} - NaNs detected in model outputs, skipping this batch."
                )
                continue

            with torch.no_grad():
                predicate_loss = self.loss_function(
                    other["predicate"], arg_max_labels, input_type="logits"
                )
                arg0_loss = self.loss_function(
                    other["arg0"], arg_max_labels, input_type="logits"
                )
                arg1_loss = self.loss_function(
                    other["arg1"], arg_max_labels, input_type="logits"
                )
                frameaxis_loss = self.loss_function(
                    other["frameaxis"], arg_max_labels, input_type="logits"
                )

            if self.check_for_nans(combined_loss, "combined_loss"):
                logger.error(
                    f"{experiment_id} - {batch_idx} - NaNs detected in combined_loss, skipping this batch."
                )
                continue

            if self.training_management == "accelerate":
                with self.accelerator.accumulate(self.model):
                    self.accelerator.backward(combined_loss)
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

            total_loss += combined_loss.item()
            supervised_total_loss += supervised_loss.item()
            unsupervised_total_loss += unsupervised_loss.item()

            current_lr_scheduler = self.scheduler.get_last_lr()[0]
            current_lr_model = self.get_lr()
            self._log_metrics(
                {
                    "batch_combined_loss": combined_loss.item(),
                    "batch_supervised_loss": supervised_loss.item(),
                    "batch_span_loss": span_loss.item(),
                    "batch_sentence_loss": sentence_loss.item(),
                    "batch_unsupervised_loss": unsupervised_loss.item(),
                    "batch_predicate_loss": predicate_loss.item(),
                    "batch_arg0_loss": arg0_loss.item(),
                    "batch_arg1_loss": arg1_loss.item(),
                    "batch_frameaxis_loss": frameaxis_loss.item(),
                    "batch": batch_idx,
                    "global_steps": global_steps,
                    "tau": tau,
                    "epoch": epoch,
                    "learning_rate_scheduler": current_lr_scheduler,
                    "learning_rate_model": current_lr_model,
                }
            )

            combined_pred = None
            span_pred = None
            sentence_pred = None
            predicate_pred = None
            arg0_pred = None
            arg1_pred = None
            frameaxis_pred = None
            combined_labels = None
            span_labels = None
            sentence_labels = None
            predicate_labels = None
            arg0_labels = None
            arg1_labels = None
            frameaxis_labels = None

            try:
                with torch.no_grad():
                    # Check train metrics every 50 steps
                    if global_steps % self.test_every_n_batches == 0:
                        logger.info(
                            f"[TRAIN] Starting to evaluate the model at epoch {epoch}, batch {global_steps}"
                        )

                        combined_pred = self.get_activation_function(combined_probs)
                        span_pred = self.get_activation_function(span_probs)
                        sentence_pred = self.get_activation_function(sent_probs)

                        # predicate, arg0, arg1, frameaxis
                        predicate_pred = self.get_activation_function(
                            other["predicate"]
                        )
                        arg0_pred = self.get_activation_function(other["arg0"])
                        arg1_pred = self.get_activation_function(other["arg1"])
                        frameaxis_pred = self.get_activation_function(
                            other["frameaxis"]
                        )

                        if self.training_management == "accelerate":
                            combined_pred, combined_labels = (
                                self.accelerator.gather_for_metrics(
                                    (combined_pred, labels)
                                )
                            )
                            span_pred, span_labels = (
                                self.accelerator.gather_for_metrics((span_pred, labels))
                            )
                            sentence_pred, sentence_labels = (
                                self.accelerator.gather_for_metrics(
                                    (sentence_pred, labels)
                                )
                            )
                            predicate_pred, predicate_labels = (
                                self.accelerator.gather_for_metrics(
                                    (predicate_pred, labels)
                                )
                            )
                            arg0_pred, arg0_labels = (
                                self.accelerator.gather_for_metrics((arg0_pred, labels))
                            )
                            arg1_pred, arg1_labels = (
                                self.accelerator.gather_for_metrics((arg1_pred, labels))
                            )
                            frameaxis_pred, frameaxis_labels = (
                                self.accelerator.gather_for_metrics(
                                    (frameaxis_pred, labels)
                                )
                            )
                        else:
                            combined_labels = labels
                            span_labels = labels
                            sentence_labels = labels
                            predicate_labels = labels
                            arg0_labels = labels
                            arg1_labels = labels
                            frameaxis_labels = labels

                        # Add per-class evaluation
                        combined_pred_np = combined_pred.cpu().numpy()
                        combined_labels_np = combined_labels.cpu().numpy()

                        # Transform from one-hot to class index
                        combined_pred = combined_pred.argmax(dim=1)
                        span_pred = span_pred.argmax(dim=1)
                        sentence_pred = sentence_pred.argmax(dim=1)
                        predicate_pred = predicate_pred.argmax(dim=1)
                        arg0_pred = arg0_pred.argmax(dim=1)
                        arg1_pred = arg1_pred.argmax(dim=1)
                        frameaxis_pred = frameaxis_pred.argmax(dim=1)
                        combined_labels = combined_labels.argmax(dim=1)
                        span_labels = span_labels.argmax(dim=1)
                        sentence_labels = sentence_labels.argmax(dim=1)
                        predicate_labels = predicate_labels.argmax(dim=1)
                        arg0_labels = arg0_labels.argmax(dim=1)
                        arg1_labels = arg1_labels.argmax(dim=1)
                        frameaxis_labels = frameaxis_labels.argmax(dim=1)

                        # Macro F1
                        f1_metric_macro.add_batch(
                            predictions=combined_pred.cpu().numpy(),
                            references=combined_labels.cpu().numpy(),
                        )
                        f1_metric_macro_span.add_batch(
                            predictions=span_pred.cpu().numpy(),
                            references=span_labels.cpu().numpy(),
                        )
                        f1_metric_macro_sentence.add_batch(
                            predictions=sentence_pred.cpu().numpy(),
                            references=sentence_labels.cpu().numpy(),
                        )
                        f1_metric_macro_predicate.add_batch(
                            predictions=predicate_pred.cpu().numpy(),
                            references=predicate_labels.cpu().numpy(),
                        )
                        f1_metric_macro_arg0.add_batch(
                            predictions=arg0_pred.cpu().numpy(),
                            references=arg0_labels.cpu().numpy(),
                        )
                        f1_metric_macro_arg1.add_batch(
                            predictions=arg1_pred.cpu().numpy(),
                            references=arg1_labels.cpu().numpy(),
                        )
                        f1_metric_macro_frameaxis.add_batch(
                            predictions=frameaxis_pred.cpu().numpy(),
                            references=frameaxis_labels.cpu().numpy(),
                        )

                        # Micro F1
                        f1_metric_micro.add_batch(
                            predictions=combined_pred.cpu().numpy(),
                            references=combined_labels.cpu().numpy(),
                        )
                        f1_metric_micro_span.add_batch(
                            predictions=span_pred.cpu().numpy(),
                            references=span_labels.cpu().numpy(),
                        )
                        f1_metric_micro_sentence.add_batch(
                            predictions=sentence_pred.cpu().numpy(),
                            references=sentence_labels.cpu().numpy(),
                        )
                        f1_metric_micro_predicate.add_batch(
                            predictions=predicate_pred.cpu().numpy(),
                            references=predicate_labels.cpu().numpy(),
                        )
                        f1_metric_micro_arg0.add_batch(
                            predictions=arg0_pred.cpu().numpy(),
                            references=arg0_labels.cpu().numpy(),
                        )
                        f1_metric_micro_arg1.add_batch(
                            predictions=arg1_pred.cpu().numpy(),
                            references=arg1_labels.cpu().numpy(),
                        )
                        f1_metric_micro_frameaxis.add_batch(
                            predictions=frameaxis_pred.cpu().numpy(),
                            references=frameaxis_labels.cpu().numpy(),
                        )

                        # Accuracy
                        accuracy_metric.add_batch(
                            predictions=combined_pred.cpu().numpy(),
                            references=combined_labels.cpu().numpy(),
                        )
                        accuracy_metric_span.add_batch(
                            predictions=span_pred.cpu().numpy(),
                            references=span_labels.cpu().numpy(),
                        )
                        accuracy_metric_sentence.add_batch(
                            predictions=sentence_pred.cpu().numpy(),
                            references=sentence_labels.cpu().numpy(),
                        )
                        accuracy_metric_predicate.add_batch(
                            predictions=predicate_pred.cpu().numpy(),
                            references=predicate_labels.cpu().numpy(),
                        )
                        accuracy_metric_arg0.add_batch(
                            predictions=arg0_pred.cpu().numpy(),
                            references=arg0_labels.cpu().numpy(),
                        )
                        accuracy_metric_arg1.add_batch(
                            predictions=arg1_pred.cpu().numpy(),
                            references=arg1_labels.cpu().numpy(),
                        )
                        accuracy_metric_frameaxis.add_batch(
                            predictions=frameaxis_pred.cpu().numpy(),
                            references=frameaxis_labels.cpu().numpy(),
                        )

                        eval_results_micro = f1_metric_micro.compute(average="micro")
                        eval_results_micro_span = f1_metric_micro_span.compute(
                            average="micro"
                        )
                        eval_results_micro_sentence = f1_metric_micro_sentence.compute(
                            average="micro"
                        )
                        eval_results_micro_predicate = (
                            f1_metric_micro_predicate.compute(average="micro")
                        )
                        eval_results_micro_arg0 = f1_metric_micro_arg0.compute(
                            average="micro"
                        )
                        eval_results_micro_arg1 = f1_metric_micro_arg1.compute(
                            average="micro"
                        )
                        eval_results_micro_frameaxis = (
                            f1_metric_micro_frameaxis.compute(average="micro")
                        )

                        eval_results_macro = f1_metric_macro.compute(average="macro")
                        eval_results_macro_span = f1_metric_macro_span.compute(
                            average="macro"
                        )
                        eval_results_macro_sentence = f1_metric_macro_sentence.compute(
                            average="macro"
                        )
                        eval_results_macro_predicate = (
                            f1_metric_macro_predicate.compute(average="macro")
                        )
                        eval_results_macro_arg0 = f1_metric_macro_arg0.compute(
                            average="macro"
                        )
                        eval_results_macro_arg1 = f1_metric_macro_arg1.compute(
                            average="macro"
                        )
                        eval_results_macro_frameaxis = (
                            f1_metric_macro_frameaxis.compute(average="macro")
                        )

                        eval_accuracy = accuracy_metric.compute()
                        eval_accuracy_span = accuracy_metric_span.compute()
                        eval_accuracy_sentence = accuracy_metric_sentence.compute()
                        eval_accuracy_predicate = accuracy_metric_predicate.compute()
                        eval_accuracy_arg0 = accuracy_metric_arg0.compute()
                        eval_accuracy_arg1 = accuracy_metric_arg1.compute()
                        eval_accuracy_frameaxis = accuracy_metric_frameaxis.compute()

                        logger.info(
                            f"[TRAIN] Epoch {epoch}, Step {global_steps}: Micro F1: {eval_results_micro}, Macro F1: {eval_results_macro}, Accuracy: {eval_accuracy}"
                        )
                        metrics = {
                            "train_micro_f1": eval_results_micro["f1"],
                            "train_micro_f1_span": eval_results_micro_span["f1"],
                            "train_micro_f1_sentence": eval_results_micro_sentence[
                                "f1"
                            ],
                            "train_macro_f1": eval_results_macro["f1"],
                            "train_macro_f1_span": eval_results_macro_span["f1"],
                            "train_macro_f1_sentence": eval_results_macro_sentence[
                                "f1"
                            ],
                            "train_accuracy": eval_accuracy["accuracy"],
                            "train_accuracy_span": eval_accuracy_span["accuracy"],
                            "train_accuracy_sentence": eval_accuracy_sentence[
                                "accuracy"
                            ],
                            "train_micro_f1_predicate": eval_results_micro_predicate[
                                "f1"
                            ],
                            "train_micro_f1_arg0": eval_results_micro_arg0["f1"],
                            "train_micro_f1_arg1": eval_results_micro_arg1["f1"],
                            "train_micro_f1_frameaxis": eval_results_micro_frameaxis[
                                "f1"
                            ],
                            "train_macro_f1_predicate": eval_results_macro_predicate[
                                "f1"
                            ],
                            "train_macro_f1_arg0": eval_results_macro_arg0["f1"],
                            "train_macro_f1_arg1": eval_results_macro_arg1["f1"],
                            "train_macro_f1_frameaxis": eval_results_macro_frameaxis[
                                "f1"
                            ],
                            "train_accuracy_predicate": eval_accuracy_predicate[
                                "accuracy"
                            ],
                            "train_accuracy_arg0": eval_accuracy_arg0["accuracy"],
                            "train_accuracy_arg1": eval_accuracy_arg1["accuracy"],
                            "train_accuracy_frameaxis": eval_accuracy_frameaxis[
                                "accuracy"
                            ],
                        }

                        self._log_metrics(metrics)

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
                            logger.info(
                                classification_report(
                                    combined_labels_np, combined_pred_np
                                )
                            )

                        # Log per-class metrics
                        for class_name, metrics in class_report.items():
                            if isinstance(
                                metrics, dict
                            ):  # Skip 'accuracy', 'macro avg', 'weighted avg'
                                self._log_metrics(
                                    {
                                        f"train_precision_class_{class_name}": metrics[
                                            "precision"
                                        ],
                                        f"train_recall_class_{class_name}": metrics[
                                            "recall"
                                        ],
                                        f"train_f1_class_{class_name}": metrics[
                                            "f1-score"
                                        ],
                                    }
                                )

                        if (
                            eval_accuracy[self.save_metric]
                            >= early_stopping[f"best_{self.save_metric}"]
                        ):
                            early_stopping["best_accuracy"] = eval_accuracy["accuracy"]
                            early_stopping["best_micro_f1"] = eval_results_micro["f1"]
                            early_stopping["best_macro_f1"] = eval_results_macro["f1"]
                            early_stopping["early_stop"] = 0

                            if eval_accuracy[self.save_metric] > self.save_threshold:
                                self._save_model(f"step_{global_steps}")
                        else:
                            early_stopping["early_stop"] += 1

                            if (
                                early_stopping["early_stop"]
                                >= self.early_stopping_patience
                            ):
                                logger.info("Early stopping triggered.")
                                early_stopping["early_stopped"] = True
                                return tau, early_stopping

            except Exception as e:
                logger.error(f"Error during metric logging at step {global_steps}: {e}")
                continue

            # Delete tensors after logging metrics
            del (
                labels,
                combined_pred,
                span_pred,
                sentence_pred,
                predicate_pred,
                arg0_pred,
                arg1_pred,
                frameaxis_pred,
            )
            del (
                combined_labels,
                span_labels,
                sentence_labels,
                predicate_labels,
                arg0_labels,
                arg1_labels,
                frameaxis_labels,
            )
            torch.cuda.empty_cache()

        avg_total_loss = total_loss / len(train_dataloader)
        avg_supervised_loss = supervised_total_loss / len(train_dataloader)
        avg_unsupervised_loss = unsupervised_total_loss / len(train_dataloader)

        logger.info(
            f"[TRAIN] Epoch {epoch}, Step {global_steps}: Avg Total Loss: {avg_total_loss}, Avg Supervised Loss: {avg_supervised_loss}, Avg Unsupervised Loss: {avg_unsupervised_loss}"
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

        # Load the evaluate metrics
        f1_metric_micro = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric = evaluate.load("accuracy", experiment_id=experiment_id)

        # Metrics for span
        f1_metric_micro_span = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_span = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_span = evaluate.load("accuracy", experiment_id=experiment_id)

        # Metrics for sentence
        f1_metric_micro_sentence = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_sentence = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_sentence = evaluate.load(
            "accuracy", experiment_id=experiment_id
        )

        # Predicate
        f1_metric_micro_predicate = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_predicate = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_predicate = evaluate.load(
            "accuracy", experiment_id=experiment_id
        )

        # Arg0
        f1_metric_micro_arg0 = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_arg0 = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_arg0 = evaluate.load("accuracy", experiment_id=experiment_id)

        # Arg1
        f1_metric_micro_arg1 = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_arg1 = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_arg1 = evaluate.load("accuracy", experiment_id=experiment_id)

        # Frameaxis
        f1_metric_micro_frameaxis = evaluate.load(
            "f1", config_name="micro", experiment_id=experiment_id
        )
        f1_metric_macro_frameaxis = evaluate.load(
            "f1", config_name="macro", experiment_id=experiment_id
        )
        accuracy_metric_frameaxis = evaluate.load(
            "accuracy", experiment_id=experiment_id
        )

        precision_dtype = (
            torch.float16
            if self.mixed_precision == "fp16"
            else torch.bfloat16 if self.mixed_precision == "bf16" else torch.float32
        )

        all_combined_preds = []
        all_combined_labels = []

        for batch_idx, batch in enumerate(
            tqdm(test_dataloader, desc=f"Evaluate - Epoch {epoch}")
        ):
            sentence_ids = (
                batch["sentence_ids"]
                if self.training_management == "accelerate"
                else batch["sentence_ids"].to(device)
            )
            sentence_attention_masks = (
                batch["sentence_attention_masks"]
                if self.training_management == "accelerate"
                else batch["sentence_attention_masks"].to(device)
            )

            predicate_ids = (
                batch["predicate_ids"]
                if self.training_management == "accelerate"
                else batch["predicate_ids"].to(device)
            )

            arg0_ids = (
                batch["arg0_ids"]
                if self.training_management == "accelerate"
                else batch["arg0_ids"].to(device)
            )

            arg1_ids = (
                batch["arg1_ids"]
                if self.training_management == "accelerate"
                else batch["arg1_ids"].to(device)
            )

            frameaxis_data = (
                batch["frameaxis"]
                if self.training_management == "accelerate"
                else batch["frameaxis"].to(device)
            )

            labels = (
                batch["labels"]
                if self.training_management == "accelerate"
                else batch["labels"].to(device)
            )

            if self.model_type == "muse-dlf":
                arg_max_labels = labels.float()
            elif self.model_type == "slmuse-dlf":
                if labels.dim() == 2:
                    logger.debug(
                        "Labels are one-hot encoded, converting to class index."
                    )
                    arg_max_labels = torch.argmax(labels, dim=1).long()

            with torch.no_grad():

                if self.training_management == "accelerate":
                    with self.accelerator.autocast():
                        (
                            unsupervised_loss,
                            span_probs,
                            sent_probs,
                            combined_probs,
                            other,
                        ) = self.model(
                            sentence_ids,
                            sentence_attention_masks,
                            predicate_ids,
                            arg0_ids,
                            arg1_ids,
                            frameaxis_data,
                            tau,
                        )

                else:
                    with autocast(
                        enabled=self.mixed_precision in ["fp16", "bf16", "fp32"],
                        dtype=precision_dtype,
                    ):
                        (
                            unsupervised_loss,
                            span_probs,
                            sent_probs,
                            combined_probs,
                            other,
                        ) = self.model(
                            sentence_ids,
                            sentence_attention_masks,
                            predicate_ids,
                            arg0_ids,
                            arg1_ids,
                            frameaxis_data,
                            tau,
                        )

                span_loss = self.loss_function(
                    span_probs, arg_max_labels, input_type="logits"
                )
                sentence_loss = self.loss_function(
                    sent_probs, arg_max_labels, input_type="logits"
                )
                # supervised_loss = self.loss_function(
                #    combined_probs, arg_max_labels, input_type="logits"
                # )

                supervised_loss = span_loss + sentence_loss

                combined_loss = (
                    alpha * supervised_loss + (1 - alpha) * unsupervised_loss
                )

                total_val_loss += combined_loss.item()

                combined_pred = self.get_activation_function(combined_probs)
                span_pred = self.get_activation_function(span_probs)
                sentence_pred = self.get_activation_function(sent_probs)

                # predicate, arg0, arg1, frameaxis
                predicate_pred = self.get_activation_function(other["predicate"])
                arg0_pred = self.get_activation_function(other["arg0"])
                arg1_pred = self.get_activation_function(other["arg1"])
                frameaxis_pred = self.get_activation_function(other["frameaxis"])

                if self.training_management == "accelerate":
                    combined_pred, combined_labels = (
                        self.accelerator.gather_for_metrics((combined_pred, labels))
                    )
                    span_pred, span_labels = self.accelerator.gather_for_metrics(
                        (span_pred, labels)
                    )
                    sentence_pred, sentence_labels = (
                        self.accelerator.gather_for_metrics((sentence_pred, labels))
                    )
                    predicate_pred, predicate_labels = (
                        self.accelerator.gather_for_metrics((predicate_pred, labels))
                    )
                    arg0_pred, arg0_labels = self.accelerator.gather_for_metrics(
                        (arg0_pred, labels)
                    )
                    arg1_pred, arg1_labels = self.accelerator.gather_for_metrics(
                        (arg1_pred, labels)
                    )
                    frameaxis_pred, frameaxis_labels = (
                        self.accelerator.gather_for_metrics((frameaxis_pred, labels))
                    )
                else:
                    combined_labels = labels
                    span_labels = labels
                    sentence_labels = labels
                    predicate_labels = labels
                    arg0_labels = labels
                    arg1_labels = labels
                    frameaxis_labels = labels

                combined_pred = combined_pred.argmax(dim=1)
                span_pred = span_pred.argmax(dim=1)
                sentence_pred = sentence_pred.argmax(dim=1)
                predicate_pred = predicate_pred.argmax(dim=1)
                arg0_pred = arg0_pred.argmax(dim=1)
                arg1_pred = arg1_pred.argmax(dim=1)
                frameaxis_pred = frameaxis_pred.argmax(dim=1)
                combined_labels = combined_labels.argmax(dim=1)
                span_labels = span_labels.argmax(dim=1)
                sentence_labels = sentence_labels.argmax(dim=1)
                predicate_labels = predicate_labels.argmax(dim=1)
                arg0_labels = arg0_labels.argmax(dim=1)
                arg1_labels = arg1_labels.argmax(dim=1)
                frameaxis_labels = frameaxis_labels.argmax(dim=1)

                all_combined_preds.append(combined_pred.cpu())
                all_combined_labels.append(combined_labels.cpu())

                # Macro F1
                f1_metric_macro.add_batch(
                    predictions=combined_pred.cpu().numpy(),
                    references=combined_labels.cpu().numpy(),
                )
                f1_metric_macro_span.add_batch(
                    predictions=span_pred.cpu().numpy(),
                    references=span_labels.cpu().numpy(),
                )
                f1_metric_macro_sentence.add_batch(
                    predictions=sentence_pred.cpu().numpy(),
                    references=sentence_labels.cpu().numpy(),
                )
                f1_metric_macro_predicate.add_batch(
                    predictions=predicate_pred.cpu().numpy(),
                    references=predicate_labels.cpu().numpy(),
                )
                f1_metric_macro_arg0.add_batch(
                    predictions=arg0_pred.cpu().numpy(),
                    references=arg0_labels.cpu().numpy(),
                )
                f1_metric_macro_arg1.add_batch(
                    predictions=arg1_pred.cpu().numpy(),
                    references=arg1_labels.cpu().numpy(),
                )
                f1_metric_macro_frameaxis.add_batch(
                    predictions=frameaxis_pred.cpu().numpy(),
                    references=frameaxis_labels.cpu().numpy(),
                )

                # Micro F1
                f1_metric_micro.add_batch(
                    predictions=combined_pred.cpu().numpy(),
                    references=combined_labels.cpu().numpy(),
                )
                f1_metric_micro_span.add_batch(
                    predictions=span_pred.cpu().numpy(),
                    references=span_labels.cpu().numpy(),
                )
                f1_metric_micro_sentence.add_batch(
                    predictions=sentence_pred.cpu().numpy(),
                    references=sentence_labels.cpu().numpy(),
                )
                f1_metric_micro_predicate.add_batch(
                    predictions=predicate_pred.cpu().numpy(),
                    references=predicate_labels.cpu().numpy(),
                )
                f1_metric_micro_arg0.add_batch(
                    predictions=arg0_pred.cpu().numpy(),
                    references=arg0_labels.cpu().numpy(),
                )
                f1_metric_micro_arg1.add_batch(
                    predictions=arg1_pred.cpu().numpy(),
                    references=arg1_labels.cpu().numpy(),
                )
                f1_metric_micro_frameaxis.add_batch(
                    predictions=frameaxis_pred.cpu().numpy(),
                    references=frameaxis_labels.cpu().numpy(),
                )

                # Accuracy
                accuracy_metric.add_batch(
                    predictions=combined_pred.cpu().numpy(),
                    references=combined_labels.cpu().numpy(),
                )
                accuracy_metric_span.add_batch(
                    predictions=span_pred.cpu().numpy(),
                    references=span_labels.cpu().numpy(),
                )
                accuracy_metric_sentence.add_batch(
                    predictions=sentence_pred.cpu().numpy(),
                    references=sentence_labels.cpu().numpy(),
                )
                accuracy_metric_predicate.add_batch(
                    predictions=predicate_pred.cpu().numpy(),
                    references=predicate_labels.cpu().numpy(),
                )
                accuracy_metric_arg0.add_batch(
                    predictions=arg0_pred.cpu().numpy(),
                    references=arg0_labels.cpu().numpy(),
                )
                accuracy_metric_arg1.add_batch(
                    predictions=arg1_pred.cpu().numpy(),
                    references=arg1_labels.cpu().numpy(),
                )
                accuracy_metric_frameaxis.add_batch(
                    predictions=frameaxis_pred.cpu().numpy(),
                    references=frameaxis_labels.cpu().numpy(),
                )

                del sentence_ids, predicate_ids, arg0_ids, arg1_ids, labels
                torch.cuda.empty_cache()

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
            # Print the classification report
            logger.info("\nPer-class metrics for evaluation data:")
            logger.info(classification_report(all_combined_labels, all_combined_preds))

        # Micro F1
        eval_results_micro = f1_metric_micro.compute(average="micro")
        eval_results_micro_span = f1_metric_micro_span.compute(average="micro")
        eval_results_micro_sentence = f1_metric_micro_sentence.compute(average="micro")
        eval_results_micro_predicate = f1_metric_micro_predicate.compute(
            average="micro"
        )
        eval_results_micro_arg0 = f1_metric_micro_arg0.compute(average="micro")
        eval_results_micro_arg1 = f1_metric_micro_arg1.compute(average="micro")
        eval_results_micro_frameaxis = f1_metric_micro_frameaxis.compute(
            average="micro"
        )

        # Macro F1
        eval_results_macro = f1_metric_macro.compute(average="macro")
        eval_results_macro_span = f1_metric_macro_span.compute(average="macro")
        eval_results_macro_sentence = f1_metric_macro_sentence.compute(average="macro")
        eval_results_macro_predicate = f1_metric_macro_predicate.compute(
            average="macro"
        )
        eval_results_macro_arg0 = f1_metric_macro_arg0.compute(average="macro")
        eval_results_macro_arg1 = f1_metric_macro_arg1.compute(average="macro")
        eval_results_macro_frameaxis = f1_metric_macro_frameaxis.compute(
            average="macro"
        )

        # Accuracy
        eval_accuracy = accuracy_metric.compute()
        eval_accuracy_span = accuracy_metric_span.compute()
        eval_accuracy_sentence = accuracy_metric_sentence.compute()
        eval_accuracy_predicate = accuracy_metric_predicate.compute()
        eval_accuracy_arg0 = accuracy_metric_arg0.compute()
        eval_accuracy_arg1 = accuracy_metric_arg1.compute()
        eval_accuracy_frameaxis = accuracy_metric_frameaxis.compute()

        logger.info(
            f"[EVALUATE] Epoch {epoch}: Micro F1: {eval_results_micro}, Macro F1: {eval_results_macro}, Accuracy: {eval_accuracy}"
        )

        metrics = {
            "micro_f1": eval_results_micro["f1"],
            "micro_f1_span": eval_results_micro_span["f1"],
            "micro_f1_sentence": eval_results_micro_sentence["f1"],
            "micro_f1_predicate": eval_results_micro_predicate["f1"],
            "micro_f1_arg0": eval_results_micro_arg0["f1"],
            "micro_f1_arg1": eval_results_micro_arg1["f1"],
            "micro_f1_frameaxis": eval_results_micro_frameaxis["f1"],
            "macro_f1": eval_results_macro["f1"],
            "macro_f1_span": eval_results_macro_span["f1"],
            "macro_f1_sentence": eval_results_macro_sentence["f1"],
            "macro_f1_predicate": eval_results_macro_predicate["f1"],
            "macro_f1_arg0": eval_results_macro_arg0["f1"],
            "macro_f1_arg1": eval_results_macro_arg1["f1"],
            "macro_f1_frameaxis": eval_results_macro_frameaxis["f1"],
            "accuracy": eval_accuracy["accuracy"],
            "accuracy_span": eval_accuracy_span["accuracy"],
            "accuracy_sentence": eval_accuracy_sentence["accuracy"],
            "accuracy_predicate": eval_accuracy_predicate["accuracy"],
            "accuracy_arg0": eval_accuracy_arg0["accuracy"],
            "accuracy_arg1": eval_accuracy_arg1["accuracy"],
            "accuracy_frameaxis": eval_accuracy_frameaxis["accuracy"],
            "epoch": epoch,
            "val_loss": avg_val_loss,  # Add the average validation loss to metrics
        }

        self._log_metrics(metrics)

        # Add per-class metrics to the metrics dictionary
        for class_name, class_metrics in class_report.items():
            if isinstance(
                class_metrics, dict
            ):  # Skip 'accuracy', 'macro avg', 'weighted avg'
                self._log_metrics(
                    {
                        f"eval_precision_class_{class_name}": class_metrics[
                            "precision"
                        ],
                        f"eval_recall_class_{class_name}": class_metrics["recall"],
                        f"eval_f1_class_{class_name}": class_metrics["f1-score"],
                    }
                )

        return metrics

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
