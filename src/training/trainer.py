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

logger = LoggerManager.get_logger(__name__)


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
        training_management=None,  # 'accelerate', 'wandb', or None
        tau_min=1,
        tau_decay=0.95,
        early_stop=20,
        mixed_precision="fp16",  # "fp16"
        clip_value=1.0,
        **kwargs,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.tau_min = tau_min
        self.tau_decay = tau_decay
        self.early_stop = early_stop

        self.scaler = GradScaler() if mixed_precision in ["fp16", "bf16"] else None

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

        self.mixed_precision = mixed_precision
        self.clip_value = clip_value

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
        global global_steps

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

        local_steps = 0
        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc=f"Train - Epoch {epoch}")
        ):
            global_steps += 1
            if global_steps % 50 == 0:
                tau = max(self.tau_min, math.exp(-self.tau_decay * global_steps))

            local_steps += 1

            self.optimizer.zero_grad()

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

            with autocast(
                enabled=self.mixed_precision in ["fp16", "bf16"],
                dtype=(
                    torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16
                ),
            ):
                (
                    unsupervised_loss,
                    span_logits,
                    sentence_logits,
                    combined_logits,
                    other,
                ) = self.model(
                    sentence_ids,
                    sentence_attention_masks,
                    predicate_ids,
                    arg0_ids,
                    arg1_ids,
                    frameaxis_data,
                    tau,
                    mixed_precision=self.mixed_precision,
                )

                # Check for NaNs in model outputs
                if (
                    self.check_for_nans(unsupervised_loss, "unsupervised_loss")
                    or self.check_for_nans(span_logits, "span_logits")
                    or self.check_for_nans(sentence_logits, "sentence_logits")
                    or self.check_for_nans(combined_logits, "combined_logits")
                    or self.check_for_nans(other["predicate"], "other['predicate']")
                    or self.check_for_nans(other["arg0"], "other['arg0']")
                    or self.check_for_nans(other["arg1"], "other['arg1']")
                    or self.check_for_nans(other["frameaxis"], "other['frameaxis']")
                ):
                    logger.error(
                        f"{experiment_id} - {batch_idx} - NaNs detected in model outputs, skipping this batch."
                    )
                    continue

                span_loss = 0.0
                sentence_loss = 0.0

                span_loss = self.loss_function(span_logits, labels.float())
                sentence_loss = self.loss_function(sentence_logits, labels.float())
                supervised_loss = span_loss + sentence_loss

                sum_of_parameters = sum(p.sum() for p in self.model.parameters())
                zero_sum = sum_of_parameters * 0.0
                combined_loss = (
                    alpha * supervised_loss + (1 - alpha) * unsupervised_loss
                ) + zero_sum

            # other loss (debug)
            predicate_loss = self.loss_function(other["predicate"], labels.float())
            arg0_loss = self.loss_function(other["arg0"], labels.float())
            arg1_loss = self.loss_function(other["arg1"], labels.float())
            frameaxis_loss = self.loss_function(other["frameaxis"], labels.float())

            # Check for NaNs in combined loss
            if self.check_for_nans(combined_loss, "combined_loss"):
                logger.error(
                    f"{experiment_id} - {batch_idx} - NaNs detected in combined_loss, skipping this batch."
                )
                continue

            # Scale the loss and perform the backward pass
            if self.training_management == "accelerate":
                self.accelerator.backward(combined_loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), self.clip_value
                    )
                self.optimizer.step()
            else:
                if self.scaler is not None:
                    self.scaler.scale(combined_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_value
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    combined_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_value
                    )
                    self.optimizer.step()

            total_loss += combined_loss.item()
            supervised_total_loss += supervised_loss.item()
            unsupervised_total_loss += unsupervised_loss.item()

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
                    "tau": tau,
                    "epoch": epoch,
                }
            )

            # Check train metrics every 50 steps
            if local_steps % 50 == 0:
                combined_pred = (torch.softmax(combined_logits, dim=1) > 0.5).int()
                span_pred = (torch.softmax(span_logits, dim=1) > 0.5).int()
                sentence_pred = (torch.softmax(sentence_logits, dim=1) > 0.5).int()

                # predicate, arg0, arg1, frameaxis
                predicate_pred = (torch.softmax(other["predicate"], dim=1) > 0.5).int()
                arg0_pred = (torch.softmax(other["arg0"], dim=1) > 0.5).int()
                arg1_pred = (torch.softmax(other["arg1"], dim=1) > 0.5).int()
                frameaxis_pred = (torch.softmax(other["frameaxis"], dim=1) > 0.5).int()

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

                # transform from one-hot to class index
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
                eval_results_micro_span = f1_metric_micro_span.compute(average="micro")
                eval_results_micro_sentence = f1_metric_micro_sentence.compute(
                    average="micro"
                )
                eval_results_micro_predicate = f1_metric_micro_predicate.compute(
                    average="micro"
                )
                eval_results_micro_arg0 = f1_metric_micro_arg0.compute(average="micro")
                eval_results_micro_arg1 = f1_metric_micro_arg1.compute(average="micro")
                eval_results_micro_frameaxis = f1_metric_micro_frameaxis.compute(
                    average="micro"
                )

                eval_results_macro = f1_metric_macro.compute(average="macro")
                eval_results_macro_span = f1_metric_macro_span.compute(average="macro")
                eval_results_macro_sentence = f1_metric_macro_sentence.compute(
                    average="macro"
                )
                eval_results_macro_predicate = f1_metric_macro_predicate.compute(
                    average="macro"
                )
                eval_results_macro_arg0 = f1_metric_macro_arg0.compute(average="macro")
                eval_results_macro_arg1 = f1_metric_macro_arg1.compute(average="macro")
                eval_results_macro_frameaxis = f1_metric_macro_frameaxis.compute(
                    average="macro"
                )

                eval_accuracy = accuracy_metric.compute()
                eval_accuracy_span = accuracy_metric_span.compute()
                eval_accuracy_sentence = accuracy_metric_sentence.compute()
                eval_accuracy_predicate = accuracy_metric_predicate.compute()
                eval_accuracy_arg0 = accuracy_metric_arg0.compute()
                eval_accuracy_arg1 = accuracy_metric_arg1.compute()
                eval_accuracy_frameaxis = accuracy_metric_frameaxis.compute()

                logger.info(
                    f"Epoch {epoch}, Micro F1: {eval_results_micro}, Macro F1: {eval_results_macro}, Accuracy: {eval_accuracy}"
                )

                metrics = {
                    "train_micro_f1": eval_results_micro["f1"],
                    "train_micro_f1_span": eval_results_micro_span["f1"],
                    "train_micro_f1_sentence": eval_results_micro_sentence["f1"],
                    "train_macro_f1": eval_results_macro["f1"],
                    "train_macro_f1_span": eval_results_macro_span["f1"],
                    "train_macro_f1_sentence": eval_results_macro_sentence["f1"],
                    "train_accuracy": eval_accuracy["accuracy"],
                    "train_accuracy_span": eval_accuracy_span["accuracy"],
                    "train_accuracy_sentence": eval_accuracy_sentence["accuracy"],
                    "train_micro_f1_predicate": eval_results_micro_predicate["f1"],
                    "train_micro_f1_arg0": eval_results_micro_arg0["f1"],
                    "train_micro_f1_arg1": eval_results_micro_arg1["f1"],
                    "train_micro_f1_frameaxis": eval_results_micro_frameaxis["f1"],
                    "train_macro_f1_predicate": eval_results_macro_predicate["f1"],
                    "train_macro_f1_arg0": eval_results_macro_arg0["f1"],
                    "train_macro_f1_arg1": eval_results_macro_arg1["f1"],
                    "train_macro_f1_frameaxis": eval_results_macro_frameaxis["f1"],
                    "train_accuracy_predicate": eval_accuracy_predicate["accuracy"],
                    "train_accuracy_arg0": eval_accuracy_arg0["accuracy"],
                    "train_accuracy_arg1": eval_accuracy_arg1["accuracy"],
                    "train_accuracy_frameaxis": eval_accuracy_frameaxis["accuracy"],
                    "epoch": epoch,
                    "batch": local_steps,
                    "global_steps": global_steps,
                }

                self._log_metrics(metrics)

                if eval_accuracy["accuracy"] > early_stopping["best_accuracy"]:
                    early_stopping["best_accuracy"] = eval_accuracy["accuracy"]
                    early_stopping["best_micro_f1"] = eval_results_micro["f1"]
                    early_stopping["best_macro_f1"] = eval_results_macro["f1"]
                    early_stopping["early_stop"] = 0

                    if eval_accuracy["accuracy"] > 0.5:
                        self._save_best_model(metrics)
                else:
                    early_stopping["early_stop"] += 1

                    if early_stopping["early_stop"] >= self.early_stop:
                        logger.info("Early stopping triggered.")
                        early_stopping["early_stopped"] = True
                        return early_stopping

            del (
                sentence_ids,
                predicate_ids,
                arg0_ids,
                arg1_ids,
                labels,
                unsupervised_loss,
                supervised_loss,
                combined_loss,
            )
            torch.cuda.empty_cache()

        avg_total_loss = total_loss / len(train_dataloader)
        avg_supervised_loss = supervised_total_loss / len(train_dataloader)
        avg_unsupervised_loss = unsupervised_total_loss / len(train_dataloader)

        logger.info(
            f"Epoch {epoch}, Avg Total Loss: {avg_total_loss}, Avg Supervised Loss: {avg_supervised_loss}, Avg Unsupervised Loss: {avg_unsupervised_loss}"
        )

        self._log_metrics(
            {
                "epoch_combined_loss": avg_total_loss,
                "epoch_supervised_loss": avg_supervised_loss,
                "epoch_unsupervised_loss": avg_unsupervised_loss,
                "epoch": epoch,
            }
        )

        return early_stopping

    def _evaluate(self, epoch, test_dataloader, device, tau, experiment_id):
        self.model.eval()

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
            else torch.bfloat16 if self.mixed_precision == "bf16" else None
        )

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

            with torch.no_grad():
                with autocast(
                    enabled=self.mixed_precision in ["fp16", "bf16"],
                    dtype=precision_dtype,
                ):
                    _, span_logits, sentence_logits, combined_logits, other = (
                        self.model(
                            sentence_ids,
                            sentence_attention_masks,
                            predicate_ids,
                            arg0_ids,
                            arg1_ids,
                            frameaxis_data,
                            tau,
                        )
                    )

                combined_pred = (torch.softmax(combined_logits, dim=1) > 0.5).int()
                span_pred = (torch.softmax(span_logits, dim=1) > 0.5).int()
                sentence_pred = (torch.softmax(sentence_logits, dim=1) > 0.5).int()

                # predicate, arg0, arg1, frameaxis
                predicate_pred = (torch.softmax(other["predicate"], dim=1) > 0.5).int()
                arg0_pred = (torch.softmax(other["arg0"], dim=1) > 0.5).int()
                arg1_pred = (torch.softmax(other["arg1"], dim=1) > 0.5).int()
                frameaxis_pred = (torch.softmax(other["frameaxis"], dim=1) > 0.5).int()

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

                # transform from one-hot to class index
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

                del sentence_ids, predicate_ids, arg0_ids, arg1_ids, labels
                torch.cuda.empty_cache()

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
            f"Epoch {epoch}, Micro F1: {eval_results_micro}, Macro F1: {eval_results_macro}, Accuracy: {eval_accuracy}"
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
        }

        self._log_metrics(metrics)

        return metrics

    def _save_best_model(self, metrics):
        # save dir path
        save_dir = os.path.join(self.save_path)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            logger.info(
                f"Warning: Could not create directory {save_dir}. Exception: {e}"
            )

        # save model
        model_save_path = os.path.join(save_dir, f"model.pth")
        try:
            torch.save(self.model.state_dict(), model_save_path)
        except Exception as e:
            logger.info(
                f"Warning: Failed to save model at {model_save_path}. Exception: {e}"
            )

        # save metrics
        metrics_save_path = os.path.join(save_dir, f"metrics.json")
        try:
            with open(metrics_save_path, "w") as f:
                json.dump(metrics, f)
        except Exception as e:
            logger.info(
                f"Warning: Failed to save metrics at {metrics_save_path}. Exception: {e}"
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
                early_stopping = self._train(
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
                    epoch, self.test_dataloader, self.device, tau, experiment_id
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

            if epoch > 2 and metrics["accuracy"] < 0.5:
                logger.info("Accuracy is below 0.5. Stopping training.")
                early_stopping["early_stopped"] = True
                early_stopping["stopping_code"] = 102
                self._log_alert(
                    title="Accuracy is below 0.5.",
                    text="The model never surpassed 0.5 accuracy.",
                )
                break

        return early_stopping
