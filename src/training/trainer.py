import os
import torch

import json
from tqdm import tqdm
import math
import evaluate

from utils.logging_manager import LoggerManager

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
        **kwargs,
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
            training_management: The training management tool to be used. Options are 'accelerate', 'wandb', or None.
            tau_min: The minimum value of tau.
            tau_decay: The decay factor for tau.

        Returns:
            None
        """
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

        # self.gradient_accumulation_steps = 4

        self.training_management = training_management
        if self.training_management == "accelerate":
            logger.info("Using Accelerate for training.")

            from accelerate import Accelerator

            if "accelerator_instance" in kwargs:
                self.accelerator: Accelerator = kwargs["accelerator_instance"]
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
                # raise error
                raise ValueError(
                    "You must provide a wandb instance if you want to use wandb for training."
                )
        else:
            logger.info("Using standard PyTorch for training.")
            self.accelerator = None

    def _log_metrics(self, metrics):
        if self.training_management == "wandb":
            self.wandb.log(metrics)
        elif self.training_management == "accelerate":
            self.accelerator.log(metrics)
        else:
            logger.info(metrics)

    def _train(
        self,
        epoch,
        model,
        train_dataloader,
        tau,
        alpha,
        device,
        early_stopping={
            "best_accuracy": 0,
            "best_micro_f1": 0,
            "best_macro_f1": 0,
            "early_stop": 0,
            "early_stopped": False,
        },
    ):
        model.train()
        total_loss, supervised_total_loss, unsupervised_total_loss = 0, 0, 0
        global global_steps

        # Load the evaluate metrics
        f1_metric_micro = evaluate.load("f1", config_name="micro")
        f1_metric_macro = evaluate.load("f1", config_name="macro")
        accuracy_metric = evaluate.load("accuracy")

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
            predicate_attention_masks = (
                batch["predicate_attention_masks"]
                if self.training_management == "accelerate"
                else batch["predicate_attention_masks"].to(device)
            )

            arg0_ids = (
                batch["arg0_ids"]
                if self.training_management == "accelerate"
                else batch["arg0_ids"].to(device)
            )
            arg0_attention_masks = (
                batch["arg0_attention_masks"]
                if self.training_management == "accelerate"
                else batch["arg0_attention_masks"].to(device)
            )

            arg1_ids = (
                batch["arg1_ids"]
                if self.training_management == "accelerate"
                else batch["arg1_ids"].to(device)
            )
            arg1_attention_masks = (
                batch["arg1_attention_masks"]
                if self.training_management == "accelerate"
                else batch["arg1_attention_masks"].to(device)
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

            unsupervised_loss, span_logits, sentence_logits, combined_logits = model(
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

            # LOSS

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

            if self.training_management == "accelerate":
                self.accelerator.backward(combined_loss)
                self.accelerator.clip_grad_norm_(model.parameters(), 1.0)
            else:
                combined_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += combined_loss.item()
            supervised_total_loss += supervised_loss.item()
            unsupervised_total_loss += unsupervised_loss.item()

            self._log_metrics(
                {
                    "batch_combined_loss": combined_loss.item(),
                    "batch_supervised_loss": supervised_loss.item(),
                    "batch_unsupervised_loss": unsupervised_loss.item(),
                    "tau": tau,
                    "epoch": epoch,
                }
            )

            # Check train metrics every 50 steps
            if local_steps % 50 == 0:
                combined_pred = (torch.softmax(combined_logits, dim=1) > 0.5).int()

                if self.training_management == "accelerate":
                    combined_pred, labels = self.accelerator.gather_for_metrics(
                        (combined_pred, labels)
                    )

                # transform from one-hot to class index
                combined_pred = combined_pred.argmax(dim=1)
                labels = labels.argmax(dim=1)

                f1_metric_macro.add_batch(
                    predictions=combined_pred.cpu().numpy(),
                    references=labels.cpu().numpy(),
                )

                f1_metric_micro.add_batch(
                    predictions=combined_pred.cpu().numpy(),
                    references=labels.cpu().numpy(),
                )

                accuracy_metric.add_batch(
                    predictions=combined_pred.cpu().numpy(),
                    references=labels.cpu().numpy(),
                )

                eval_results_micro = f1_metric_micro.compute(average="micro")
                eval_results_macro = f1_metric_macro.compute(average="macro")
                eval_accuracy = accuracy_metric.compute()

                logger.info(
                    f"Epoch {epoch}, Micro F1: {eval_results_micro}, Macro F1: {eval_results_macro}, Accuracy: {eval_accuracy}"
                )

                metrics = {
                    "train_micro_f1": eval_results_micro["f1"],
                    "train_macro_f1": eval_results_macro["f1"],
                    "train_accuracy": eval_accuracy["accuracy"],
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
                        # Save the best model
                        self._save_best_model(model, metrics)
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
            },
        )

        return early_stopping

    def _evaluate(self, epoch, model, test_dataloader, device, tau):
        model.eval()

        # Load the evaluate metrics
        f1_metric_micro = evaluate.load("f1", config_name="micro")
        f1_metric_macro = evaluate.load("f1", config_name="macro")
        accuracy_metric = evaluate.load("accuracy")

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
            predicate_attention_masks = (
                batch["predicate_attention_masks"]
                if self.training_management == "accelerate"
                else batch["predicate_attention_masks"].to(device)
            )

            arg0_ids = (
                batch["arg0_ids"]
                if self.training_management == "accelerate"
                else batch["arg0_ids"].to(device)
            )
            arg0_attention_masks = (
                batch["arg0_attention_masks"]
                if self.training_management == "accelerate"
                else batch["arg0_attention_masks"].to(device)
            )

            arg1_ids = (
                batch["arg1_ids"]
                if self.training_management == "accelerate"
                else batch["arg1_ids"].to(device)
            )
            arg1_attention_masks = (
                batch["arg1_attention_masks"]
                if self.training_management == "accelerate"
                else batch["arg1_attention_masks"].to(device)
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
                _, _, _, combined_logits = model(
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

            combined_pred = (torch.softmax(combined_logits, dim=1) > 0.5).int()

            if self.training_management == "accelerate":
                combined_pred, labels = self.accelerator.gather_for_metrics(
                    (combined_pred, labels)
                )

            # transform from one-hot to class index
            combined_pred = combined_pred.argmax(dim=1)
            labels = labels.argmax(dim=1)

            f1_metric_macro.add_batch(
                predictions=combined_pred.cpu().numpy(),
                references=labels.cpu().numpy(),
            )

            f1_metric_micro.add_batch(
                predictions=combined_pred.cpu().numpy(),
                references=labels.cpu().numpy(),
            )

            accuracy_metric.add_batch(
                predictions=combined_pred.cpu().numpy(),
                references=labels.cpu().numpy(),
            )

            # Explicitly delete tensors to free up memory
            del (
                sentence_ids,
                predicate_ids,
                arg0_ids,
                arg1_ids,
                labels,
            )
            torch.cuda.empty_cache()

        eval_results_micro = f1_metric_micro.compute(average="micro")
        eval_results_macro = f1_metric_macro.compute(average="macro")
        eval_accuracy = accuracy_metric.compute()

        logger.info(
            f"Epoch {epoch}, Micro F1: {eval_results_micro}, Macro F1: {eval_results_macro}, Accuracy: {eval_accuracy}"
        )

        metrics = {
            "micro_f1": eval_results_micro["f1"],
            "macro_f1": eval_results_macro["f1"],
            "accuracy": eval_accuracy["accuracy"],
            "epoch": epoch,
        }

        self._log_metrics(metrics)

        return metrics

    def _save_best_model(self, model, metrics):
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
            torch.save(model.state_dict(), model_save_path)
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

    def OLD_save_model(self, epoch, model, metrics, keep_last_n=3):
        save_dir = os.path.join(self.save_path)
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            logger.info(
                f"Warning: Could not create directory {save_dir}. Exception: {e}"
            )

        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
        try:
            torch.save(model.state_dict(), model_save_path)
        except Exception as e:
            logger.info(
                f"Warning: Failed to save model at {model_save_path}. Exception: {e}"
            )

        metrics_save_path = os.path.join(save_dir, f"metrics_epoch_{epoch}.json")
        try:
            with open(metrics_save_path, "w") as f:
                json.dump(metrics, f)
        except Exception as e:
            logger.info(
                f"Warning: Failed to save metrics at {metrics_save_path}. Exception: {e}"
            )

        # Handling model files
        try:
            model_files = [
                file
                for file in os.listdir(save_dir)
                if file.startswith("model_epoch_") and file.endswith(".pth")
            ]
            model_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

            if len(model_files) > keep_last_n:
                for file_to_delete in model_files[:-keep_last_n]:
                    file_path = os.path.join(save_dir, file_to_delete)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.info(
                                f"Warning: Failed to delete file {file_path}. Exception: {e}"
                            )
        except Exception as e:
            logger.info(
                f"Warning: An error occurred while managing model files. Exception: {e}"
            )

        # Handling metric files
        try:
            metric_files = [
                file
                for file in os.listdir(save_dir)
                if file.startswith("metrics_epoch_") and file.endswith(".json")
            ]
            metric_files.sort(key=lambda x: int(x.split("_")[2].split(".")[0]))

            if len(metric_files) > keep_last_n:
                for file_to_delete in metric_files[:-keep_last_n]:
                    file_path = os.path.join(save_dir, file_to_delete)
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            logger.info(
                                f"Warning: Failed to delete file {file_path}. Exception: {e}"
                            )
        except Exception as e:
            logger.info(
                f"Warning: An error occurred while managing metric files. Exception: {e}"
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
        }

        for epoch in range(1, epochs + 1):
            early_stopping = self._train(
                epoch,
                self.model,
                self.train_dataloader,
                tau,
                alpha,
                self.device,
                early_stopping,
            )

            if early_stopping["early_stopped"]:
                break

            self._evaluate(epoch, self.model, self.test_dataloader, self.device, tau)
