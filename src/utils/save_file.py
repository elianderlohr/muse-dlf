# Import necessary libraries
import torch
from transformers import RobertaForSequenceClassification
import wandb

# Load the fine-tuned RoBERTa model
model_path = "models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482"
model = RobertaForSequenceClassification.from_pretrained(model_path)

# Save the model locally
save_path = model_path + "/roberta_model.pth"
torch.save(model.state_dict(), save_path)

# Initialize wandb and save the model as an artifact for version control
run = wandb.init(project="roberta-base-finetune")
artifact = wandb.Artifact("roberta_model", type="model")
artifact.add_file(save_path)
run.log_artifact(artifact)
run.finish()
