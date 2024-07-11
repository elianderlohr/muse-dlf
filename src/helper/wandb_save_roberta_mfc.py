from pathlib import Path
import wandb

# Initialize the W&B run
run = wandb.init(project="collection-linking-quickstart")

# Define the path to the model directory
model_dir = Path(
    "models/roberta-base-finetune/roberta-base-finetune-2024-05-20_08-02-29-65707/checkpoint-16482"
)

# Log the entire directory as an artifact
artifact = wandb.Artifact(name="roberta-base-finetune-checkpoint-16482", type="model")

# Add all files in the directory to the artifact
artifact.add_dir(model_dir)

# Log the artifact to W&B
run.log_artifact(artifact)

# Link the artifact to a specific path in the W&B model registry
run.link_artifact(
    artifact=artifact,
    target_path="elianderlohr-org/wandb-registry-model/mfc-roberta-finetune",
)

# Finish the W&B run
run.finish()
