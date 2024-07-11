from pathlib import Path
import wandb

wandb.require("core")


def save_model_to_wandb(run, project, dir_path, artifact_name, artifact_type):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory.")

    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # Add specific files to the artifact
    for file_name in ["model.pth", "metrics.json", "config.json"]:
        file_path = dir_path / file_name
        if file_path.exists():
            artifact.add_file(str(file_path))
        else:
            print(f"Warning: {file_path} does not exist and will not be added.")

    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait()

    # Hardcoded base target path
    target_path = f"elianderlohr-org/wandb-registry-model/{project}"

    run.link_artifact(artifact=logged_artifact, target_path=target_path)
