import argparse
from pathlib import Path
import wandb

wandb.require("core")


def main(project, dir_path, artifact_name, artifact_type):
    run = wandb.init(project=project)

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory.")

    artifact = wandb.Artifact(artifact_name, type=artifact_type)

    # Add files to the artifact
    for file_path in dir_path.iterdir():
        if file_path.suffix in [".pth", ".json"]:
            artifact.add_file(str(file_path))

    logged_artifact = run.log_artifact(artifact)
    logged_artifact.wait()

    # Hardcoded base target path
    target_path = "elianderlohr-org/wandb-registry-model/" + project

    run.link_artifact(artifact=logged_artifact, target_path=target_path)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Log and link an artifact with Weights & Biases."
    )
    parser.add_argument("--project", required=True, help="The project name in W&B.")
    parser.add_argument(
        "--dir_path",
        required=True,
        help="The directory path containing the artifact files.",
    )
    parser.add_argument(
        "--artifact_name", required=True, help="The name of the artifact."
    )
    parser.add_argument(
        "--artifact_type", required=True, help="The type of the artifact (e.g., model)."
    )

    args = parser.parse_args()
    main(
        args.project,
        args.dir_path,
        args.artifact_name,
        args.artifact_type,
    )
