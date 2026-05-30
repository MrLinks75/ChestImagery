"""Verify that saved checkpoints correspond to logged MLflow runs.

Run after training: python scripts/verify_mlflow_consistency.py
"""
import mlflow
from pathlib import Path

EXPECTED_MODELS = ["cnn_scratch", "densenet121", "vit_base_patch16_224"]

mlflow.set_tracking_uri("runs/")

client = mlflow.MlflowClient()
exp = client.get_experiment_by_name("supervised_classification")
if exp is None:
    print("No 'supervised_classification' experiment found. Train models first.")
    exit(1)

runs = client.search_runs(exp.experiment_id)
run_names = {r.data.tags.get("mlflow.runName") for r in runs}
print(f"MLflow runs found: {run_names}")

all_ok = True
for model_name in EXPECTED_MODELS:
    ckpt = Path(f"checkpoints/{model_name}_best.pth")
    in_mlflow = model_name in run_names
    on_disk = ckpt.exists()
    status = "OK" if (in_mlflow and on_disk) else "MISSING"
    print(f"  {status}  {model_name:30s} | MLflow={in_mlflow} | checkpoint={on_disk}")
    if status != "OK":
        all_ok = False

if all_ok:
    print("\nAll checkpoints consistent with MLflow runs.")
else:
    print("\nInconsistencies found — retrain missing models.")
