"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Package the model with MLflow.
"""

from pathlib import Path
import os
import json
from datetime import datetime

import mlflow.pyfunc
from huggingface_hub import hf_hub_download

from aurora.foundry.common.model import models
import aurora
from aurora.foundry.server.mlflow_wrapper import AuroraModelWrapper

artifacts: dict[str, str] = {}

# Download checkpoints into a local directory which will be included in the package.
ckpt_dir = Path("checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)
selected = os.environ.get("AURORA_SELECTED_MODELS")
names = [n.strip() for n in selected.split(",") if n.strip()] if selected else list(models.keys())
for name in names:
    hf_hub_download(
        repo_id="microsoft/aurora",
        filename=f"{name}.ckpt",
        local_dir=ckpt_dir,
    )
    artifacts[name] = str(ckpt_dir / f"{name}.ckpt")

# Write packaging metadata including checkpoint revisions per variant.
metadata_dir = ckpt_dir
metadata = {
    "packaged_at": datetime.utcnow().isoformat() + "Z",
    "models": [],
}
for name in names:
    # Resolve default checkpoint revision via aurora classes
    try:
        cls = {
            "aurora-0.25-finetuned": aurora.Aurora,
            "aurora-0.25-pretrained": aurora.AuroraPretrained,
            "aurora-0.25-small-pretrained": aurora.AuroraSmallPretrained,
            "aurora-0.25-12h-pretrained": aurora.Aurora12hPretrained,
            "aurora-0.1-finetuned": aurora.AuroraHighRes,
            "aurora-0.4-air-pollution": aurora.AuroraAirPollution,
            "aurora-0.25-wave": aurora.AuroraWave,
        }[name]
        revision = getattr(cls, "default_checkpoint_revision", None)
    except Exception:
        revision = None
    metadata["models"].append({"name": name, "checkpoint_revision": revision})

versions_path = metadata_dir / "artifacts_versions.json"
with open(versions_path, "w") as f:
    json.dump(metadata, f, indent=2)
artifacts["artifacts_versions"] = str(versions_path)


mlflow_pyfunc_model_path = "./aurora_mlflow_pyfunc"
mlflow.pyfunc.save_model(
    path=mlflow_pyfunc_model_path,
    code_paths=["aurora"],
    python_model=AuroraModelWrapper(),
    artifacts=artifacts,
    conda_env={
        "name": "aurora-mlflow-env",
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.11.11",
            "pip<=24.3.1",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "cloudpickle==3.1.1",
                    "defusedxml==0.7.1",
                    "einops==0.8.1",
                    "jaraco-collections==5.1.0",
                    "numpy==2.3.0",
                    "scipy==1.15.3",
                    "timm==1.0.15",
                    "torch==2.5.1",
                    "torchvision==0.20.1",
                    "huggingface-hub==0.33.0",
                    "pydantic==2.11.7",
                    "xarray==2025.6.1",
                    "netCDF4==1.7.2",
                    "azure-storage-blob==12.25.1",
                ],
            },
        ],
    },
)
