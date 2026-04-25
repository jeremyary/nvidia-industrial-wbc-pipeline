# This project was developed with assistance from AI tools.
"""Download the latest training checkpoint from MinIO/S3."""

import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    config=Config(s3={"addressing_style": "path", "use_accelerate_endpoint": False}),
)

prefix = os.environ.get("S3_PREFIX", "overnight-flat-30k")
iteration = os.environ.get("ITERATION", "latest")
bucket = os.environ.get("S3_BUCKET", "wbc-training")
output_dir = os.environ.get("OUTPUT_DIR", str(PROJECT_DIR / "checkpoints"))

os.makedirs(output_dir, exist_ok=True)

# List checkpoints
paginator = s3.get_paginator("list_objects_v2")
model_files = []
for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/model_"):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".pt"):
            iter_num = int(key.split("model_")[1].split(".")[0])
            model_files.append((iter_num, key))

if not model_files:
    print(f"No checkpoints found at s3://{bucket}/{prefix}/")
    sys.exit(1)

model_files.sort(key=lambda x: x[0])

if iteration == "latest":
    iter_num, key = model_files[-1]
else:
    target = int(iteration)
    matches = [(n, k) for n, k in model_files if n == target]
    if not matches:
        available = ", ".join(str(n) for n, _ in model_files[-10:])
        print(f"Iteration {target} not found. Latest available: {available}")
        sys.exit(1)
    iter_num, key = matches[0]

filename = os.path.basename(key)
safe_prefix = prefix.replace("/", "_")
local_path = os.path.join(output_dir, f"{safe_prefix}_{filename}")
s3.download_file(bucket, key, local_path)
print(f"Downloaded s3://{bucket}/{key} -> {local_path}")

# Also grab policy.onnx if it exists
onnx_key = f"{prefix}/policy.onnx"
onnx_path = os.path.join(output_dir, f"{safe_prefix}_policy.onnx")
try:
    s3.download_file(bucket, onnx_key, onnx_path)
    print(f"Downloaded s3://{bucket}/{onnx_key} -> {onnx_path}")
except Exception:
    pass  # no ONNX yet (training still running)

print()
print(f"Checkpoint: {local_path}")
print(f"Iteration:  {iter_num}")
print(f"Available:  {model_files[0][0]} - {model_files[-1][0]} ({len(model_files)} checkpoints)")
