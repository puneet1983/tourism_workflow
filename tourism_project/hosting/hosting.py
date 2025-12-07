from huggingface_hub import HfApi
import os
import subprocess
from google.colab import userdata

user_id = "puneet83"

HF_TOKEN = userdata.get("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("❌ HF_TOKEN not found. Please set it using userdata.set().")

api = HfApi(token=userdata.get('HF_TOKEN'))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=f"{user_id}/tourism_product",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
