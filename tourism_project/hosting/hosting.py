from huggingface_hub import HfApi
import os
import subprocess

try:
    from google.colab import userdata
except ModuleNotFoundError:
    userdata = None

user_id = "puneet83"


if userdata:
    token = userdata.get("HF_TOKEN")
else:
    token = os.environ["HF_TOKEN"]
# Initialize API client
api = HfApi(token=token)
# api = HfApi(token=userdata.get('HF_TOKEN'))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=f"{user_id}/tourism_product",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
