from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os
# from google.colab import userdata

try:
    from google.colab import userdata
except ModuleNotFoundError:
    userdata = None

user_id = "puneet83"
# user_id ="<-----Hugging Face User ID ----->"
repo_id = f"{user_id}/tourism_product"
repo_type = "dataset"


if userdata:
    token = userdata.get("HF_TOKEN")
else:
    token = os.environ["HF_TOKEN"]
# Initialize API client
api = HfApi(token=token)


# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
