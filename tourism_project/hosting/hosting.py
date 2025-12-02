"""
This script uploads all deployment files into a Hugging Face Space.

Make sure:
1. HF_TOKEN is set in your environment.
2. You already created or want this script to create a Space repo.
3. Your deployment files (Dockerfile, app.py, requirements.txt, etc.)
   are located in: tourism_project/deployment/
"""

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable not found. Please export it first.")

api = HfApi(token=HF_TOKEN)

# ---------------------------
# CONFIG — EDIT THESE VALUES
# ---------------------------

SPACE_REPO_ID = "sathishaiuse/tourism-predictor"
LOCAL_DEPLOY_FOLDER = "tourism_project/deployment"           # <--- Deployment folder with Dockerfile + app.py

# ---------------------------
# CREATE SPACE IF NOT EXISTS
# ---------------------------

try:
    api.repo_info(repo_id=SPACE_REPO_ID, repo_type="space")
    print(f"Space '{SPACE_REPO_ID}' already exists.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO_ID}' does NOT exist. Creating now...")
    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        private=False,
        token=HF_TOKEN
    )
    print(f"Space '{SPACE_REPO_ID}' successfully created.")

# ---------------------------
# UPLOAD DEPLOYMENT FILES
# ---------------------------

print(f"\nUploading folder '{LOCAL_DEPLOY_FOLDER}' to space '{SPACE_REPO_ID}'...")

api.upload_folder(
    folder_path=LOCAL_DEPLOY_FOLDER,
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    path_in_repo="",    # root of the Space repo
    commit_message="Upload deployment files"
)

print("\n Deployment files uploaded successfully!")
print(f"Your Space will build automatically: https://huggingface.co/spaces/{SPACE_REPO_ID}")
