import os
import sys
import traceback
from huggingface_hub import HfApi, create_repo, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

# -----------------------
# Configuration (prefer env vars in CI)
# -----------------------
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HF_TOKEN is not set. Export your Hugging Face token and retry.", file=sys.stderr)
    sys.exit(1)

# Target Space repo id (format: username/space-name)
# Set HF_SPACE_ID in GitHub secrets or export locally; fallback default is below (change if needed)
SPACE_REPO_ID = os.getenv("HF_SPACE_ID", "sathishaiuse/tourism-predictor")

# Which SDK the Space should use when created: docker | gradio | streamlit | static
SPACE_SDK = os.getenv("SPACE_SDK", "docker").lower()
if SPACE_SDK not in ("docker", "gradio", "streamlit", "static"):
    print(f"ERROR: Invalid SPACE_SDK '{SPACE_SDK}'. Choose one of: docker, gradio, streamlit, static", file=sys.stderr)
    sys.exit(1)

# Local folder containing deployment files to upload
LOCAL_DEPLOY_FOLDER = os.getenv("LOCAL_DEPLOY_FOLDER", "tourism_project/deployment")
# Privacy setting
PRIVATE = os.getenv("SPACE_PRIVATE", "false").lower() in ("1", "true", "yes")

# -----------------------
# Utility helpers
# -----------------------
def split_repo_id(repo_id: str):
    """Return (org, name) or (None, name) for a repo_id like 'org/name' or 'name'."""
    if "/" in repo_id:
        parts = repo_id.split("/")
        if len(parts) != 2:
            raise ValueError("HF_SPACE_ID must be 'username/space-name' (single slash).")
        return parts[0], parts[1]
    return None, repo_id

# -----------------------
# Initialize API
# -----------------------
api = HfApi(token=HF_TOKEN)

# -----------------------
# Ensure deployment folder exists
# -----------------------
if not os.path.isdir(LOCAL_DEPLOY_FOLDER):
    print(f"ERROR: Local deployment folder '{LOCAL_DEPLOY_FOLDER}' not found.", file=sys.stderr)
    print("Create it and add Dockerfile + requirements.txt + app.py (or for gradio/streamlit, just app.py),", file=sys.stderr)
    sys.exit(1)

# -----------------------
# Ensure Space exists (create if not)
# -----------------------
org, name = split_repo_id(SPACE_REPO_ID)

try:
    api.repo_info(repo_id=SPACE_REPO_ID, repo_type="space")
    print(f"Space '{SPACE_REPO_ID}' already exists. Proceeding to upload files.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO_ID}' not found. Creating new Space with SDK='{SPACE_SDK}'...")
    try:
        # create_repo expects `name` and optional `organization`, and for spaces requires space_sdk
        create_repo(
            name=name,
            repo_type="space",
            space_sdk=SPACE_SDK,
            organization=org if org else None,
            private=PRIVATE,
            token=HF_TOKEN,
        )
        print(f"Space '{SPACE_REPO_ID}' created.")
    except Exception as e:
        print("Failed to create space:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
except HfHubHTTPError as e:
    print("HTTP error when checking the space:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print("Unexpected error when checking/creating space:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

# -----------------------
# Upload files to the Space repo
# -----------------------
print(f"Uploading contents of '{LOCAL_DEPLOY_FOLDER}' to space '{SPACE_REPO_ID}' ...")
try:
    upload_folder(
        folder_path=LOCAL_DEPLOY_FOLDER,
        path_in_repo=".",   # root of the space repo
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        token=HF_TOKEN,
        commit_message="Upload deployment files from CI"
    )
    print("Upload finished successfully.")
    print(f"Visit your space at: https://huggingface.co/spaces/{SPACE_REPO_ID}")
except Exception as e:
    print("Upload failed:", e, file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)
