import os
import sys
from huggingface_hub import HfApi

# Exclude local caches, notebook folders, and git history to keep upload fast and small
EXCLUDED_PATTERNS = [
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    "NoteBooks",
    "deploy_to_hf.py",
    ".streamlit/config.toml" # Hugging Face handles its own streamlit theme config or we can include it
]

def deploy():
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    
    if not token or not repo_id:
        print("Error: Missing HF_TOKEN or HF_REPO_ID environment variables.")
        sys.exit(1)
        
    print(f"Uploading files to Hugging Face Space: {repo_id}...")
    api = HfApi()
    
    # We assume the Space already exists as configured by the user.
    try:
        # Uploading the folder contents
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            token=token,
            ignore_patterns=EXCLUDED_PATTERNS
        )
        print("\nDeployment completed successfully!")
        print(f"View your live app here: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        # Avoid crashing when printing exception messages with non-ASCII characters
        safe_error = str(e).encode('ascii', 'ignore').decode('ascii')
        print(f"\nError during file upload: {safe_error}")
        sys.exit(1)

if __name__ == "__main__":
    deploy()
