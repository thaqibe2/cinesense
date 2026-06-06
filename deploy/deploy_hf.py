"""
One-shot Hugging Face Space deployer for CineSense.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # paste a WRITE token from https://huggingface.co/settings/tokens
    python deploy/deploy_hf.py YOUR_HF_USERNAME

This creates (or reuses) a Gradio Space named <username>/cinesense and uploads
the whole project, preserving folder structure and binary models. Prints the URL.
"""
import sys, os
from huggingface_hub import HfApi, create_repo

def main():
    if len(sys.argv) < 2:
        print("Usage: python deploy/deploy_hf.py YOUR_HF_USERNAME [space_name]"); sys.exit(1)
    user = sys.argv[1]
    space = sys.argv[2] if len(sys.argv) > 2 else "cinesense"
    repo_id = f"{user}/{space}"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api = HfApi()
    api.upload_folder(
        repo_id=repo_id, repo_type="space", folder_path=root,
        ignore_patterns=["**/__pycache__/**", "reports/cache/**", ".git/**", "deploy/**"],
    )
    print("\nDeployed:  https://huggingface.co/spaces/" + repo_id)
    print("It will build for a minute; the app appears under the 'App' tab.")

if __name__ == "__main__":
    main()
