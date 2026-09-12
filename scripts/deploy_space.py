"""
Push the web demo to Hugging Face: the two model repos (v1, v2) it loads from,
and the Space itself (webapp/).

One-time setup:
    pip install -U huggingface_hub
    huggingface-cli login          # paste a token with "write" access
                                    # (huggingface.co/settings/tokens)

Usage:
    python scripts/deploy_space.py --user YOUR_HF_USERNAME              # all three
    python scripts/deploy_space.py --user YOUR_HF_USERNAME --only space  # just the Space
    python scripts/deploy_space.py --user YOUR_HF_USERNAME --only v1,v2  # just the models

Repo names (override with --v1-repo / --v2-repo / --space-repo if you want):
    <user>/arrodeio-gpt2-cordel        <- model/cordel-ft/            (~480 MB)
    <user>/arrodeio-tucano-cordel-lora <- model/cordel-sft-lora/      (~19 MB)
    <user>/arrodeio-llm                <- webapp/                     (the Space)

After pushing v1/v2, either rely on webapp/app.py's defaults matching these
names, or set V1_MODEL / V2_ADAPTER as Space secrets/variables if you used
different repo names.
"""

import argparse
import os

BASE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your Hugging Face username")
    ap.add_argument("--only", default="space,v1,v2",
                    help="comma list: space,v1,v2 (default: all)")
    ap.add_argument("--v1-repo", default=None)
    ap.add_argument("--v2-repo", default=None)
    ap.add_argument("--space-repo", default=None)
    ap.add_argument("--private", action="store_true", help="make the model repos private")
    args = ap.parse_args()
    targets = set(args.only.split(","))

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        who = api.whoami()
        print(f"logged in as {who['name']}")
    except Exception:
        raise SystemExit("Not logged in. Run:  huggingface-cli login")

    v1_repo = args.v1_repo or f"{args.user}/arrodeio-gpt2-cordel"
    v2_repo = args.v2_repo or f"{args.user}/arrodeio-tucano-cordel-lora"
    space_repo = args.space_repo or f"{args.user}/arrodeio-llm"

    if "v1" in targets:
        path = os.path.join(BASE, "..", "model", "cordel-ft")
        if not os.path.isdir(path):
            print(f"  !! {path} not found — run scripts/finetune.py first, skipping v1")
        else:
            print(f"pushing {path} -> {v1_repo} (model repo, ~480 MB, this takes a while)")
            api.create_repo(v1_repo, repo_type="model", private=args.private, exist_ok=True)
            api.upload_folder(folder_path=path, repo_id=v1_repo, repo_type="model")
            print(f"  done: https://huggingface.co/{v1_repo}")

    if "v2" in targets:
        path = os.path.join(BASE, "..", "model", "cordel-sft-lora")
        if not os.path.isdir(path):
            print(f"  !! {path} not found — download it from the Colab run first, skipping v2")
        else:
            print(f"pushing {path} -> {v2_repo} (LoRA adapter, ~19 MB)")
            api.create_repo(v2_repo, repo_type="model", private=args.private, exist_ok=True)
            api.upload_folder(folder_path=path, repo_id=v2_repo, repo_type="model")
            print(f"  done: https://huggingface.co/{v2_repo}")

    if "space" in targets:
        path = os.path.join(BASE, "..", "webapp")
        print(f"pushing {path} -> {space_repo} (Space)")
        api.create_repo(space_repo, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=path, repo_id=space_repo, repo_type="space",
                          ignore_patterns=["__pycache__", "*.pyc"])
        print(f"  done: https://huggingface.co/spaces/{space_repo}")
        print("  (first build takes a few minutes — check the Space's 'Logs' tab)")

        if v1_repo != f"{args.user}/arrodeio-gpt2-cordel" or v2_repo != f"{args.user}/arrodeio-tucano-cordel-lora":
            print(f"\n  Using non-default repo names — set these as Space variables:")
            print(f"    V1_MODEL={v1_repo}")
            print(f"    V2_ADAPTER={v2_repo}")


if __name__ == "__main__":
    main()
