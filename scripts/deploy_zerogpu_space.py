"""
Push space/ (the Gradio version) to a Hugging Face Space and switch its
hardware to ZeroGPU — free for personal accounts (verified email, account
30+ days old), up to 2 such Spaces.

Run scripts/deploy_space.py first (or already have) so the v1/v2 model repos
this app loads from actually exist on the Hub.

    huggingface-cli login          # one-time, needs a write token
    python scripts/deploy_zerogpu_space.py --user YOUR_HF_USERNAME
"""

import argparse
import os

BASE = os.path.dirname(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--space-repo", default=None)
    args = ap.parse_args()

    from huggingface_hub import HfApi, SpaceHardware
    api = HfApi()
    try:
        who = api.whoami()
        print(f"logged in as {who['name']} (pro: {who.get('isPro')})")
    except Exception:
        raise SystemExit("Not logged in. Run:  python -m huggingface_hub.commands.huggingface_cli login")

    space_repo = args.space_repo or f"{args.user}/arrodeio-llm-gradio"
    path = os.path.join(BASE, "..", "space")

    print(f"creating {space_repo} with ZeroGPU hardware (must be set AT creation — "
          f"the default cpu-basic tier now requires PRO, even if you plan to "
          f"switch hardware right after)")
    api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True,
                    space_hardware=SpaceHardware.ZERO_A10G)
    print(f"pushing {path} -> {space_repo}")
    api.upload_folder(folder_path=path, repo_id=space_repo, repo_type="space",
                      ignore_patterns=["__pycache__", "*.pyc"])
    print(f"  pushed: https://huggingface.co/spaces/{space_repo}")

    print(f"\nSpace: https://huggingface.co/spaces/{space_repo}")
    print("First build takes a few minutes — watch the Space's 'Logs' tab.")


if __name__ == "__main__":
    main()
