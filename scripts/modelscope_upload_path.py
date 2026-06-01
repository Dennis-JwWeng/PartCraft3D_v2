#!/usr/bin/env python
import os
import sys
from pathlib import Path

from modelscope.hub.api import HubApi


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: modelscope_upload_path.py <local_path> <path_in_repo> <commit_message>", file=sys.stderr)
        return 2

    local_path = Path(sys.argv[1])
    path_in_repo = sys.argv[2]
    commit_message = sys.argv[3]
    repo_id = os.environ.get("REPO_ID", "zhenxinDiao/H3D1")
    repo_type = os.environ.get("REPO_TYPE", "dataset")
    revision = os.environ.get("MS_REVISION", "main")
    max_workers = int(os.environ.get("MAX_WORKERS", "1"))
    token = os.environ.get("MODELSCOPE_TOKEN")

    if not token:
        print("MODELSCOPE_TOKEN is required", file=sys.stderr)
        return 2

    api = HubApi()
    api.login(token)

    if local_path.is_dir():
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(local_path),
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            token=token,
            commit_message=commit_message,
            max_workers=max_workers,
            revision=revision,
            ignore_patterns=["**/.git/**", "**/__pycache__/**", "**/*.pyc"],
        )
    else:
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            token=token,
            commit_message=commit_message,
            revision=revision,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
