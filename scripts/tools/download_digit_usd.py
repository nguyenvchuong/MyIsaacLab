#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Download Digit V4 asset folder (USD + sublayers + configs/materials) from Nucleus to local directory.

This fixes errors like:
  Could not load sublayer @configuration/digit_v4_robot_schema.usd@ ...
because the original script only downloaded digit_v4.usd (without dependencies).
"""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

# ----------------------------
# CLI
# ----------------------------
parser = argparse.ArgumentParser(description="Download Digit V4 asset folder from Nucleus Server.")
parser.add_argument(
    "--download_dir",
    type=str,
    default="./digit_assets",
    help="Local directory where the Digit folder will be downloaded. Defaults to './digit_assets'.",
)
parser.add_argument(
    "--force_download",
    action="store_true",
    help="If set, re-download files even if they already exist locally.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app (needed so omni.client is available)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------
# Everything else follows
# ----------------------------
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
import omni.client  # noqa: E402


def _is_dir_entry(entry) -> bool:
    """Heuristic: Nucleus entries that can have children are folders."""
    return bool(entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN)


def download_nucleus_folder_recursive(src_url: str, dst_dir: str, force: bool = False) -> None:
    """Recursively download a Nucleus folder (omniverse://...) to a local directory."""
    os.makedirs(dst_dir, exist_ok=True)

    result, entries = omni.client.list(src_url)
    if result != omni.client.Result.OK:
        raise RuntimeError(f"Failed to list Nucleus path: {src_url} (Result={result})")

    for entry in entries:
        name = entry.relative_path  # name of file/folder under src_url
        if not name:
            continue

        child_src = src_url.rstrip("/") + "/" + name
        child_dst = os.path.join(dst_dir, name)

        if _is_dir_entry(entry):
            download_nucleus_folder_recursive(child_src, child_dst, force=force)
        else:
            # Skip if exists and not forcing
            if (not force) and os.path.exists(child_dst) and os.path.getsize(child_dst) > 0:
                continue

            os.makedirs(os.path.dirname(child_dst), exist_ok=True)

            # omni.client.copy can copy from omniverse:// to local path
            copy_result = omni.client.copy(child_src, child_dst)
            if copy_result != omni.client.Result.OK:
                print(f"[WARN]: Failed to copy {child_src} -> {child_dst} (Result={copy_result})")


def main():
    # Nucleus Digit folder (NOT just the .usd file)
    digit_folder_url = f"{ISAAC_NUCLEUS_DIR}/Robots/Agility/Digit"
    download_root = os.path.abspath(args_cli.download_dir)
    local_digit_folder = os.path.join(download_root, "Digit")

    print("[INFO]: Downloading Digit V4 folder from Nucleus Server...")
    print(f"[INFO]: Source folder: {digit_folder_url}")
    print(f"[INFO]: Destination:   {local_digit_folder}")
    print(f"[INFO]: Force:         {args_cli.force_download}")

    download_nucleus_folder_recursive(digit_folder_url, local_digit_folder, force=args_cli.force_download)

    local_usd = os.path.join(local_digit_folder, "digit_v4.usd")
    if os.path.exists(local_usd):
        size_mb = os.path.getsize(local_usd) / (1024 * 1024)
        print(f"[INFO]: Done. Local USD: {local_usd}")
        print(f"[INFO]: digit_v4.usd size: {size_mb:.2f} MB")
    else:
        print("[ERROR]: Finished download, but digit_v4.usd was not found locally.")
        print("[ERROR]: Check that the Nucleus path exists and you have access permissions.")


if __name__ == "__main__":
    try:
        main()
    finally:
        # close sim app
        simulation_app.close()
