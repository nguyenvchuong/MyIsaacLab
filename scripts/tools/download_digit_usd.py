#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to download Digit V4 USD file from Nucleus Server to local directory."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Download Digit V4 USD file from Nucleus Server.")
parser.add_argument(
    "--download_dir",
    type=str,
    default="./digit_assets",
    help="Directory where the USD file will be downloaded. Defaults to './digit_assets'.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

# Digit USD file path on Nucleus
digit_usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Agility/Digit/digit_v4.usd"

print(f"[INFO]: Downloading Digit V4 USD file from Nucleus Server...")
print(f"[INFO]: Source: {digit_usd_path}")
print(f"[INFO]: Destination directory: {args_cli.download_dir}")

try:
    # Download the file
    local_path = retrieve_file_path(digit_usd_path, download_dir=args_cli.download_dir, force_download=True)
    print(f"[INFO]: Successfully downloaded to: {local_path}")
    print(f"[INFO]: File size: {os.path.getsize(local_path) / (1024 * 1024):.2f} MB")
except FileNotFoundError as e:
    print(f"[ERROR]: File not found on Nucleus Server: {e}")
    print(f"[ERROR]: Make sure you have access to the Nucleus Server and Isaac Sim is properly configured.")
except RuntimeError as e:
    print(f"[ERROR]: Failed to download file: {e}")
    print(f"[ERROR]: Make sure the Nucleus Server is accessible and you have write permissions to: {args_cli.download_dir}")

# close sim app
simulation_app.close()
