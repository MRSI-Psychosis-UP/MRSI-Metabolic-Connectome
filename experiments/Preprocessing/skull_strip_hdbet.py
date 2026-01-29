#!/usr/bin/env python3
"""
Batch brain extraction with HD-BET over BIDS T1w files.

Loops over T1w images matching a pattern within a BIDS root (skips derivatives),
runs `hd-bet` with `--save_bet_mask`, and writes:
- sub-<sub>_ses-<ses>_acq-<acq>_desc-brain_T1w.nii.gz
- sub-<sub>_ses-<ses>_acq-<acq>_desc-brainmask_T1w.nii.gz

Metadata fields are parsed from the input filename using `mridata.extract_metadata`.
"""

import argparse
import os
import subprocess
from pathlib import Path
import sys
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, SpinnerColumn


def import_mridata():
    """Import mridata.extract_metadata, adding repo root to sys.path if needed."""
    try:
        from tools.mridata import MRIData  # type: ignore
    except ModuleNotFoundError:
        # try relative to this script (MRSI-Metabolic-Connectome/tools)
        repo_root = Path(__file__).resolve().parents[2] / "MRSI-Metabolic-Connectome"
        sys.path.append(str(repo_root))
        from tools.mridata import MRIData  # type: ignore
    return MRIData.extract_metadata


extract_metadata = import_mridata()


def find_t1_files(bids_dir: Path, pattern: str):
    """Yield T1 files matching pattern under bids_dir, skipping derivatives folders."""
    for path in bids_dir.rglob(pattern):
        if "derivatives" in path.parts:
            continue
        if path.is_file():
            yield path


def build_outputs(bids_dir: Path, t1_path: Path, metadata: dict):
    """Return (brain_path, brainmask_path) under derivatives/skullstrip with normalized session."""
    sub = metadata.get("sub") or "unknown"
    ses = metadata.get("ses") or ""
    if ses and not ses.upper().startswith("V"):
        ses = f"V{ses.lstrip('vV')}"
    acq = metadata.get("acq")
    parts = [f"sub-{sub}"]
    if ses:
        parts.append(f"ses-{ses}")
    if acq:
        parts.append(f"acq-{acq}")
    base = "_".join(parts)
    deriv_dir = bids_dir / "derivatives" / "skullstrip" / f"sub-{sub}" / (f"ses-{ses}" if ses else "")
    deriv_dir.mkdir(parents=True, exist_ok=True)
    brain = deriv_dir / f"{base}_desc-brain_T1w.nii.gz"
    brainmask = deriv_dir / f"{base}_desc-brainmask_T1w.nii.gz"
    return brain, brainmask


def run_hd_bet(t1_path: Path, out_brain: Path, hd_bet_bin: str, device: str, disable_tta: bool, verbose: bool):
    """Run hd-bet with required args (brain only; mask will be computed via fslmaths)."""
    os.makedirs(os.path.split(out_brain)[0],exist_ok=True)
    cmd = [
        hd_bet_bin,
        "-i",
        str(t1_path),
        "-o",
        str(out_brain),
        "-device",
        device,
    ]
    if disable_tta:
        cmd.append("--disable_tta")
    if verbose:
        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description="Batch HD-BET over BIDS T1w files.")
    parser.add_argument("--bids-dir", "-b", type=Path, required=True, help="BIDS root directory.")
    parser.add_argument(
        "--pattern",
        "-p",
        default="*T1w.nii.gz",
        help="Glob pattern for T1 files (relative, used with rglob). Default: *T1w.nii.gz",
    )
    parser.add_argument("--hd-bet-bin", default="hd-bet", help="Path to hd-bet executable. Default: hd-bet")
    parser.add_argument("--device", default="cuda", help="hd-bet device argument (e.g., cuda or cpu). Default: cuda")
    parser.add_argument("--disable-tta", action="store_true", help="Pass --disable_tta to hd-bet.")
    parser.add_argument("--verbose", action="store_true", help="Print commands and progress.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing brain/brainmask outputs if present.",
    )
    args = parser.parse_args()

    bids_dir = args.bids_dir
    if not bids_dir.is_dir():
        raise SystemExit(f"BIDS dir not found: {bids_dir}")

    t1_files = list(find_t1_files(bids_dir, args.pattern))
    if not t1_files:
        raise SystemExit("No T1 files found.")

    with Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[green]HD-BET:", total=len(t1_files))
        for t1_path in t1_files:
            meta = extract_metadata(str(t1_path))
            sub = meta.get("sub") or "unknown"
            ses = meta.get("ses") or ""
            if ses and not ses.upper().startswith("V"):
                ses = f"V{ses.lstrip('vV')}"
            progress.update(task, description=f"[green]{sub} {ses}".strip())

            out_brain, out_mask = build_outputs(bids_dir, t1_path, meta)
            if out_brain.exists() and out_mask.exists() and not args.overwrite:
                if args.verbose:
                    print(f"[skip] outputs exist: {out_brain} | {out_mask}")
                progress.update(task, advance=1)
                continue
            if args.verbose:
                print(f"[proc] {t1_path} -> {out_brain}")
            # hd-bet writes mask alongside output with _mask suffix
            run_hd_bet(
                t1_path=t1_path,
                out_brain=out_brain,
                hd_bet_bin=args.hd_bet_bin,
                device=args.device,
                disable_tta=args.disable_tta,
                verbose=args.verbose,
            )
            # create brainmask from brain using fslmaths (bin + fillh)
            tmp_mask = out_mask.with_name("tmp_" + out_mask.name)
            cmd_bin = ["fslmaths", str(out_brain), "-bin", str(tmp_mask)]
            cmd_fill = ["fslmaths", str(tmp_mask), "-fillh", str(out_mask)]
            try:
                if args.verbose:
                    print(f"[run] {' '.join(cmd_bin)}")
                subprocess.run(
                    cmd_bin,
                    check=True,
                    stdout=subprocess.DEVNULL if not args.verbose else None,
                    stderr=subprocess.DEVNULL if not args.verbose else None,
                )
                if args.verbose:
                    print(f"[run] {' '.join(cmd_fill)}")
                subprocess.run(
                    cmd_fill,
                    check=True,
                    stdout=subprocess.DEVNULL if not args.verbose else None,
                    stderr=subprocess.DEVNULL if not args.verbose else None,
                )
            finally:
                if tmp_mask.exists():
                    tmp_mask.unlink()

            progress.update(task, advance=1)


if __name__ == "__main__":
    main()
