#!/usr/bin/env python3
"""
Batch brain extraction with HD-BET over BIDS T1w files.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
from rich.progress import Progress, BarColumn, SpinnerColumn, TextColumn, TimeRemainingColumn


def import_mridata():
    """Import mridata.extract_metadata, adding repo root to sys.path if needed."""
    try:
        from tools.mridata import MRIData  # type: ignore
        return MRIData.extract_metadata
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        try:
            sys.path.append(str(repo_root))
            from tools.mridata import MRIData  # type: ignore
            return MRIData.extract_metadata
        except Exception:
            def _fallback_extract_metadata(filename):
                base = os.path.basename(str(filename))
                metadata = {}
                for key in ("sub", "ses", "acq"):
                    match = re.search(rf"{key}-([^_]+)", base)
                    metadata[key] = match.group(1) if match else None
                return metadata
            return _fallback_extract_metadata


extract_metadata = import_mridata()


def _emit(log_callback, message):
    if log_callback is not None:
        log_callback(message)


def _strip_nifti_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def prepare_hd_bet_env():
    """Return a subprocess environment that avoids the MKL/libgomp conflict."""
    env = os.environ.copy()
    env["MKL_THREADING_LAYER"] = "GNU"
    return env


def find_t1_files(bids_dir: Path, pattern: str):
    """Yield T1 files matching pattern under bids_dir, skipping derivatives folders."""
    for path in bids_dir.rglob(pattern):
        if "derivatives" in path.parts:
            continue
        if path.is_file():
            yield path


def build_outputs(bids_dir: Path, t1_path: Path, metadata: dict):
    """Return brain and brainmask outputs under derivatives/skullstrip."""
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
    """Run hd-bet on a single T1 image."""
    cmd = [hd_bet_bin, "-i", str(t1_path), "-o", str(out_brain), "-device", device]
    if disable_tta:
        cmd.append("--disable_tta")
    if verbose:
        print(f"[run] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        check=False,
        env=prepare_hd_bet_env(),
        stdout=None if verbose else subprocess.PIPE,
        stderr=None if verbose else subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        details = stderr or stdout or f"hd-bet exited with code {result.returncode}"
        raise RuntimeError(details)


def create_brainmask(out_brain: Path, out_mask: Path, verbose: bool):
    tmp_mask = out_mask.with_name("tmp_" + out_mask.name)
    cmd_bin = ["fslmaths", str(out_brain), "-bin", str(tmp_mask)]
    cmd_fill = ["fslmaths", str(tmp_mask), "-fillh", str(out_mask)]
    try:
        if verbose:
            print(f"[run] {' '.join(cmd_bin)}")
        subprocess.run(
            cmd_bin,
            check=True,
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.DEVNULL if not verbose else None,
        )
        if verbose:
            print(f"[run] {' '.join(cmd_fill)}")
        subprocess.run(
            cmd_fill,
            check=True,
            stdout=subprocess.DEVNULL if not verbose else None,
            stderr=subprocess.DEVNULL if not verbose else None,
        )
    finally:
        if tmp_mask.exists():
            tmp_mask.unlink()


def process_t1_file(
    bids_dir: Path,
    t1_path: Path,
    hd_bet_bin="hd-bet",
    device="cuda",
    disable_tta=False,
    verbose=False,
    overwrite=False,
    log_callback=None,
):
    meta = extract_metadata(str(t1_path))
    out_brain, out_mask = build_outputs(bids_dir, t1_path, meta)

    if out_brain.exists() and out_mask.exists() and not overwrite:
        _emit(log_callback, f"[skip] outputs exist: {out_brain} | {out_mask}")
        return {"brain": out_brain, "mask": out_mask, "status": "skipped"}

    _emit(log_callback, f"[proc] {t1_path} -> {out_brain}")
    tmp_input = None
    hdbet_input = t1_path
    temp_dir = None
    if t1_path.name.endswith(".nii.gz"):
        temp_dir = tempfile.TemporaryDirectory(prefix="hd_bet_input_")
        tmp_input = Path(temp_dir.name) / f"{_strip_nifti_suffix(t1_path.name)}.nii"
        _emit(log_callback, f"[tmp] {t1_path} -> {tmp_input}")
        img = nib.load(str(t1_path))
        nib.save(img, str(tmp_input))
        if not tmp_input.exists():
            raise FileNotFoundError(f"Failed to create temporary HD-BET input: {tmp_input}")
        hdbet_input = tmp_input
    try:
        run_hd_bet(
            t1_path=hdbet_input,
            out_brain=out_brain,
            hd_bet_bin=hd_bet_bin,
            device=device,
            disable_tta=disable_tta,
            verbose=verbose,
        )
    except Exception as exc:
        raise RuntimeError(
            f"HD-BET failed for input {hdbet_input} (exists={hdbet_input.exists()}): {exc}"
        ) from exc
    finally:
        if tmp_input and tmp_input.exists():
            tmp_input.unlink()
        if temp_dir is not None:
            temp_dir.cleanup()

    create_brainmask(out_brain, out_mask, verbose=verbose)
    return {"brain": out_brain, "mask": out_mask, "status": "processed"}


def process_t1_files(
    bids_dir: Path,
    t1_files,
    hd_bet_bin="hd-bet",
    device="cuda",
    disable_tta=False,
    verbose=False,
    overwrite=False,
    log_callback=None,
):
    results = []
    for t1_path in t1_files:
        results.append(
            process_t1_file(
                bids_dir=bids_dir,
                t1_path=Path(t1_path),
                hd_bet_bin=hd_bet_bin,
                device=device,
                disable_tta=disable_tta,
                verbose=verbose,
                overwrite=overwrite,
                log_callback=log_callback,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch HD-BET over BIDS T1w files.")
    parser.add_argument("--bids-dir", "-b", type=Path, required=True, help="BIDS root directory.")
    parser.add_argument(
        "--pattern",
        "-p",
        default="*T1w.nii*",
        help="Glob pattern for T1 files (relative, used with rglob). Default: *T1w.nii*",
    )
    parser.add_argument("--hd-bet-bin", default="hd-bet", help="Path to hd-bet executable. Default: hd-bet")
    parser.add_argument("--device", default="cuda", help="hd-bet device argument (e.g., cuda or cpu). Default: cuda")
    parser.add_argument("--disable-tta", action="store_true", help="Pass --disable_tta to hd-bet.")
    parser.add_argument("--verbose", action="store_true", help="Print commands and progress.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
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
            process_t1_file(
                bids_dir=bids_dir,
                t1_path=t1_path,
                hd_bet_bin=args.hd_bet_bin,
                device=args.device,
                disable_tta=args.disable_tta,
                verbose=args.verbose,
                overwrite=args.overwrite,
                log_callback=print if args.verbose else None,
            )
            progress.update(task, advance=1)


if __name__ == "__main__":
    main()
