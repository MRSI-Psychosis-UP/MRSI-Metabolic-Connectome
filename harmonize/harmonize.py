import argparse
import os
import shutil
import time
from pathlib import Path
import math

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from rich.progress import Progress, BarColumn, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

from tools.debug import Debug
from tools.filetools import FileTools
from tools.datautils import DataUtils
from tools.mridata import MRIData
try:
    from neuroCombat import neuroCombat
except ImportError:
    from neuroCombat.neuroCombat import neuroCombat


debug = Debug()
dutils = DataUtils()
ftools = FileTools()

METABOLITE_LIST = ["NAANAAG", "GluGln", "GPCPCh", "CrPCr", "Ins"]
data_dict = {}


def _normalize_session(session):
    if session is None or pd.isna(session):
        return ""
    ses = str(session).strip()
    if not ses:
        return ""
    upper = ses.upper()
    if upper.startswith("T") and len(upper) > 1:
        return f"V{upper[1:]}"
    if upper.startswith("V") and len(upper) > 1:
        return f"V{upper[1:]}"
    if ses.isdigit():
        return f"V{ses}"
    return ses


def _resolve_columns(df):
    col_map = {col.lower(): col for col in df.columns}
    def pick(*names):
        for name in names:
            if name in col_map:
                return col_map[name]
        return None
    group_col = pick("group")
    sub_col = pick("participant_id", "subject_id", "subject", "sub", "id")
    ses_col = pick("session_id", "session", "ses")
    site_col = pick("site", "scanner", "scanner_id", "batch")
    sex_col = pick("sex", "gender")
    state_col = pick("state")
    age_col = pick("age")
    missing = [
        name
        for name, col in (
            ("group", group_col),
            ("participant_id", sub_col),
            ("session_id", ses_col),
            ("site", site_col),
            ("sex", sex_col),
            ("state", state_col),
            ("age", age_col),
        )
        if col is None
    ]
    if missing:
        raise ValueError(f"Participants file missing columns: {', '.join(missing)}")
    return group_col, sub_col, ses_col, site_col, sex_col, state_col, age_col


def _read_participants(path):
    sep = "\t" if path.lower().endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ValueError(f"No rows found in participants file: {path}")
    return df


def _load_qmask(group, met, res):
    qmask_dir = os.path.join(dutils.BIDSDATAPATH, group, "derivatives", "group", "qmask")
    qmask_path = os.path.join(
        qmask_dir,
         f"{group}_space-mni_met-{met}_res-{res}_desc-qmask_mrsi.nii.gz",
    )
    return nib.load(qmask_path).get_fdata().astype(bool)


def _coerce_age(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_sex(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _coerce_state(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _coerce_site(value):
    if pd.isna(value):
        return None
    cleaned = str(value).strip()
    return cleaned if cleaned else None


def _coerce_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _offset_nonzero(values):
    values = values.copy()
    nonzero = values != 0
    if np.any(nonzero):
        values[nonzero] += -values[nonzero].min()
    return values


def _print_combat_summary(estimates, met):
    if not estimates:
        return
    batches = estimates.get("batches")
    gamma = estimates.get("gamma.star")
    if gamma is None:
        gamma = estimates.get("gamma_hat")
    delta = estimates.get("delta.star")
    if delta is None:
        delta = estimates.get("delta_hat")
    if batches is None or gamma is None or delta is None:
        return
    gamma = np.array(gamma)
    delta = np.array(delta)
    if gamma.ndim != 2 or delta.ndim != 2:
        return
    table = Table(title=f"ComBat estimates - {met}")
    table.add_column("Batch")
    table.add_column("Median |gamma|", justify="right")
    table.add_column("Median |log(delta)|", justify="right")
    table.add_column("Mean |gamma|", justify="right")
    table.add_column("Mean |log(delta)|", justify="right")
    for idx, batch in enumerate(batches):
        g = gamma[idx]
        d = delta[idx]
        d = np.where(d <= 0, np.nan, d)
        med_abs_gamma = np.nanmedian(np.abs(g))
        med_abs_log_delta = np.nanmedian(np.abs(np.log(d)))
        mean_abs_gamma = np.nanmean(np.abs(g))
        mean_abs_log_delta = np.nanmean(np.abs(np.log(d)))
        table.add_row(
            str(batch),
            f"{med_abs_gamma:.4g}",
            f"{med_abs_log_delta:.4g}",
            f"{mean_abs_gamma:.4g}",
            f"{mean_abs_log_delta:.4g}",
        )
    debug.console.print(table)


def _load_weights(weights_path):
    weights = {}
    with open(weights_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("metabolite"):
                continue
            parts = [p for p in line.split() if p]
            if len(parts) < 4:
                continue
            metabolite = parts[1] if parts[0].isdigit() else parts[0]
            try:
                intercept = float(parts[-2])
                beta = float(parts[-1])
            except ValueError:
                continue
            weights[metabolite] = (beta, intercept)
    return weights


def _collect_paths(df, columns, metabolites, space, desc, option, res):
    group_col, sub_col, ses_col, site_col, sex_col, state_col, age_col = columns
    per_met = {
        met: {
            "paths": [],
            "group": [],
            "subject": [],
            "session": [],
            "site": [],
            "sex": [],
            "state": [],
            "age": [],
        }
        for met in metabolites
    }
    progress_columns = [
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ]
    with Progress(*progress_columns) as progress:
        task = progress.add_task("Collecting MRSI paths", total=len(df))
        for _, row in df.iterrows():
            group   = _coerce_text(row[group_col])
            subject = _coerce_text(row[sub_col])
            session = _normalize_session(row[ses_col])
            site    = _coerce_site(row[site_col])
            sex     = _coerce_sex(row[sex_col])
            state   = _coerce_state(row[state_col])
            age     = _coerce_age(row[age_col])
            # debug.info(group,subject,session,site,sex,state,age)
            if not group or not subject or not session:
                debug.warning("Skipping row with missing group/subject/session")
                progress.advance(task)
                continue
            if site is None or sex is None or state is None or age is None:
                debug.warning(f"Skipping {subject} {session}: missing site/sex/state/age")
                progress.advance(task)
                continue
            mridata = MRIData(subject, session, group)
            for met in metabolites:
                path = mridata.get_mri_filepath(
                    "mrsi",
                    space,
                    desc,
                    met=met,
                    option=option,
                    res=res,
                )
                if not path or not os.path.exists(path):
                    debug.warning(f"Missing MRSI file for {subject} {session} {met}")
                    continue
                per_met[met]["paths"].append(path)
                per_met[met]["group"].append(group)
                per_met[met]["subject"].append(subject)
                per_met[met]["session"].append(session)
                per_met[met]["site"].append(site)
                per_met[met]["sex"].append(sex)
                per_met[met]["state"].append(state)
                per_met[met]["age"].append(age)
            progress.advance(task)
    return per_met


def _load_data_for_met(per_met_entry, met, res, apply_mask):
    paths = per_met_entry["paths"]
    if not paths:
        return None
    data_cols = []
    ref_img = nib.load(paths[0])
    intersection_mask = None
    if apply_mask:
        for group in sorted(set(per_met_entry["group"])):
            group_mask = _load_qmask(group, met, res)
            intersection_mask = group_mask if intersection_mask is None else (intersection_mask & group_mask)
    for idx, path in enumerate(paths):
        data = nib.load(path).get_fdata().astype(np.float32)
        if apply_mask:
            data_cols.append(data[intersection_mask].ravel())
        else:
            data_cols.append(data.reshape(-1))
    if not data_cols:
        return None
    data_volumes = np.stack(data_cols, axis=1)
    filtered = {key: per_met_entry[key] for key in per_met_entry}
    covars = pd.DataFrame(
        {
            "site": filtered["site"],
            "sex": filtered["sex"],
            "state": filtered["state"],
            "age": filtered["age"],
        }
    )
    return {
        "data_volumes": data_volumes,
        "mask": intersection_mask if apply_mask else None,
        "covars": covars,
        "metadata": filtered,
        "ref_shape": ref_img.shape,
        "ref_header": ref_img.header,
    }


def main():

    parser = argparse.ArgumentParser(description="Harmonize MRSI maps with neuroCombat.")
    parser.add_argument(
        "--participants",
        default=None,
        help="TSV/CSV file with Group, participant_id, session_id, Site, Sex, Age columns.",
    )
    parser.add_argument(
        "--group",
        default="LPN-BioPsych-Project",
        help="Group name used to resolve default participants file if --participants is not set.",
    )
    parser.add_argument("--space", default="mni", help="MRSI space (default: mni).")
    parser.add_argument("--desc", default="signal", help="MRSI description (default: signal).")
    parser.add_argument(
        "--option",
        default="filtbiharmonic_pvcorr",
        help="MRSI preprocessing option suffix (default: filtbiharmonic_pvcorr).",
    )
    parser.add_argument("--res", type=int, default=5, help="MRSI resolution in mm (default: 5).")
    parser.add_argument(
        "--metabolites",
        nargs="+",
        default=METABOLITE_LIST,
        help="List of metabolites to harmonize.",
    )
    parser.add_argument(
        "--apply-mask",
        action="store_true",
        help="Apply an MNI brain mask before harmonization (default: off).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing harmonized outputs if present.",
    )
    parser.add_argument(
        "--save-before-combat",
        action="store_true",
        help="Save pre-harmonization volumes in a before_combat subfolder.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to weights file to bypass ComBat and apply fixed scaling/offset per metabolite.",
    )
    parser.add_argument(
        "--weights-as-input",
        action="store_true",
        help="If set, apply weights first and then run ComBat on the transformed inputs.",
    )
    args = parser.parse_args()
    weights = _load_weights(args.weights) if args.weights else None

    if args.participants:
        participants_path = Path(args.participants)
    else:
        if not args.group:
            raise SystemExit("--group is required when --participants is not specified.")
        participants_path = Path(dutils.BIDSDATAPATH) / args.group / "participants_allsessions.tsv"

    df = _read_participants(str(participants_path))
    columns = _resolve_columns(df)
    global data_dict
    data_dict = {"participants": str(participants_path), "metabolites": {}}
    per_met = _collect_paths(
        df,
        columns,
        args.metabolites,
        args.space,
        args.desc,
        args.option,
        args.res,
    )

    

    qc_copied = set()
    for met in args.metabolites:
        met_entry = per_met.get(met)
        if not met_entry or not met_entry["paths"]:
            debug.warning(f"No data found for metabolite {met}")
            continue

        loaded = _load_data_for_met(met_entry, met, args.res, args.apply_mask)
        if loaded is None:
            debug.warning(f"Skipping metabolite {met}: no usable data")
            continue
        if args.apply_mask and loaded["mask"] is not None:
            qmask_dir = os.path.join(dutils.BIDSDATAPATH, args.group, "derivatives", "group", "qmask")
            os.makedirs(qmask_dir, exist_ok=True)
            qmask_path = os.path.join(
                qmask_dir,
                f"{args.group}_space-mni_met-{met}_res-{args.res}_desc-qmask_mrsi.nii.gz",
            )
            if not os.path.exists(qmask_path) or args.overwrite:
                ftools.save_nii_file(loaded["mask"].astype(np.uint8), qmask_path, header=loaded["ref_header"])

        if weights is not None:
            if met not in weights:
                debug.warning(f"No weights found for {met}; skipping.")
                continue
            beta, intercept = weights[met]
            adjusted = loaded["data_volumes"].copy()
            site_vals = pd.to_numeric(pd.Series(loaded["metadata"]["site"]), errors="coerce").to_numpy()
            for idx, site_val in enumerate(site_vals):
                if site_val == 1:
                    adjusted[:, idx] = adjusted[:, idx] * beta + intercept
            if not args.weights_as_input:
                harmonized = {"data": adjusted, "estimates": None}
                debug.success(f"Applied weights for {met}")
            else:
                start = time.time()
                harmonized = neuroCombat(
                    dat=adjusted,
                    covars=loaded["covars"],
                    batch_col="site",
                    categorical_cols=["sex","state"],
                    continuous_cols=["age"],
                    eb=True,
                    #ref_batch=1,
                    parametric=False,
                    mean_only=False,
                )
                debug.success(f"Harmonized {met} (weights as input) in {round(time.time() - start, 2)}s")
                _print_combat_summary(harmonized.get("estimates") if isinstance(harmonized, dict) else None, met)
        else:
            start = time.time()
            harmonized = neuroCombat(
                dat=loaded["data_volumes"],
                covars=loaded["covars"],
                batch_col="site",
                categorical_cols=["sex"],
                continuous_cols=["age"],
                eb=True,
                #ref_batch=1,
                parametric=False,
                mean_only=False,
            )
            debug.success(f"Harmonized {met} in {round(time.time() - start, 2)}s")
            _print_combat_summary(harmonized.get("estimates") if isinstance(harmonized, dict) else None, met)

        data_dict["metabolites"][met] = {
            "data_volumes": loaded["data_volumes"],
            "mask": loaded["mask"],
            "covars": loaded["covars"],
            "metadata": loaded["metadata"],
            "harmonized": harmonized,
        }

        harmonized_data = harmonized["data"] if isinstance(harmonized, dict) else harmonized
        if harmonized_data is None:
            debug.warning(f"No harmonized data returned for {met}")
            continue
        if harmonized_data.shape != loaded["data_volumes"].shape:
            if harmonized_data.T.shape == loaded["data_volumes"].shape:
                harmonized_data = harmonized_data.T
            else:
                debug.warning(f"Unexpected harmonized shape for {met}: {harmonized_data.shape}")
                continue


        # Collect data for distribution plots
        metadata = loaded["metadata"]
        sites = sorted(set(metadata["site"]))
        before_by_site = {}
        after_by_site = {}
        for site in sites:
            site_indices = [i for i, s in enumerate(metadata["site"]) if s == site]
            if not site_indices:
                continue
            before_vals = []
            after_vals = []
            for idx in site_indices:
                before_vals.append(_offset_nonzero(loaded["data_volumes"][:, idx]))
                after_vals.append(_offset_nonzero(harmonized_data[:, idx]))
            if not before_vals or not after_vals:
                continue
            before_arr = np.concatenate(before_vals)
            after_arr = np.concatenate(after_vals)
            if before_arr.size == 0 or after_arr.size == 0:
                continue
            before_by_site[site] = before_arr
            after_by_site[site] = after_arr

        ref_shape = loaded["ref_shape"]
        ref_header = loaded["ref_header"]
        if ref_shape is None or ref_header is None:
            debug.warning(f"Missing reference header/shape for {met}")
            continue

        # Collect data for distribution plots
        for idx, subject in enumerate(metadata["subject"]):
            session = metadata["session"][idx]
            src_group = metadata["group"][idx]
            src_mridb = MRIData(subject, session, src_group)
            dst_mridb = MRIData(subject, session, args.group)
            outpath = dst_mridb.get_mri_filepath(
                "mrsi",
                "mni",
                "signal",
                met=met,
                option=args.option,
                res=args.res,
                construct_path=True,
            )
            os.makedirs(os.path.dirname(outpath), exist_ok=True)
            if os.path.exists(outpath) and not args.overwrite:
                continue
            if args.apply_mask and loaded["mask"] is not None:
                volume = np.zeros(ref_shape, dtype=np.float32)
                intersection_mask = loaded["mask"]
                volume[intersection_mask] = harmonized_data[:, idx]
            else:
                volume = harmonized_data[:, idx].reshape(ref_shape)
            nonzero_mask = volume != 0
            if np.any(nonzero_mask):
                volume[nonzero_mask] += -volume[nonzero_mask].min()
            if not os.path.exists(outpath) or args.overwrite:
                ftools.save_nii_file(volume, outpath, header=ref_header)

            crlb_src = src_mridb.get_mri_filepath(
                "mrsi",
                "mni",
                "crlb",
                met=met,
                res=args.res,
            )
            crlb_dst = dst_mridb.get_mri_filepath(
                "mrsi",
                "mni",
                "crlb",
                met=met,
                res=args.res,
                construct_path=True,
            )
            # debug.info("Copying",os.path.exists(crlb_src))
            if crlb_src and os.path.exists(crlb_src):
                os.makedirs(os.path.dirname(crlb_dst), exist_ok=True)
                if not os.path.exists(crlb_dst) or args.overwrite:
                    shutil.copy2(crlb_src, crlb_dst)

            qc_key = (subject, session)
            if qc_key not in qc_copied:
                for desc in ("snr", "fwhm"):
                    qc_src = src_mridb.get_mri_filepath(
                        "mrsi",
                        "mni",
                        desc,
                        met=None,
                        option=args.option,
                        res=args.res,
                    )
                    qc_dst = dst_mridb.get_mri_filepath(
                        "mrsi",
                        "mni",
                        desc,
                        met=None,
                        option=args.option,
                        res=args.res,
                        construct_path=True,
                    )
                    if qc_src and os.path.exists(qc_src):
                        os.makedirs(os.path.dirname(qc_dst), exist_ok=True)
                        if not os.path.exists(qc_dst) or args.overwrite:
                            shutil.copy2(qc_src, qc_dst)
                qc_copied.add(qc_key)

        if args.save_before_combat:
            before_dir = os.path.join(os.path.dirname(outpath), "before_combat")
            os.makedirs(before_dir, exist_ok=True)
            before_path = os.path.join(before_dir, f"OG_{os.path.basename(outpath)}")
            if args.apply_mask and loaded["mask"] is not None:
                before_volume = np.zeros(ref_shape, dtype=np.float32)
                intersection_mask = loaded["mask"]
                if weights is not None and args.weights_as_input:
                    before_volume[intersection_mask] = adjusted[:, idx]
                else:
                    before_volume[intersection_mask] = loaded["data_volumes"][:, idx]
            else:
                if weights is not None and args.weights_as_input:
                    before_volume = adjusted[:, idx].reshape(ref_shape)
                else:
                    before_volume = loaded["data_volumes"][:, idx].reshape(ref_shape)
                before_nonzero = before_volume != 0
                if np.any(before_nonzero):
                    before_volume[before_nonzero] += -before_volume[before_nonzero].min()
                if not os.path.exists(before_path) or args.overwrite:
                    ftools.save_nii_file(before_volume, before_path, header=ref_header)

        if before_by_site:
            n_sites = len(before_by_site)
            ncols = 2
            site0_key = None
            site1_key = None
            for key in before_by_site:
                try:
                    if int(key) == 0:
                        site0_key = key
                    elif int(key) == 1:
                        site1_key = key
                except (TypeError, ValueError):
                    if str(key).strip() == "0":
                        site0_key = key
                    elif str(key).strip() == "1":
                        site1_key = key
            extra_row = 1 if site0_key is not None and site1_key is not None else 0
            nrows_sites = math.ceil(n_sites / ncols)
            nrows = nrows_sites + extra_row
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows), squeeze=False)
            for idx, site in enumerate(sorted(before_by_site.keys())):
                ax = axes[idx // ncols][idx % ncols]
                before_vals = before_by_site[site]
                after_vals = after_by_site[site]
                bins = np.histogram_bin_edges(
                    np.concatenate([before_vals, after_vals]),
                    bins=80,
                )
                ax.hist(before_vals, bins=bins, density=True, alpha=0.45, label="before")
                ax.hist(after_vals, bins=bins, density=True, alpha=0.45, label="after")
                ax.set_title(f"Site {site}")
                ax.legend()
            for idx in range(n_sites, nrows_sites * ncols):
                axes[idx // ncols][idx % ncols].axis("off")
            if extra_row:
                before_ax = axes[nrows - 1][0]
                after_ax = axes[nrows - 1][1]
                before_vals_0 = before_by_site[site0_key]
                before_vals_1 = before_by_site[site1_key]
                after_vals_0 = after_by_site[site0_key]
                after_vals_1 = after_by_site[site1_key]
                before_bins = np.histogram_bin_edges(
                    np.concatenate([before_vals_0, before_vals_1]),
                    bins=80,
                )
                after_bins = np.histogram_bin_edges(
                    np.concatenate([after_vals_0, after_vals_1]),
                    bins=80,
                )
                before_ax.hist(before_vals_0, bins=before_bins, density=True, alpha=0.45, label="site 0 before")
                before_ax.hist(before_vals_1, bins=before_bins, density=True, alpha=0.45, label="site 1 before")
                before_ax.set_title("Before: site 0 vs site 1")
                before_ax.legend()
                before_ax.set_xscale("log")
                after_ax.hist(after_vals_0, bins=after_bins, density=True, alpha=0.45, label="site 0 after")
                after_ax.hist(after_vals_1, bins=after_bins, density=True, alpha=0.45, label="site 1 after")
                after_ax.set_title("After: site 0 vs site 1")
                after_ax.legend()
            fig.suptitle(f"Distribution by site - {met}")
            plot_dir = os.path.join(dutils.ANARESULTSPATH, "harmonize")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{args.group}_met-{met}_site-distributions.png")
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            debug.success(f"Saved distribution plot to {plot_path}")

        


if __name__ == "__main__":
    main()
