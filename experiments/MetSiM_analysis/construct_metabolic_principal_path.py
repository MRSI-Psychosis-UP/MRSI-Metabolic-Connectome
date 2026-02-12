import numpy as np
from mrsitoolbox.tools.datautils import DataUtils
from mrsitoolbox.tools.debug import Debug
from os.path import join, split 
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting, image, datasets
import copy, sys, os, time, argparse, subprocess
from mrsitoolbox.graphplot.colorbar import ColorBar
import pandas as pd
from rich.progress import Progress
import seaborn as sns
import matplotlib as mpl
import networkx as nx
import argparse

from mrsitoolbox.connectomics.netfibre import NetFibre
from mrsitoolbox.connectomics.nettools import NetTools
from mrsitoolbox.connectomics.network import NetBasedAnalysis


from mrsitoolbox.graphplot.netplot import NetPlot
from dipy.viz import window
from mrsitoolbox.graphplot.colorbar import ColorBar


MAIN_FIBRE_OPACITY     = 1 #1
MAIN_FIBRE_CURVE_WIDTH = 1 #1 


colorbar    = ColorBar()
nettools    = NetTools()
dutils      = DataUtils()       
debug       = Debug()
nba         = NetBasedAnalysis()


FONTSIZE      = 16
manual_selection = False


parser = argparse.ArgumentParser(description="Process some input parameters.")
parser.add_argument('--group', type=str,default="LPN-Project") 
parser.add_argument('--parc', type=str, default="LFMIHIFIS",
                    help='Chimera parcellation scheme, valid choices: LFMIHIFIS,LFIIIIFIS, LFMIHIFIF. Default: LFMIHIFIS')
parser.add_argument('--scale', type=int, default=3,
                    help="Cortical parcellation scale (default: 3)")
parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1],
                    help="Overwrite existing parcellation (default: 0)")
parser.add_argument('--npert', type=int, default=50,
                    help='Number of perturbations as comma-separated integers (default: 50)')
parser.add_argument('--nperm', type=int, default=100,
                    help='Number of randgeom permutations (default: 10)')
parser.add_argument('--perplexity', type=float, default=30, help='Perplexity value for t-SNE (default: 30)')
parser.add_argument('--dimalg', type=str,default="pca_tsne",choices=['pca_tsne', 'umap'],help='Manifold discovery algorithm (default: pca-tsne)') 
parser.add_argument('--diag', type=str, default="controls",choices=['group', 'controls','patients'],
                help="Only inlcude controls, patients or all ('group'[default])")
parser.add_argument('--preproc', type=str, default="filtbiharmonic_pvcorr_GM",help="Preprocessing of orig MRSI files (default: filtbiharmonic_pvcorr_GM)")

parser.add_argument('--subject_id', type=str, help='subject id', default=None)
parser.add_argument('--session', type=str, help='recording session_patx', default=None)
parser.add_argument('--participants', type=str, default=None,
                    help="Path to TSV/CSV containing subject-session pairs to process in batch.")
parser.add_argument('--batch', type=str, default='off', choices=['off', 'all', 'file'],
                    help="Batch mode: 'all' uses all available subject-session pairs; 'file' uses --participants; 'off' processes a single couplet.")
parser.add_argument('--lobe', type=str, default="ctx",help="Cortical 'ctx' or subCortical 'subc' region ('ctx'[default])")
parser.add_argument('--lpath', type=int,default=12,help="Path length cutoff for DFS path search")


args        = parser.parse_args()
parc_scheme = args.parc 
group       = args.group
dimalg      = args.dimalg
npert       = args.npert
perplexity  = args.perplexity
scale       = args.scale
preproc_str = args.preproc.replace("filt","")
dimalg      = args.dimalg
diag        = args.diag
brainlobe   = args.lobe
nperm       = args.nperm
path_length = args.lpath


subject_id_patx     = args.subject_id
session_patx        = args.session
preproc_string      = args.preproc
participants_file   = args.participants

########################### parcel image: ###########################
atlas              = f"cubic-{scale}" if "cubic" in parc_scheme else f"chimera-{parc_scheme}-{scale}" 
gm_mask            = datasets.load_mni152_gm_mask().get_fdata().astype(bool)
parcel_mni_img_nii = nib.load(join(dutils.DEVDATAPATH,"atlas",f"{atlas}",f"{atlas}.nii.gz"))
parcel_mni_img_np  = parcel_mni_img_nii.get_fdata().astype(int)
########################### Load Group MeSiMs ###########################
con_dir        = join(dutils.BIDSDATAPATH, group, "derivatives", "group", "connectivity", "mrsi")
con_file       = f"{group}_atlas-chimera{parc_scheme}_scale{scale}_npert-{npert}_filt-{preproc_str}_desc-{diag}_connectivity_mrsi.npz"
data           = np.load(join(con_dir, con_file))
MeSiM_pop_avg  = data["MeSiM_pop_avg"]
MeSiM_list     = data["MeSiM_subj_list"]
subject_id_all = data["subject_id_list"]
session_all    = data["session_id_list"]
metabolites    = data["metabolites"]
metab_profiles_subj_list = data["metab_profiles_subj_list"]
df_net         = pd.DataFrame({"parcel_labels": data["parcel_labels_group"]})
df_net["parcel_names"] = data["parcel_names_group"]

# ------------------------- Optional batch driver -------------------------
def _read_pairs_from_file(path):
    # Accept TSV or CSV with flexible column names
    if path.lower().endswith('.tsv'):
        dfp = pd.read_csv(path, sep='\t')
    else:
        dfp = pd.read_csv(path)
    col_sub = None
    col_ses = None
    for c in dfp.columns:
        cl = c.lower()
        if col_sub is None and cl in ("subject", "subject_id", "participant_id", "sub", "id"):
            col_sub = c
        if col_ses is None and cl in ("session", "session_id", "ses"):
            col_ses = c
    if col_sub is None or col_ses is None:
        raise ValueError("Participants file must contain subject and session columns.")
    pairs = [(str(r[col_sub]), str(r[col_ses])) for _, r in dfp.iterrows()]
    return pairs

def _spawn_for_pairs(pairs):
    common = [
        f"--group={group}",
        f"--parc={parc_scheme}",
        f"--scale={scale}",
        f"--npert={npert}",
        f"--nperm={nperm}",
        f"--perplexity={perplexity}",
        f"--dimalg={dimalg}",
        f"--diag={diag}",
        f"--preproc={preproc_string}",
        f"--lobe={brainlobe}",
        f"--lpath={path_length}",
    ]
    for sid, ses in pairs:
        cmd = [sys.executable, __file__] + common + [f"--subject_id={sid}", f"--session={ses}"]
        print(f"[batch] Processing {sid}-{ses} ...")
        subprocess.run(cmd, check=True)

# Decide batch mode before doing single-run work (unless user passed a specific couplet)
if args.batch != 'off' and not (subject_id_patx and session_patx):
    if args.batch == 'file':
        if not participants_file:
            raise ValueError("--batch=file requires --participants to be provided.")
        pair_list = _read_pairs_from_file(participants_file)
    elif args.batch == 'all':
        pair_list = list(zip(subject_id_all.astype(str), session_all.astype(str)))
    else:
        pair_list = []
    if len(pair_list) == 0:
        print("[batch] No subject-session pairs found. Exiting.")
        sys.exit(0)
    _spawn_for_pairs(pair_list)
    sys.exit(0)


########################### Create Control Average ###########################
try:
    df = pd.read_csv(join(dutils.BIDSDATAPATH, group, f"controls_{group}_sessions_mrsi.tsv"), sep='\t')
except Exception as e:
    debug.warning("Could not find list of control subjects,reverting to group list")
    df = pd.read_csv(join(dutils.BIDSDATAPATH, group, f"best_{group}_sessions_mrsi.tsv"), sep='\t')


########################### Select MetSiM ###########################
if subject_id_patx and session_patx:
    row_id = np.where((subject_id_all == subject_id_patx) & (session_all == session_patx))[0]
    if len(row_id)>0:
        MetSiM_patx = MeSiM_list[row_id[0]]
        metab_profiles_patx = metab_profiles_subj_list[row_id[0]].mean(axis=-1)
        prefix=f"{subject_id_patx}-{session_patx}"
        resultdirname = f"{subject_id_patx}-{session_patx}"
    else:
        debug.error(f"{subject_id_patx}-{session_patx} not found")
        raise("Exit")
else:
    MetSiM_patx = MeSiM_pop_avg
    metab_profiles_patx = metab_profiles_subj_list.mean(axis=(0,-1))
    prefix=f"{group} {diag}-population average"
    resultdirname = f"{group}-{diag}-population_average"


########################### RESULTSDIR ###########################
resultdir = join(dutils.ANARESULTSPATH,"controls_vs_patients","path_disruptions",resultdirname)
os.makedirs(resultdir,exist_ok=True)
netpathdir = join(dutils.BIDSDATAPATH,group,"derivatives","group","networkpaths",atlas)

########################### Title ###########################
title_str = f"Construct Principal Path for {prefix}"
debug.title(title_str)
########################## Get MS Mode array  ##########################
df_net["MS_mode_1D"]       = nettools.dimreduce_matrix(MetSiM_patx,
                                                method       = dimalg,
                                                output_dim   = 1,
                                                perplexity   = perplexity)


MS_mode_dict = {label: mode for mode, label in zip(df_net["MS_mode_1D"].to_numpy(), 
                                                   df_net["parcel_labels"].to_numpy()
)}
MS_mode_map  = nettools.project_to_3dspace(df_net["MS_mode_1D"].to_numpy(),
                                            parcel_mni_img_np,
                                            df_net["parcel_labels"].to_list())

########################### Load default boundary conditions ###########################
filename = f"{group}_{atlas}_lh_{brainlobe}_desc-GMadjacency.npz"
adj_data_lh = np.load(join(netpathdir,filename))
adj_dict_lh = nettools.mat_to_graphdict(adj_data_lh["adjacency_mat"], labels=adj_data_lh["parcel_labels"], symmetric=True)

adj_data_rh = np.load(join(netpathdir,filename.replace("_lh_","_rh_")))
adj_dict_rh = nettools.mat_to_graphdict(adj_data_rh["adjacency_mat"], labels=adj_data_rh["parcel_labels"], symmetric=True)

print("LH",adj_dict_lh.keys())
print("RH",adj_dict_rh.keys())
# sys.exit()

if parc_scheme=="LFMIHIFIS":
    if brainlobe=="ctx":
        start_node_rh = 74
        stop_node_rh  = 20
        start_node_lh = 212
        stop_node_lh  = 158
    elif brainlobe=="subc":
        start_node_rh = 109
        stop_node_rh  = 2007
        start_node_lh = 247
        stop_node_lh  = 2005 

def _adjust_endpoints(start_node, stop_node, adj_dict, ms_dict, hemi_label):
    a = ms_dict.get(start_node, None)
    b = ms_dict.get(stop_node, None)
    if a is None or b is None:
        return start_node, stop_node
    base_diff = abs(b - a)
    best_start, best_stop, best_diff = start_node, stop_node, base_diff
    # Try moving start to an adjacent node
    for s2 in adj_dict.get(start_node, []):
        ms = ms_dict.get(s2, None)
        if ms is None:
            continue
        d = abs(b - ms)
        if d > best_diff:
            best_start, best_stop, best_diff = s2, stop_node, d
    # Try moving stop to an adjacent node
    for t2 in adj_dict.get(stop_node, []):
        ms = ms_dict.get(t2, None)
        if ms is None:
            continue
        d = abs(ms - a)
        if d > best_diff:
            best_start, best_stop, best_diff = start_node, t2, d
    if (best_start, best_stop) != (start_node, stop_node):
        debug.info(f"Adjusted {hemi_label} endpoints from ({start_node},{stop_node}) to ({best_start},{best_stop}) to increase |ΔMS| from {base_diff:.3f} to {best_diff:.3f}")
    else:
        debug.info(f"Kept {hemi_label} endpoints ({start_node},{stop_node}); |ΔMS|={base_diff:.3f}")
    return best_start, best_stop

# Adjust endpoints per hemisphere to maximize |ΔMS| by one-hop change
start_node_lh, stop_node_lh = _adjust_endpoints(start_node_lh, stop_node_lh, adj_dict_lh, MS_mode_dict, "LH")
start_node_rh, stop_node_rh = _adjust_endpoints(start_node_rh, stop_node_rh, adj_dict_rh, MS_mode_dict, "RH")

# Ensure MS Mode orientation goes from start -> stop increasing.
try:
    a_lh = MS_mode_dict.get(start_node_lh, None)
    b_lh = MS_mode_dict.get(stop_node_lh, None)
    a_rh = MS_mode_dict.get(start_node_rh, None)
    b_rh = MS_mode_dict.get(stop_node_rh, None)
    need_flip = False
    if a_lh is not None and b_lh is not None and a_lh > b_lh:
        need_flip = True
    if a_rh is not None and b_rh is not None and a_rh > b_rh:
        need_flip = True
    if need_flip:
        vals = df_net["MS_mode_1D"].to_numpy().astype(float)
        flipped = (vals - np.nanmax(vals)) * -1.0
        df_net["MS_mode_1D"] = flipped
        MS_mode_dict = {
            label: mode
            for mode, label in zip(df_net["MS_mode_1D"].to_numpy(), df_net["parcel_labels"].to_numpy())
        }
        MS_mode_map = nettools.project_to_3dspace(
            df_net["MS_mode_1D"].to_numpy(), parcel_mni_img_np, df_net["parcel_labels"].to_list()
        )
        debug.info("Flipped MS Mode orientation so start<stop across hemispheres")
except Exception as _e:
    debug.warning(f"Could not evaluate MS Mode flip condition: {_e}")

# Helper to ensure path files exist; if missing, generate via find_all_network_paths.py
def _ensure_paths(hemi, start_node, stop_node, path_length):
    fname = f"{group}_{atlas}_{hemi}_{brainlobe}_desc-start-{start_node}_stop-{stop_node}_l-{path_length}_netpaths.npy"
    ipath = join(netpathdir, fname)
    if not os.path.exists(ipath):
        debug.warning(f"Paths file not found: {ipath}. Generating...")
        script = join(dutils.DEVANALYSEPATH, "experiments", "MetSiM_analysis", "find_all_network_paths.py")
        cmd = [
            sys.executable, script,
            "--group", group,
            "--parc", parc_scheme,
            "--scale", str(scale),
            "--lobe", brainlobe,
            "--hemi", hemi,
            "--lpath", str(path_length),
            "--start", str(start_node),
            "--stop", str(stop_node),
        ]
        subprocess.run(cmd, check=True)
        if not os.path.exists(ipath):
            raise FileNotFoundError(f"Failed to generate paths file: {ipath}")
        debug.success(f"Generated {ipath}")
    else:
        debug.success(f"Found existing paths file: {ipath}")
    return np.load(ipath, allow_pickle=True)

def _ensure_paths_parallel(hemi_specs, path_length):
    """
    Generate path files for multiple hemispheres in parallel if missing.
    hemi_specs: list of tuples (hemi, start_node, stop_node)
    Returns a dict {hemi: np.ndarray(paths)} for each requested hemi.
    """
    procs = {}
    targets = {}
    # Prepare commands only for missing targets
    for hemi, start_node, stop_node in hemi_specs:
        fname = f"{group}_{atlas}_{hemi}_{brainlobe}_desc-start-{start_node}_stop-{stop_node}_l-{path_length}_netpaths.npy"
        ipath = join(netpathdir, fname)
        targets[hemi] = ipath
        if not os.path.exists(ipath):
            debug.warning(f"Paths file not found: {ipath}. Generating...")
            script = join(dutils.DEVANALYSEPATH, "experiments", "MetSiM_analysis", "find_all_network_paths.py")
            cmd = [
                sys.executable, script,
                "--group", group,
                "--parc", parc_scheme,
                "--scale", str(scale),
                "--lobe", brainlobe,
                "--hemi", hemi,
                "--lpath", str(path_length),
                "--start", str(start_node),
                "--stop", str(stop_node),
            ]
            procs[hemi] = subprocess.Popen(cmd)
        else:
            debug.success(f"Found existing paths file: {ipath}")
    # Wait for all started processes
    for hemi, proc in procs.items():
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Path generation failed for {hemi} (exit code {ret}).")
        if not os.path.exists(targets[hemi]):
            raise FileNotFoundError(f"Failed to generate paths file: {targets[hemi]}")
        debug.success(f"Generated {targets[hemi]}")
    # Load all requested
    out = {}
    for hemi, ipath in targets.items():
        out[hemi] = np.load(ipath, allow_pickle=True)
    return out

########################### Load all GM paths though Adj matrix ###########################
paths_dict = _ensure_paths_parallel([
    ("lh", start_node_lh, stop_node_lh),
    ("rh", start_node_rh, stop_node_rh),
], path_length)
paths_lh = paths_dict["lh"]
paths_rh = paths_dict["rh"]

################ Delete path that are not covered by MS mode map #################
# Build the whitelist of allowed labels (robust to dtype issues)
allowed_labels = set(
    pd.to_numeric(df_net["parcel_labels"], errors="coerce")
      .dropna()
      .astype(int)
      .tolist()
)
def filter_paths(paths, allowed):
    """Keep only those paths whose nodes are all in `allowed`."""
    return [p for p in paths if set(p).issubset(allowed)]

# Apply to both hemispheres
paths_rh = filter_paths(paths_rh, allowed_labels)
paths_lh = filter_paths(paths_lh, allowed_labels)

debug.success(f"LH-{brainlobe}: Found {len(paths_lh)} viable paths.")
debug.success(f"RH-{brainlobe}: Found {len(paths_rh)} viable paths.")

################################# Compute all costs #################################
centroids_mni = nettools.compute_centroids(parcel_mni_img_nii,
                                           df_net["parcel_labels"].to_numpy(),
                                           world=False)
df_net["centroids_mni"] = list(centroids_mni)
netfibre     = NetFibre(None,centroids_mni,None,None)



cost_noise_lh, grad_noise_lh, entr_noise_lh = netfibre.process_paths(
    paths_lh, MS_mode_dict, n_jobs=32)


cost_noise_rh, grad_noise_rh, entr_noise_rh = netfibre.process_paths(
    paths_rh, MS_mode_dict, n_jobs=32)




########################### Select  optimal Paths ############################
colormap_spectrum      = ColorBar().load_fsl_cmap()
centroids_mni_all = nettools.compute_centroids(parcel_mni_img_nii,np.unique(parcel_mni_img_np),world=False)
centroid_all_dict = dict()
for i,image_label in enumerate(np.unique(parcel_mni_img_np)):
    centroid_all_dict[image_label]  = centroids_mni_all[i]




best_path_rh_id = np.argmin(cost_noise_rh)
best_path_lh_id = np.argmin(cost_noise_lh)
best_path_rh = paths_rh[best_path_rh_id]
best_path_lh = paths_lh[best_path_lh_id]
label_to_index = dict(zip(df_net["parcel_labels"], df_net.index))
idx_path_lh = [label_to_index[lbl] for lbl in best_path_lh if lbl in label_to_index]
idx_path_rh = [label_to_index[lbl] for lbl in best_path_rh if lbl in label_to_index]



from scipy.interpolate import make_interp_spline
from scipy.stats import zscore
metab_profiles_lh = zscore(metab_profiles_patx[idx_path_lh])
metab_profiles_rh = zscore(metab_profiles_patx[idx_path_rh])



msmode_rh = np.array([MS_mode_dict[p] for p in best_path_rh if p in MS_mode_dict])
msmode_lh = np.array([MS_mode_dict[p] for p in best_path_lh if p in MS_mode_dict])

grad_lh,grad_rh = grad_noise_lh[best_path_lh_id],grad_noise_rh[best_path_rh_id]
entr_lh,entr_rh = entr_noise_lh[best_path_lh_id],entr_noise_rh[best_path_rh_id]
cost_lh,cost_rh = cost_noise_lh[best_path_lh_id],cost_noise_rh[best_path_rh_id]

debug.success(f"Found optimal path in LH-{brainlobe}: grad {grad_lh:.3f} H {entr_lh:.3f} cost {cost_lh:.3f}")
debug.success(f"Found optimal path in RH-{brainlobe}: grad {grad_rh:.3f} H {entr_rh:.3f} cost {cost_rh:.3f}")



################## Random Geometric Null Model ##################
# nperm = 1000 
msmode_null_lh = np.zeros((nperm, len(msmode_lh)))
msmode_null_rh = np.zeros((nperm, len(msmode_rh)))

with Progress() as progress:
    task = progress.add_task("[cyan]Generating null models...", total=2*nperm)
    for i in range(nperm):
        master_rng = np.random.default_rng(i)
        mesim_sim = nba.rand_geom_model_global(
            centroids=np.vstack(df_net["centroids_mni"].to_numpy()),
            mat_true=MetSiM_patx,
            bins=120,
            rng=master_rng
        )
        progress.update(task, advance=1)
        msi_rand_arr = nettools.dimreduce_matrix(
            mesim_sim * 2 - 1,
            method=dimalg,
            output_dim=1,
            perplexity=perplexity,
            scale_factor=255.0
        )
        msi_rand_1D_dict = dict(zip(df_net["parcel_labels"].to_numpy(), msi_rand_arr))
        msmode_null_lh[i] = np.array(
            [msi_rand_1D_dict[label] for label in best_path_lh if label in MS_mode_dict]
        )
        msmode_null_rh[i] = np.array(
            [msi_rand_1D_dict[label] for label in best_path_rh if label in MS_mode_dict]
        )
        progress.update(task, advance=1)


entr_lh_null = netfibre.shannon_entropy(msmode_null_lh)
entr_rh_null = netfibre.shannon_entropy(msmode_null_rh)
grad_lh_null = np.array([netfibre.total_cost(mode)[1] for mode in msmode_null_lh])
grad_rh_null = np.array([netfibre.total_cost(mode)[1] for mode in msmode_null_rh])


n_better_or_equal = np.sum((grad_lh <= grad_rh_null) &
                           (entr_lh_null >= entr_lh))
p_joint_LH = (n_better_or_equal + 1) / (nperm + 1)

n_better_or_equal = np.sum((grad_rh <= grad_rh_null) &
                           (entr_rh_null >= entr_rh))
p_joint_RH = (n_better_or_equal + 1) / (nperm + 1)
debug.info(f"LH-{brainlobe} path: p = {p_joint_LH}")
debug.info(f"RH-{brainlobe} path: p = {p_joint_RH}")


################## Save results ##################
node_names_lh = df_net.set_index("parcel_labels").loc[list(best_path_lh), "parcel_names"].to_list()
node_names_rh = df_net.set_index("parcel_labels").loc[list(best_path_rh), "parcel_names"].to_list()
node_xyz_lh   = np.vstack(df_net.set_index("parcel_labels").loc[list(best_path_lh), "centroids_mni"].to_list())
node_xyz_rh   = np.vstack(df_net.set_index("parcel_labels").loc[list(best_path_rh), "centroids_mni"].to_list())
rows = []
# helper to add one hemisphere
def add_hemi(hemi, node_labels, node_names, node_positions, ms_mode,
             gradient, entropy, pvalue):
    for lbl, name, pos, ms in zip(node_labels, node_names, node_positions, ms_mode):
        rows.append({
            "hemisphere": hemi,
            "node_label": lbl,
            "node_name":  name,
            "node_position": pos,   # (x,y,z) triple or list
            "ms_mode":    ms,
            "gradient":   gradient,
            "entropy":    entropy,
            "pvalue":     pvalue,
        })

# fill for LH and RH
add_hemi("LH", best_path_lh, node_names_lh, node_xyz_lh, msmode_lh,
         grad_lh, entr_lh, p_joint_LH)
add_hemi("RH", best_path_rh, node_names_rh, node_xyz_rh, msmode_rh,
         grad_rh, entr_rh, p_joint_RH)
outpath = join(
    resultdir,
    f"metabolic_ppath_{prefix}_atlas-{atlas}_nperm-{nperm}_"
    f"desc-{brainlobe}_start-{start_node_lh}_stop-{stop_node_lh}_"
    f"l-{path_length}.csv"
)
pd.DataFrame(rows).to_csv(outpath, index=False)
np_filepath = outpath.replace(".json",".npz")
np.savez(np_filepath,
         entr_lh_null = entr_lh_null,
         entr_rh_null = entr_rh_null,
         grad_lh_null = grad_lh_null,
         grad_rh_null = grad_rh_null,
         grad_rh_meas = grad_rh,
         grad_lh_meas = grad_rh,
         entr_rh_meas = entr_rh,
         entr_lh_meas = entr_lh,)

debug.success("Saved results to dir:",split(outpath)[0])

############################################################################
############################################################################
############################################################################
############################ Display Results ###############################
############################################################################
############################################################################
############################################################################


FONTSIZE = 16
centroids_world         = nettools.compute_centroids(parcel_mni_img_nii,
                                                     df_net["parcel_labels"].to_numpy(),
                                                     world=True)


def plot_optimal_paths(
    msmode_lh, msmode_rh,
    grad_lh, grad_rh,
    entr_lh, entr_rh,
    grad_lh_null, grad_rh_null,
    entr_lh_null, entr_rh_null,
    best_path_rh, best_path_lh,
    centroids_world,colormap_spectrum,
    brainlobe="",msmin=0,msmax=255,
):
    """
    Make a 2x3 figure: LH row, RH row.
    Col1: vertical simple-path graph (1..N) with parcel names.
    Col2: msmode values along optimal path with x=1..N.
    Col3: KDE plot of null distribution vs observed gradient/entropy.
    """
    # ------- Setup figure: 2 rows x 3 columns top + 1 row brains -------
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(nrows=3, ncols=4, height_ratios=[1.0, 1.0, 1.1], hspace=0.6, wspace=0.4)
    # Top two rows (2x4)
    axes = np.empty((2, 4), dtype=object)
    for r in range(2):
        for c in range(4):
            axes[r, c] = fig.add_subplot(gs[r, c])
    axs = axes.ravel()
    # Bottom row: 4 compact brain views under all four columns
    gs_brains = gs[2, :].subgridspec(1, 4, wspace=0.25)
    ax_brain_axial = fig.add_subplot(gs_brains[0, 0])
    ax_brain_sag_l = fig.add_subplot(gs_brains[0, 1])
    ax_brain_sag_r = fig.add_subplot(gs_brains[0, 2])
    ax_brain_cor   = fig.add_subplot(gs_brains[0, 3])
    # ------- Null distributions into DataFrame for seaborn -------
    df_lh = pd.DataFrame({
        "Gradient Null": grad_lh_null,
        "Entropy Null": entr_lh_null,
        "Hemisphere": "LH"
    })
    df_rh = pd.DataFrame({
        "Gradient Null": grad_rh_null,
        "Entropy Null": entr_rh_null,
        "Hemisphere": "RH"
    })
    # Common font sizes
    TICK_FONTSIZE = FONTSIZE - 4
    XLABEL_FONTSIZE = FONTSIZE - 2
    YLABEL_FONTSIZE = FONTSIZE - 1
    # ===== Column 1: vertical simple-path graphs with parcel names =====
    # colormap for nodes
    # msmin = float(np.nanmin([np.nanmin(msmode_lh), np.nanmin(msmode_rh)]))
    # msmax = float(np.nanmax([np.nanmax(msmode_lh), np.nanmax(msmode_rh)]))
    parcel_labels = df_net["parcel_labels"].to_numpy()
    parcel_names  = df_net["parcel_names"].to_numpy()
    def draw_path_graph(ax, path_labels, node_values, title, color):
        names = [
            parcel_names[np.where(parcel_labels == node_lbl)[0][0]]
            if np.any(parcel_labels == node_lbl) else ""
            for node_lbl in path_labels
        ]
        # ensure values align 1:1 with path labels
        if node_values is None or len(node_values) != len(path_labels):
            node_values = [MS_mode_dict.get(lbl, np.nan) for lbl in path_labels]
        n = len(names)
        G = nx.path_graph(n)
        pos = {i: (0.0, -i) for i in range(n)}  # vertical layout
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=400,
            node_color=node_values, cmap=colormap_spectrum, vmin=msmin, vmax=msmax,
        )
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=color, width=2)
        nx.draw_networkx_labels(
            G, pos,
            labels={i: str(i+1) for i in range(n)},
            font_color="white", font_size=FONTSIZE-6, ax=ax,
        )
        for i, name in enumerate(names):
            ax.text(0.2, -i, name, va="center", ha="left", fontsize=FONTSIZE-3)
        ax.set_xlim(-0.6, 8.0)
        ax.set_ylim(-n + 0.6, 0.6)
        ax.set_axis_off()
        ax.set_title(title, fontsize=FONTSIZE)
    draw_path_graph(axes[0, 0], best_path_lh, msmode_lh, f"LH-{brainlobe.upper()} Path", color="C0")
    draw_path_graph(axes[1, 0], best_path_rh, msmode_rh, f"RH-{brainlobe.upper()} Path", color="C1")
    # ===== Column 2: msmode path plots with numeric x (1..N) =====
    # LH
    ax_lh_msmode = axes[0, 1]
    x_lh = np.arange(1, len(msmode_lh) + 1)
    ax_lh_msmode.plot(x_lh, msmode_lh, marker="o", color="C0")
    ax_lh_msmode.set_title(f"LH-{brainlobe.upper()} Optimal Path", fontsize=FONTSIZE)
    step_lh = max(1, int(np.ceil(len(x_lh) / 12)))
    ax_lh_msmode.set_xticks(x_lh[::step_lh])
    ax_lh_msmode.set_xlim(0.5, len(x_lh) + 0.5)
    ax_lh_msmode.set_xlabel("Node index along path", fontsize=XLABEL_FONTSIZE)
    ax_lh_msmode.set_ylabel("MS Mode", fontsize=YLABEL_FONTSIZE)
    ax_lh_msmode.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax_lh_msmode.grid(True, alpha=0.3)
    # RH
    ax_rh_msmode = axes[1, 1]
    x_rh = np.arange(1, len(msmode_rh) + 1)
    ax_rh_msmode.plot(x_rh, msmode_rh, marker="o", color="C1")
    ax_rh_msmode.set_title(f"RH-{brainlobe.upper()} Optimal Path", fontsize=FONTSIZE)
    step_rh = max(1, int(np.ceil(len(x_rh) / 12)))
    ax_rh_msmode.set_xticks(x_rh[::step_rh])
    ax_rh_msmode.set_xlim(0.5, len(x_rh) + 0.5)
    ax_rh_msmode.set_xlabel("Node index along path", fontsize=XLABEL_FONTSIZE)
    ax_rh_msmode.set_ylabel("MS Mode", fontsize=YLABEL_FONTSIZE)
    ax_rh_msmode.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax_rh_msmode.grid(True, alpha=0.3)
    # ===== Column 4: metabolic profiles along path (CrPCr baseline) =====
    def draw_metab_profiles(ax, profiles, hemi_label):
        x = np.arange(1, profiles.shape[0] + 1)  # 1..N to match MSM indexing
        # reference baseline index
        idx_cr = np.where(metabolites == "CrPCr")[0]
        idx_cr = idx_cr[0] if len(idx_cr) > 0 else None
        # pretty labels for legend
        name_map = {"NAANAAG": "tNAA", "GPCPCh": "Cho", "CluGln": "Glx"}
        for j, metabolite in enumerate(metabolites):
            if metabolite == "CrPCr":
                continue
            y = profiles[:, j]
            if idx_cr is not None:
                y = y - profiles[:, idx_cr]
            x_smooth = np.linspace(x.min(), x.max(), 120)
            try:
                spline = make_interp_spline(x, y, k=3)
                y_smooth = spline(x_smooth)
                ax.plot(x_smooth, y_smooth, label=name_map.get(str(metabolite), str(metabolite)), linewidth=2)
            except Exception:
                ax.plot(x, y, label=name_map.get(str(metabolite), str(metabolite)), linewidth=2)
        step = max(1, int(np.ceil(len(x) / 12)))
        ax.set_xticks(x[::step])
        ax.set_xlim(0.5, len(x) + 0.5)
        ax.set_xlabel("Node index along path", fontsize=XLABEL_FONTSIZE)
        ax.set_ylabel("Z-scored MetabProfile", fontsize=XLABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=TICK_FONTSIZE, loc="best", ncol=1, frameon=False)
        ax.set_title(f"{hemi_label}-{brainlobe.upper()} Metabolites", fontsize=FONTSIZE)
    draw_metab_profiles(axes[0, 3], metab_profiles_lh, "LH")
    draw_metab_profiles(axes[1, 3], metab_profiles_rh, "RH")
    # ===== Column 2: KDE null vs observed =====
    def plot_joint_panel(ax, df_group, grad_meas, ent_meas, title, color_idx, pvalue, nperm):
        import matplotlib.patches as mpatches
        # Null KDE
        sns.kdeplot(
            data=df_group,
            x="Gradient Null", y="Entropy Null",
            fill=True, cmap="Greys",
            thresh=0.02, levels=50, ax=ax
        )
        # Observed point
        point_color = mpl.rcParams["axes.prop_cycle"].by_key()["color"][color_idx]
        obs = ax.scatter(
            grad_meas, ent_meas,
            color=point_color, edgecolor="black", s=70, marker="D", zorder=10,
            label=f"Observed p={pvalue:.3f}"
        )
        # Dashed lines
        ax.axvline(grad_meas, color=point_color, linestyle="--", lw=1)
        ax.axhline(ent_meas, color=point_color, linestyle="--", lw=1)
        # Axis limits normalized to [0,1]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Normalized MS Gradient", fontsize=XLABEL_FONTSIZE)
        ax.set_ylabel("Entropy", fontsize=YLABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        ax.grid(True, alpha=0.3)
        # Legend entries: RandGeom and Observed
        kde_patch = mpatches.Patch(color="lightgrey", alpha=0.6, label=f"RandGeom n={nperm}")
        ax.legend(handles=[kde_patch, obs], fontsize=FONTSIZE-4, loc="lower right", frameon=True)
    # LH KDE
    plot_joint_panel(axes[0, 2], df_lh, grad_lh, entr_lh, title="LH", color_idx=0, pvalue=p_joint_LH, nperm=nperm)
    axes[0, 2].set_title("LH Null vs Observed", fontsize=FONTSIZE)
    # RH KDE
    plot_joint_panel(axes[1, 2], df_rh, grad_rh, entr_rh, title="RH", color_idx=1, pvalue=p_joint_RH, nperm=nperm)
    axes[1, 2].set_title("RH Null vs Observed", fontsize=FONTSIZE)
    # ===== Bottom row: glass brain connectomes (only path nodes) =====
    label_to_index = {lbl: i for i, lbl in enumerate(parcel_labels)}
    def subset_for_path(path_labels):
        idx = [label_to_index[lbl] for lbl in path_labels if lbl in label_to_index]
        coords = centroids_world[idx]
        n = len(idx)
        adj = np.zeros((n, n))
        for k in range(n - 1):
            adj[k, k + 1] = 1
            adj[k + 1, k] = 1
        return adj, coords
    adj_lh, coords_lh = subset_for_path(best_path_lh)
    values_lh = np.array([
        MS_mode_dict[lbl]
        for lbl in best_path_lh
        if lbl in label_to_index and lbl in MS_mode_dict
    ])
    adj_rh, coords_rh = subset_for_path(best_path_rh)
    values_rh = np.array([
        MS_mode_dict[lbl]
        for lbl in best_path_rh
        if lbl in label_to_index and lbl in MS_mode_dict
    ])
    # Axial overlay: LH + RH
    display_axial = plotting.plot_connectome(
        adj_lh,
        node_coords=coords_lh,
        node_color=values_lh,
        node_size=33,
        edge_kwargs={"color": "black", "linewidth": 2.2},
        display_mode="z",
        axes=ax_brain_axial,
        annotate=False,
        black_bg=False,
        alpha=0.9,
        colorbar=False,
        node_kwargs={"cmap": colormap_spectrum, "vmin": msmin, "vmax": msmax},
    )
    display_axial.add_graph(
        adj_rh,
        coords_rh,
        node_color=values_rh,
        node_size=33,
        edge_kwargs={"color": "black", "linewidth": 2.2},
        node_kwargs={"cmap": colormap_spectrum, "vmin": msmin, "vmax": msmax},
    )
    ax_brain_axial.set_title("LH + RH", fontsize=FONTSIZE-1)
    # Sagittal left: LH only
    plotting.plot_connectome(
        adj_lh,
        node_coords=coords_lh,
        node_color=values_lh,
        node_size=33,
        edge_kwargs={"color": "black", "linewidth": 2.2},
        display_mode="l",
        axes=ax_brain_sag_l,
        annotate=False,
        black_bg=False,
        alpha=0.9,
        colorbar=False,
        node_kwargs={"cmap": colormap_spectrum, "vmin": msmin, "vmax": msmax},
    )
    ax_brain_sag_l.set_title("LH", fontsize=FONTSIZE-1)
    # Sagittal right: RH only
    plotting.plot_connectome(
        adj_rh,
        node_coords=coords_rh,
        node_color=values_rh,
        node_size=33,
        edge_kwargs={"color": "black", "linewidth": 2.2},
        display_mode="r",
        axes=ax_brain_sag_r,
        annotate=False,
        black_bg=False,
        alpha=0.9,
        colorbar=False,
        node_kwargs={"cmap": colormap_spectrum, "vmin": msmin, "vmax": msmax},
    )
    ax_brain_sag_r.set_title("RH", fontsize=FONTSIZE-1)
    # Coronal (both paths)
    display_cor = plotting.plot_connectome(
        adj_lh,
        node_coords=coords_lh,
        node_color=values_lh,
        node_size=33,
        edge_kwargs={"color": "black", "linewidth": 2.2},
        display_mode="y",
        axes=ax_brain_cor,
        annotate=False,
        black_bg=False,
        alpha=0.9,
        colorbar=False,
        node_kwargs={"cmap": colormap_spectrum, "vmin": msmin, "vmax": msmax},
    )
    display_cor.add_graph(
        adj_rh,
        coords_rh,
        node_color=values_rh,
        node_size=33,
        edge_kwargs={"color": "black", "linewidth": 2.2},
        node_kwargs={"cmap": colormap_spectrum, "vmin": msmin, "vmax": msmax},
    )
    ax_brain_cor.set_title("LH  RH", fontsize=FONTSIZE-1)
    # add suptitle
    fig.suptitle(prefix, fontsize=FONTSIZE + 2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, axs




fig, axs = plot_optimal_paths(
    msmode_lh, msmode_rh,
    grad_lh, grad_rh,
    entr_lh, entr_rh,
    grad_lh_null, grad_rh_null,
    entr_lh_null, entr_rh_null,
    best_path_rh, best_path_lh,
    centroids_world,colormap_spectrum,
    brainlobe=brainlobe
)

outpath = outpath.replace(".csv",".pdf")
plt.savefig(outpath)
# plt.show()





    
