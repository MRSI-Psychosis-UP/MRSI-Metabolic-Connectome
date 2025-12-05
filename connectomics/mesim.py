import numpy as np
from scipy.stats import linregress, spearmanr
from concurrent.futures import ProcessPoolExecutor
import warnings, copy, time
from rich.progress import Progress,track
from tools.debug import Debug
from sklearn.linear_model import LinearRegression

debug  = Debug()


class MeSiM(object):
    def __init__(self):
        """Helper class to compute metabolic similarity matrices."""

    def __compute_metsim(
        self,
        mrsirand,
        parcel_mrsi_np,
        parcel_header_dict,
        parcel_label_ids_ignore=None,
        N_PERT=50,
        corr_mode="spearman",
        rescale="mean",
        n_permutations=10,
        return_parcel_conc=True,
        n_proc=16,
        hide_progress  = False,
        show_elapsed_t = True,
        progress_msg   = None,
    ):
        """
        Compute parcel-wise metabolic similarity matrix from perturbed MRSI data.

        Parameters
        ----------
        mrsirand : Randomize
            Randomized MRSI generator that provides noisy 4D volumes.
        parcel_mrsi_np : np.ndarray
            Parcellation volume in MRSI space.
        parcel_header_dict : dict
            Parcel metadata keyed by parcel id.
        parcel_label_ids_ignore : list, optional
            Parcel ids to skip in similarity computation.
        N_PERT : int
            Number of perturbations to draw.
        corr_mode : str
            Correlation mode ('spearman' or 'pearson').
        rescale : str
            Rescaling strategy for intensities before parcellation.
        n_permutations : int
            Reserved for future use; kept for API compatibility.
        return_parcel_conc : bool
            Whether to return parcel concentrations array.
        n_proc : int
            Number of worker processes.
        """
        parcel_label_ids_ignore = parcel_label_ids_ignore or []
        parcel_concentrations = None
        self.metabolites = mrsirand.metabolites
        self.npert = N_PERT
        for _ in track(range(N_PERT), description="Parcellate MRSI maps...",disable=hide_progress):
            met_image4D_data = mrsirand.sample_noisy_img4D()
            parcel_concentrations = self.parcellate_vectorized(
                met_image4D_data,
                parcel_mrsi_np,
                parcel_header_dict,
                rescale=rescale,
                parcel_concentrations=parcel_concentrations,
            )

        metsimatrix, pvalue = self._compute_metsim_parallel(
            parcel_concentrations,
            parcel_label_ids_ignore,
            corr_mode      = corr_mode,
            n_permutations = n_permutations,
            n_workers      = n_proc,
            show_elapsed_t = show_elapsed_t,
            progress_msg   = progress_msg,
        )

        parcel_concentrations_np = None
        if return_parcel_conc:
            parcel_concentrations = None
            for _ in track(range(N_PERT), description="Retrieve MRSI levels..."):
                met_image4D_data = mrsirand.sample_noisy_img4D()
                parcel_concentrations = self.parcellate_vectorized(
                    met_image4D_data,
                    parcel_mrsi_np,
                    parcel_header_dict,
                    rescale="raw",
                    parcel_concentrations=parcel_concentrations,
                )
            parcel_concentrations_np = self.map_mrsi_to_parcel(parcel_concentrations)
        return metsimatrix, pvalue, parcel_concentrations_np


    def compute_metsim(self, mrsirand, parcel_mrsi_np, parcel_header_dict, parcel_label_ids_ignore=None,
                                        N_PERT=50, corr_mode="spearman", rescale="mean", n_permutations=10,
                                        n_proc=16, leave_one_out=False):
        """
        Public entry point to compute metabolic similarity matrices.

        Parameters
        ----------
        mrsirand : Randomize
            Noise sampler that returns perturbed 4D metabolite volumes.
        parcel_mrsi_np : np.ndarray
            Parcellation labels in MRSI space.
        parcel_header_dict : dict
            Parcel metadata keyed by parcel id.
        parcel_label_ids_ignore : list, optional
            Parcel ids to skip when computing correlations.
        N_PERT : int
            Number of perturbations to draw.
        corr_mode : str
            Correlation metric, "spearman" (default) or "pearson".
        rescale : str
            Rescaling strategy passed to `parcellate_vectorized`.
        n_permutations : int
            Reserved for future use; kept for API compatibility.
        n_proc : int
            Number of worker processes for similarity computation.
        leave_one_out : bool
            If True, also returns matrices recomputed while leaving out each metabolite.

        Returns
        -------
        tuple
            metsimatrix, pvalue, parcel_concentrations, metsimatrix_leave_out (or None).
        """
        metsimatrix, pvalue, parcel_concentrations = self.__compute_metsim(
            mrsirand,
            parcel_mrsi_np,
            parcel_header_dict,
            parcel_label_ids_ignore=parcel_label_ids_ignore,
            N_PERT=N_PERT,
            corr_mode=corr_mode,
            rescale=rescale,
            n_permutations=n_permutations,
            return_parcel_conc=True,
            n_proc=n_proc,
        )
        metsimatrix_leave_out = None
        if leave_one_out:
            parcel_label_ids_ignore = parcel_label_ids_ignore or []
            metabolites = list(mrsirand.metabolites)
            metsimatrix_arr = []
            for metabolite_to_remove in metabolites:
                progress_msg = f"Leave-out {metabolite_to_remove}"
                filtered_metabolites = [met for met in metabolites if met != metabolite_to_remove]
                if not filtered_metabolites:
                    continue
                rand_subset = mrsirand.subset(filtered_metabolites)
                metsimatrix, pvalue,_    = self.__compute_metsim(rand_subset,parcel_mrsi_np,parcel_header_dict,
                                                                 parcel_label_ids_ignore,N_PERT,
                                                                corr_mode = corr_mode,
                                                                rescale   = rescale,
                                                                return_parcel_conc=False,
                                                                n_proc         = n_proc,
                                                                hide_progress  = False,
                                                                show_elapsed_t = False,
                                                                progress_msg   = progress_msg)
                metsimatrix[pvalue>0.05/N_PERT] = 0
                metsimatrix_arr.append(metsimatrix)
            metsimatrix_leave_out=  np.array(metsimatrix_arr)

        return metsimatrix, pvalue, parcel_concentrations, metsimatrix_leave_out


    def map_mrsi_to_parcel(self,parcel_concentrations):
        """
        Convert parcel concentration dict to ndarray [parcel, metabolite, perturbation].

        Parameters
        ----------
        parcel_concentrations : dict
            Keys are parcel ids, values are flat lists of concentrations.

        Returns
        -------
        np.ndarray
            Array indexed as [parcel, metabolite, perturbation].
        """
        n_parcels                = len(list(parcel_concentrations.keys()))
        parcel_concentrations_np = np.zeros([n_parcels,len(self.metabolites),self.npert])
        for idx,parcel_id in enumerate(parcel_concentrations):
            array = np.array(parcel_concentrations[parcel_id])
            parcel_concentrations_np[idx] = self.extract_metabolite_per_parcel(array)
        return parcel_concentrations_np


    def parcellate_vectorized(self,met_image4D_data, parcel_image3D, parcel_header_dict, rescale="zscore", parcel_concentrations=None):
        """
        Update or create parcel_concentrations with average metabolite concentrations.
        
        Parameters:
        - met_image4D_data: 4D numpy array of metabolite concentrations.
        - parcel_image3D: 3D numpy array mapping voxels to parcels.
        - parcel_header_dict: Dictionary with parcel IDs as keys.
        - rescale: str, whether to rescale concentrations by their mean.
        - parcel_concentrations: Optional dictionary to update with new concentrations.
        
        Returns:
        - Updated or new dictionary with parcel IDs as keys and lists of average concentrations as values.
        """
        if isinstance(parcel_header_dict,dict):
            parcel_ids_list = list(parcel_header_dict.keys())
        elif isinstance(parcel_header_dict,list):
            parcel_ids_list = copy.deepcopy(parcel_header_dict)

        n_metabolites   = met_image4D_data.shape[0]

        if rescale=="mean":
            for idm in range(n_metabolites):
                mean_val = np.mean(met_image4D_data[idm, ...])
                met_image4D_data[idm, ...] /= mean_val if mean_val != 0 else 1
        elif rescale=="zscore":
            for idm in range(n_metabolites):
                mean_val = np.mean(met_image4D_data[idm, ...])
                std_val  = np.std(met_image4D_data[idm, ...])
                if mean_val != 0 and  std_val != 0: 
                    met_image4D_data[idm, ...] = (met_image4D_data[idm, ...]-mean_val)/std_val
        elif rescale=="raw":
            pass

        if parcel_concentrations is None:
            parcel_concentrations = {}

        for parcel_id in parcel_ids_list:
            if parcel_id==0:continue
            parcel_mask = parcel_image3D == parcel_id
            for met_idx in range(n_metabolites):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    metabolite_image = met_image4D_data[met_idx, ...]
                    if parcel_mask.sum() != 0:
                        avg_concentration = np.nanmean(metabolite_image[parcel_mask])
                    else:
                        avg_concentration = 0
                    if parcel_id not in parcel_concentrations:
                        parcel_concentrations[parcel_id] = []
                    parcel_concentrations[parcel_id].append(avg_concentration)

        return parcel_concentrations

    def _compute_metsim_parallel(self, parcel_concentrations, 
                                 parcel_labels_ignore=None, corr_mode="pearson", 
                                 n_permutations=10, n_workers=32,
                                 show_elapsed_t=True,progress_msg=None):
        """
        Compute similarity matrix in parallel over parcels.

        Parameters
        ----------
        parcel_concentrations : dict
            Parcel id -> list of averaged concentrations (flattened over metabolites/perturbations).
        parcel_labels_ignore : list, optional
            Parcel ids to skip.
        corr_mode : str
            "spearman", "pearson", or "mi" (discretized mutual information surrogate).
        show_progress : bool
            Whether to render a progress bar while gathering futures.
        n_permutations : int
            Placeholder to match API; not used.
        n_workers : int
            Number of processes to dispatch for chunked computation.

        Returns
        -------
        tuple
            metsimatrix (np.ndarray), metsimatrix_pvalue (np.ndarray).
        """
        parcel_labels_ignore = parcel_labels_ignore or []
        parcel_labels_ids = list(parcel_concentrations.keys())
        n_parcels         = len(parcel_labels_ids)
        metsimatrix         = np.zeros((n_parcels, n_parcels))
        metsimatrix_pvalue  = np.zeros((n_parcels, n_parcels))
        if corr_mode=="mi":
            for idp in parcel_concentrations.keys():
                parcel_conc = parcel_concentrations[idp] 
                _, bin_edges = np.histogram(parcel_conc, bins="auto")
                parcel_concentrations[idp]  = np.digitize(parcel_conc , bin_edges[:-1])
        batch_size = (n_parcels + n_workers - 1) // n_workers  # Ceiling division
        start = time.time()
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for k in range(n_workers):
                start_idx = k * batch_size
                end_idx   = min((k + 1) * batch_size, n_parcels)
                sub_parcel_ids = list()
                for idx in np.arange(start_idx,end_idx):
                    sub_parcel_ids.append(parcel_labels_ids[idx])
                if start_idx >= n_parcels:
                    break
                futures.append(executor.submit(_compute_submatrix,k, start_idx, end_idx, 
                                               sub_parcel_ids, parcel_concentrations, parcel_labels_ignore, 
                                               corr_mode, n_permutations))
            
            with Progress() as progress:
                msg = progress_msg if progress_msg else "Collecting results"
                task = progress.add_task(f"{msg}", total=len(futures))
                for future in futures:
                    start_idx, end_idx, submatrix, submatrix_pvalue = future.result()
                    metsimatrix[start_idx:end_idx, :]                 = submatrix
                    metsimatrix_pvalue[start_idx:end_idx, :]          = submatrix_pvalue
                    progress.update(task, advance=1)

        metsimatrix = np.nan_to_num(metsimatrix, nan=0.0)
        if show_elapsed_t:
            debug.success(f"Time elapsed: {round(time.time()-start)} sec")
        return metsimatrix, metsimatrix_pvalue


    def extract_metabolite_per_parcel(self,array):
        """
        Reformat flat metabolite list into [metabolite, perturbation] array.
        """
        N_metabolites = len(self.metabolites)
        if len(array) % N_metabolites != 0:
            raise ValueError(f"The length of the array must be a multiple of {N_metabolites}.")
        # N_pert = len(array) // N_metabolites
        N_pert = self.npert
        metabolite_conc = np.zeros([N_metabolites,N_pert])
        for i,el in enumerate(array):
            met_pos = i % N_metabolites
            el_pos = round(np.floor(i/N_metabolites))
            metabolite_conc[met_pos,el_pos] = el
        return metabolite_conc


    def speanman_corr_quadratic(self,X, Y):
        """
        Compute Spearman correlation after fitting a quadratic term.

        Parameters
        ----------
        X, Y : np.ndarray
            Vectors to correlate. A linear fit is removed from Y before
            computing Spearman correlation with X**2 to capture curvature.

        Returns
        -------
        dict
            Dictionary with 'corr' and 'pvalue' entries.
        """
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), Y)
        slope = model.coef_[0]
        intercept = model.intercept_
        spearman_corr_quadratic, spearman_pvalue_quadratic = spearmanr(X**2, Y-slope*X-intercept)
        return {
            "corr": spearman_corr_quadratic,
            "pvalue": spearman_pvalue_quadratic
        }


def _compute_submatrix(worker_id,start_idx, end_idx, sub_parcel_indices, parcel_concentrations, parcel_labels_ignore, corr_mode, n_permutations):
    """
    Worker function used by ProcessPoolExecutor to compute a similarity submatrix.
    """
    parcel_labels_ignore = parcel_labels_ignore or []
    submatrix_size        = len(sub_parcel_indices)
    all_parcel_labels_ids = list(parcel_concentrations.keys())
    n_parcels             = len(all_parcel_labels_ids)
    submatrix             = np.zeros((submatrix_size, n_parcels))
    submatrix_pvalue      = np.zeros((submatrix_size, n_parcels))
    for i, parcel_idx_i in enumerate(sub_parcel_indices):
        if parcel_idx_i in parcel_labels_ignore:
            continue
        parcel_x = np.array(parcel_concentrations[parcel_idx_i]).flatten()
        for j, parcel_idx_j in enumerate(all_parcel_labels_ids):
            if parcel_idx_j in parcel_labels_ignore:
                continue
            parcel_y = np.array(parcel_concentrations[parcel_idx_j]).flatten()
            if np.isnan(np.mean(parcel_x)) or np.isnan(np.mean(parcel_y)):
                continue
            elif parcel_x.sum() == 0 or parcel_y.sum() == 0:
                continue
            try:
                if corr_mode == "spearman":
                    result = spearmanr(parcel_x, parcel_y, alternative="two-sided")
                    corr = result.statistic
                    pvalue = result.pvalue
                elif corr_mode == "pearson":
                    result = linregress(parcel_x, parcel_y)
                    corr = result.rvalue
                    pvalue = result.pvalue
                submatrix[i, j]        = corr
                submatrix_pvalue[i, j] = pvalue
                try:
                    submatrix[j, i]        = corr
                    submatrix_pvalue[j, i] = pvalue
                except:  # symmetric fill best-effort
                    pass
            except Exception as e:
                debug.warning(f"_compute_submatrix: {e}, parcel X {parcel_idx_i} - parcel Y {parcel_idx_j}")
                continue
    return start_idx, end_idx, submatrix, submatrix_pvalue
