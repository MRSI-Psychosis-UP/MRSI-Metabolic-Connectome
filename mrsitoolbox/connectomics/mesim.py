import numpy as np
from scipy.stats import linregress, spearmanr
from concurrent.futures import ProcessPoolExecutor
import warnings, copy, time
from rich.progress import Progress,track
from ..tools.debug import Debug
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
            Similarity mode ('spearman', 'pearson', 'mi', or 'gkaff' for Gaussian kernel affinity).
        rescale : str
            Rescaling strategy for intensities before parcellation.
        return_parcel_conc : bool
            Whether to return parcel concentrations array.
        n_proc : int
            Number of worker processes.
        """
        parcel_label_ids_ignore = parcel_label_ids_ignore or []
        parcel_concentrations = None
        self.metabolites = mrsirand.metabolites
        self.npert = N_PERT
        for _ in track(range(N_PERT), description="Parcellate MRSI maps...",
                       disable=hide_progress):
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
            n_workers      = n_proc,
            show_elapsed_t = show_elapsed_t,
            progress_msg   = progress_msg,
            hide_progress  = hide_progress
        )

        parcel_concentrations_np = None
        if return_parcel_conc:
            parcel_concentrations = None
            for _ in track(range(N_PERT), description="Retrieve MRSI levels...",
                           disable=hide_progress):
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


    def compute_metsim(self, mrsirand, parcel_mrsi_np, parcel_header_dict, 
                       parcel_label_ids_ignore=None,
                       N_PERT=50, corr_mode="spearman", 
                       rescale="mean", 
                       n_proc=16, leave_one_out=False,
                       hide_progress  = False,
                       return_parcel_conc=True,
                       show_elapsed_t = True):
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
            Similarity metric: "spearman" (default), "pearson", or "gkaff" (Gaussian kernel affinity).
        rescale : str
            Rescaling strategy passed to `parcellate_vectorized`.
        n_proc : int
            Number of worker processes for similarity computation.
        leave_one_out : bool
            If True, also returns matrices recomputed while leaving out each metabolite.
        show_elapsed_t: bool
            If True, display total processing time
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
            N_PERT             = N_PERT,
            corr_mode          = corr_mode,
            rescale            = rescale,
            return_parcel_conc = return_parcel_conc,
            n_proc             = n_proc,
            hide_progress      = hide_progress,
            show_elapsed_t     = show_elapsed_t
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
                                                                hide_progress  = True,
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

    def create_pop_matrix(self, matrices, labels_list, names_list):
        """
        Align a list of connectivity matrices to the label order of the
        largest matrix, zero-filling missing rows/cols.

        Returns
        -------
        aligned_mats : np.ndarray
            Array of shape [n_subjects, n_parcels, n_parcels].
        base_labels : np.ndarray
            Label ids defining the common ordering.
        base_names : np.ndarray
            Parcel names aligned to base_labels.
        """
        sizes = [len(lbls) if lbls is not None else 0 for lbls in labels_list]
        base_idx = int(np.argmax(sizes)) if sizes else 0
        base_labels = np.array(labels_list[base_idx]) if sizes else np.array([])
        base_names = np.array(names_list[base_idx]) if sizes else np.array([])
        label_to_pos = {lbl: idx for idx, lbl in enumerate(base_labels)}
        aligned = []
        for mat, labels in zip(matrices, labels_list):
            mat = np.asarray(mat)
            full = np.zeros((len(base_labels), len(base_labels)))
            if labels is None or len(label_to_pos) == 0:
                aligned.append(full)
                continue
            labels = np.array(labels)
            mapped_base = []
            mapped_curr = []
            for idx_curr, lbl in enumerate(labels):
                if lbl in label_to_pos:
                    mapped_base.append(label_to_pos[lbl])
                    mapped_curr.append(idx_curr)
            if mapped_base:
                mb = np.array(mapped_base)
                mc = np.array(mapped_curr)
                full[np.ix_(mb, mb)] = mat[np.ix_(mc, mc)]
            aligned.append(full)
        return np.array(aligned), base_labels, base_names

    def create_pop_profiles(self, profiles, labels_list, base_labels, 
                            default_tail=None, subject_ids=None, session_ids=None):
        """
        Align parcel-level profiles (e.g., concentrations) to a common label set.

        Parameters
        ----------
        profiles : list of np.ndarray or None
            Each array is shaped [parcel, ...] (e.g., [parcel, npert, nmet]).
        labels_list : list of iterables or None
            Parcel labels corresponding to each profile.
        base_labels : array-like
            Target label ordering to align to.
        default_tail : tuple, optional
            Shape to use for trailing dims when profile is missing; if None, inferred
            from the first non-empty profile.

        Returns
        -------
        np.ndarray
            Aligned profiles with shape [n_subjects, n_parcels, ...].
        """
        base_labels = np.asarray(base_labels)
        label_to_pos = {lbl: idx for idx, lbl in enumerate(base_labels)}

        def _subject_tag(idx_prof):
            subj_tag = ""
            if subject_ids is not None and session_ids is not None:
                try:
                    subj_tag = f" sub-{subject_ids[idx_prof]}_ses-{session_ids[idx_prof]}"
                except Exception:
                    subj_tag = ""
            return subj_tag

        def _coerce_profile_array(prof, n_labels, idx_prof):
            prof = np.asarray(prof)
            if prof.ndim == 0 or n_labels <= 0:
                return prof

            if prof.ndim == 4:
                if prof.shape[1] == n_labels:
                    prof = np.nanmedian(prof, axis=0) if prof.shape[0] > 1 else prof[0]
                elif prof.shape[0] == n_labels:
                    prof = np.nanmedian(prof, axis=1) if prof.shape[1] > 1 else prof[:, 0, ...]
            if prof.ndim == 2 and prof.shape[0] != n_labels and prof.shape[1] == n_labels:
                debug.info(
                    f"Profile axis normalization: subject index {idx_prof}{_subject_tag(idx_prof)} "
                    f"moved parcel axis 1->0 for shape {prof.shape}."
                )
                prof = np.moveaxis(prof, 1, 0)
            if prof.ndim >= 3 and prof.shape[0] != n_labels:
                if prof.shape[1] == n_labels:
                    debug.info(
                        f"Profile axis normalization: subject index {idx_prof}{_subject_tag(idx_prof)} "
                        f"moved parcel axis 1->0 for shape {prof.shape}."
                    )
                    prof = np.moveaxis(prof, 1, 0)
                elif prof.shape[2] == n_labels:
                    debug.info(
                        f"Profile axis normalization: subject index {idx_prof}{_subject_tag(idx_prof)} "
                        f"moved parcel axis 2->0 for shape {prof.shape}."
                    )
                    prof = np.moveaxis(prof, 2, 0)
                elif prof.ndim >= 4 and prof.shape[3] == n_labels:
                    debug.info(
                        f"Profile axis normalization: subject index {idx_prof}{_subject_tag(idx_prof)} "
                        f"moved parcel axis 3->0 for shape {prof.shape}."
                    )
                    prof = np.moveaxis(prof, 3, 0)
            return prof

        def _merge_tail_shapes(current_tail, new_tail):
            current_tail = tuple(current_tail or ())
            new_tail = tuple(new_tail or ())
            max_len = max(len(current_tail), len(new_tail))
            current_tail = current_tail + (1,) * (max_len - len(current_tail))
            new_tail = new_tail + (1,) * (max_len - len(new_tail))
            return tuple(max(int(a), int(b)) for a, b in zip(current_tail, new_tail))

        def _pad_profile_tail(prof, target_tail):
            prof = np.asarray(prof)
            target_tail = tuple(target_tail or ())
            if tuple(prof.shape[1:]) == target_tail:
                return prof
            prof_tail = tuple(prof.shape[1:])
            max_len = max(len(prof_tail), len(target_tail))
            prof_tail_ext = prof_tail + (1,) * (max_len - len(prof_tail))
            target_tail_ext = target_tail + (1,) * (max_len - len(target_tail))
            if any(int(src) > int(dst) for src, dst in zip(prof_tail_ext, target_tail_ext)):
                return prof
            reshape_shape = (prof.shape[0],) + prof_tail_ext
            prof = prof.reshape(reshape_shape)
            out = np.zeros((prof.shape[0],) + target_tail_ext, dtype=prof.dtype)
            slices = [slice(None)]
            for dim in prof.shape[1:]:
                slices.append(slice(0, int(dim)))
            out[tuple(slices)] = prof
            return out

        coerced_profiles = []
        coerced_labels = []
        if default_tail is None:
            inferred_tail = ()
        else:
            inferred_tail = tuple(default_tail)

        for idx_prof, (prof, labels) in enumerate(zip(profiles, labels_list)):
            if prof is None or labels is None:
                coerced_profiles.append(None)
                coerced_labels.append(None if labels is None else np.asarray(labels))
                continue
            labels = np.asarray(labels)
            prof = _coerce_profile_array(prof, len(labels), idx_prof)
            coerced_profiles.append(prof)
            coerced_labels.append(labels)
            if default_tail is None and prof.ndim >= 1:
                inferred_tail = _merge_tail_shapes(inferred_tail, prof.shape[1:])

        default_tail = tuple(inferred_tail) if inferred_tail else ()

        aligned = []
        for idx_prof, (prof, labels) in enumerate(zip(coerced_profiles, coerced_labels)):
            subj_tag = _subject_tag(idx_prof)
            base_shape = (len(base_labels),) + default_tail
            out = np.zeros(base_shape)
            if prof is None or labels is None or len(label_to_pos) == 0:
                # debug.warning(
                #     f"Profile shape mismatch: subject index {idx_prof}{subj_tag} has missing profile or labels."
                # )
                aligned.append(out)
                continue
            prof = np.asarray(prof)
            labels = np.asarray(labels)
            n_labels = len(labels)
            if prof.shape[0] != n_labels:
                debug.warning(
                    f"Profile axis mismatch: subject index {idx_prof}{subj_tag} expected parcel axis length {n_labels}, "
                    f"got shape {prof.shape}; zero-filling unmatched."
                )
                aligned.append(out)
                continue
            mapped_base = []
            mapped_curr = []
            for idx_curr, lbl in enumerate(labels):
                if lbl in label_to_pos:
                    mapped_base.append(label_to_pos[lbl])
                    mapped_curr.append(idx_curr)
            if mapped_base:
                mb = np.array(mapped_base)
                mc = np.array(mapped_curr)
                prof_tail = prof.shape[1:]
                if prof_tail != default_tail:
                    if len(prof_tail) == 2 and prof_tail == default_tail[::-1]:
                        prof = np.swapaxes(prof, 1, 2)
                        prof_tail = prof.shape[1:]
                    else:
                        padded = _pad_profile_tail(prof, default_tail)
                        if tuple(padded.shape[1:]) == default_tail:
                            prof = padded
                            prof_tail = prof.shape[1:]
                        else:
                            debug.warning(
                                f"Profile shape mismatch: subject index {idx_prof}{subj_tag} expected tail {default_tail}, got {prof_tail}; zero-filling unmatched."
                            )
                if prof_tail == default_tail and prof[mc].shape[1:] == default_tail:
                    out[mb] = prof[mc]
            aligned.append(out)
        return np.array(aligned)


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
                                 n_workers=32,
                                 show_elapsed_t=True,progress_msg=None,
                                 hide_progress=False):
        """
        Compute similarity matrix in parallel over parcels.

        Parameters
        ----------
        parcel_concentrations : dict
            Parcel id -> list of averaged concentrations (flattened over metabolites/perturbations).
        parcel_labels_ignore : list, optional
            Parcel ids to skip.
        corr_mode : str
            "spearman", "pearson", "mi" (discretized mutual information surrogate), or "gkaff" (Gaussian kernel affinity).
        hide_progress : bool
            Whether to render a progress bar while gathering futures.
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
        start = time.time()

        if corr_mode == "gkaff":
            ignore_set = set(parcel_labels_ignore)
            metsimatrix = self._compute_gaussian_kernel_affinity(parcel_concentrations,
                                                                 parcel_labels_ids,
                                                                 ignore_set)
            metsimatrix_pvalue = np.full_like(metsimatrix, np.nan)
            if show_elapsed_t:
                debug.success(f"Time elapsed: {round(time.time()-start)} sec")
            return metsimatrix, metsimatrix_pvalue

        if corr_mode=="mi":
            for idp in parcel_concentrations.keys():
                parcel_conc = parcel_concentrations[idp] 
                _, bin_edges = np.histogram(parcel_conc, bins="auto")
                parcel_concentrations[idp]  = np.digitize(parcel_conc , bin_edges[:-1])
        batch_size = (n_parcels + n_workers - 1) // n_workers  # Ceiling division
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
                                               corr_mode))
            
            msg = progress_msg if progress_msg else "Collecting results"
            if hide_progress:
                for future in futures:
                    start_idx, end_idx, submatrix, submatrix_pvalue = future.result()
                    metsimatrix[start_idx:end_idx, :]                 = submatrix
                    metsimatrix_pvalue[start_idx:end_idx, :]          = submatrix_pvalue
            else:
                with Progress() as progress:
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


    def _compute_gaussian_kernel_affinity(self, parcel_concentrations, parcel_labels_ids, ignore_set):
        """
        Compute Gaussian kernel affinity on z-scored metabolite profiles (averaged across perturbations).
        """
        n_parcels = len(parcel_labels_ids)
        n_metabolites = len(self.metabolites) if hasattr(self, "metabolites") else 0
        affinity = np.zeros((n_parcels, n_parcels))
        if n_parcels == 0 or n_metabolites == 0:
            return affinity

        profiles = np.zeros((n_parcels, n_metabolites))
        expected = n_metabolites * self.npert if getattr(self, "npert", None) else None
        for idx, parcel_id in enumerate(parcel_labels_ids):
            parcel_values = np.asarray(parcel_concentrations[parcel_id], dtype=float)
            if parcel_values.size == 0:
                reshaped = np.zeros((n_metabolites, 1))
            else:
                if expected and parcel_values.size != expected:
                    debug.warning(f"Parcel {parcel_id} has {parcel_values.size} values, expected {expected}; zero-filling mismatch.")
                try:
                    reshaped = parcel_values.reshape(n_metabolites, -1)
                except ValueError:
                    reshaped = np.zeros((n_metabolites, 1))
            profiles[idx] = np.nanmean(reshaped, axis=1)

        valid_mask = np.array([pid not in ignore_set for pid in parcel_labels_ids])
        if valid_mask.any():
            valid_profiles = profiles[valid_mask]
            mean = np.nanmean(valid_profiles, axis=0, keepdims=True)
            std = np.nanstd(valid_profiles, axis=0, keepdims=True)
        else:
            mean = np.zeros((1, n_metabolites))
            std = np.ones((1, n_metabolites))

        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std[std == 0] = 1

        z_profiles = (profiles - mean) / std
        z_profiles = np.nan_to_num(z_profiles, nan=0.0)
        sq_norms = np.sum(z_profiles ** 2, axis=1, keepdims=True)
        dist_sq = sq_norms + sq_norms.T - 2 * (z_profiles @ z_profiles.T)
        dist_sq = np.maximum(dist_sq, 0.0)

        if valid_mask.any():
            valid_dist = np.sqrt(dist_sq[np.ix_(valid_mask, valid_mask)])
            upper = valid_dist[np.triu_indices(valid_dist.shape[0], k=1)]
            finite_upper = upper[np.isfinite(upper) & (upper > 0)]
        else:
            finite_upper = np.array([])
        sigma = np.median(finite_upper) if finite_upper.size else 1.0
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0

        affinity = np.exp(-dist_sq / (2 * sigma ** 2))
        if ignore_set:
            ignored_idx = [idx for idx, pid in enumerate(parcel_labels_ids) if pid in ignore_set]
            affinity[ignored_idx, :] = 0
            affinity[:, ignored_idx] = 0

        return affinity


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


def _compute_submatrix(worker_id,start_idx, end_idx, sub_parcel_indices, parcel_concentrations, parcel_labels_ignore, corr_mode):
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
