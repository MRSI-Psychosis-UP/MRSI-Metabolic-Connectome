import numpy as np
from scipy.stats import linregress, spearmanr
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings, copy, time, itertools
from rich.progress import Progress,track
from tools.debug import Debug
from sklearn.linear_model import LinearRegression

debug  = Debug()


class MeSiM(object):
    def __init__(self):
        pass

    def compute_simmatrix(self,mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore=[],N_PERT=50,corr_mode = "spearman",
                            rescale="mean",n_permutations=10,return_parcel_conc=True,n_proc=16):
            parcel_concentrations = None
            self.metabolites      = mrsirand.metabolites
            self.npert            = N_PERT
            for i in track(range(N_PERT), description="Map parcellation image to MRSI space..."):
                met_image4D_data           = mrsirand.sample_noisy_img4D()
                parcel_concentrations      = self.parcellate_vectorized(met_image4D_data,parcel_mrsi_np,
                                                                        parcel_header_dict,rescale=rescale,
                                                                        parcel_concentrations=parcel_concentrations)
            simmatrix, pvalue         = self.__compute_simmatrix_parallel(parcel_concentrations,
                                                                        parcel_label_ids_ignore,
                                                                        corr_mode = corr_mode,
                                                                        show_progress=True,
                                                                        n_permutations=n_permutations,
                                                                        n_workers=n_proc)
            
            
            parcel_concentrations_np = None
            if return_parcel_conc:
                parcel_concentrations = None
                for i in track(range(N_PERT), description="Retrieve MRSI levels..."):
                    met_image4D_data      = mrsirand.sample_noisy_img4D()
                    parcel_concentrations = self.parcellate_vectorized(
                        met_image4D_data,
                        parcel_mrsi_np,
                        parcel_header_dict,
                        rescale="raw",
                        parcel_concentrations=parcel_concentrations,
                    )
                parcel_concentrations_np = self.map_mrsi_to_parcel(parcel_concentrations)
            return simmatrix, pvalue, parcel_concentrations_np


    def map_mrsi_to_parcel(self,parcel_concentrations):
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

    def __compute_simmatrix_parallel(self, parcel_concentrations, parcel_labels_ignore=[], corr_mode="pearson", show_progress=False, n_permutations=10, n_workers=32):
        parcel_labels_ids = list(parcel_concentrations.keys())
        parcel_indices    = list(range(len(parcel_labels_ids)))
        n_parcels         = len(parcel_labels_ids)
        simmatrix         = np.zeros((n_parcels, n_parcels))
        simmatrix_pvalue  = np.zeros((n_parcels, n_parcels))
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
                futures.append(executor.submit(self.compute_submatrix,k, start_idx, end_idx, 
                                               sub_parcel_ids, parcel_concentrations, parcel_labels_ignore, 
                                               corr_mode, n_permutations))
            
            with Progress() as progress:
                task = progress.add_task(f"Collecting results", total=len(futures))
                for future in futures:
                    start_idx, end_idx, submatrix, submatrix_pvalue = future.result()
                    simmatrix[start_idx:end_idx, :]                 = submatrix
                    simmatrix_pvalue[start_idx:end_idx, :]          = submatrix_pvalue
                    progress.update(task, advance=1)

        simmatrix = np.nan_to_num(simmatrix, nan=0.0)
        debug.success(f"Time elapsed: {round(time.time()-start)} sec")
        return simmatrix, simmatrix_pvalue


    def extract_metabolite_per_parcel(self,array):
        N_metabolites = len(self.metabolites)
        if len(array) % N_metabolites != 0:
            raise ValueError("The length of the array must be a multiple of 5.")
        # N_pert = len(array) // N_metabolites
        N_pert = self.npert
        metabolite_conc = np.zeros([N_metabolites,N_pert])
        for i,el in enumerate(array):
            met_pos = i % N_metabolites
            el_pos = round(np.floor(i/N_metabolites))
            metabolite_conc[met_pos,el_pos] = el
        return metabolite_conc


    def compute_submatrix(self,worker_id,start_idx, end_idx, sub_parcel_indices, parcel_concentrations, parcel_labels_ignore, corr_mode, n_permutations):
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
                    elif corr_mode == "spearman2":
                        result = self.speanman_corr_quadratic(parcel_x, parcel_y)
                        corr = result["corr"]
                        pvalue = result["pvalue"]
                    submatrix[i, j]        = corr
                    submatrix_pvalue[i, j] = pvalue
                    try:
                        submatrix[j, i]        = corr
                        submatrix_pvalue[j, i] = pvalue
                    except:pass
                except Exception as e:
                    debug.warning(f"compute_submatrix: {e}, parcel X {parcel_idx_i} - parcel Y {parcel_idx_j}")
                    continue
        return start_idx, end_idx, submatrix, submatrix_pvalue

    def leave_one_out(self,simmatrix_ref,mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,corr_mode = "spearman",rescale="zscore"):
        METABOLITES           = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
        pairwise_combinations = list(itertools.combinations(METABOLITES, 2))
        mets_remove_list      = [[a] for a in METABOLITES]
        # mets_remove_list.extend([[a,b] for a, b in pairwise_combinations])
        delta_arr = np.zeros(len(mets_remove_list))
        simmatrix_arr = list()
        for ids,mets_remove in enumerate(mets_remove_list):
            ids_to_remove  = list()
            metabolite_arr = copy.deepcopy(np.array(METABOLITES))
            for target in mets_remove:
                _ids = np.where(metabolite_arr==target)[0][0]
                ids_to_remove.append(_ids)
            # debug.warning("ids_to_remove",ids_to_remove)
            metabolite_arr = np.delete(metabolite_arr, ids_to_remove)
            # debug.warning("leave_one_out: Remove",METABOLITES[ids_to_remove[0]])
            mrsirand.metabolites   = metabolite_arr
            simmatrix, pvalue,_    = self.compute_simmatrix(mrsirand,parcel_mrsi_np,parcel_header_dict,parcel_label_ids_ignore,N_PERT,
                                                            corr_mode = "spearman",rescale="zscore",return_parcel_conc=False)
            simmatrix[pvalue>0.005] = 0
            delta = np.abs(simmatrix-simmatrix_ref).mean()
            delta_arr[ids] = delta
            simmatrix_arr.append(simmatrix)
        return np.array(simmatrix_arr)

    def speanman_corr_quadratic(self,X, Y):
        model = LinearRegression()
        model.fit(X.reshape(-1, 1), Y)
        slope = model.coef_[0]
        intercept = model.intercept_
        # Compute Spearman's rank correlation for second-order (quadratic) relationship
        spearman_corr_quadratic, spearman_pvalue_quadratic = spearmanr(X**2, Y-slope*X-intercept)
        return {
            "corr": spearman_corr_quadratic,
            "pvalue": spearman_pvalue_quadratic
        }
