import os,sys
import numpy as np
from bct.algorithms import get_components
from scipy.stats import ttest_1samp
from scipy.sparse.csgraph import connected_components
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from tqdm import tqdm

from ..graphplot.simmatrix import SimMatrixPlot
from ..tools.debug import Debug

debug   = Debug()
simplt  = SimMatrixPlot()





class NBS(object):
    def __init__(self):
        pass

    def bct_corr(
        self,
        matrices,
        designmat_dict,
        nuisance=None,
        regress="state",
        permtest="freedman",
        gm_adj=None,
        t_thresh=3.5,
        n_perms=5000,
        nthreads=8,
        alpha=0.05,
        tail='both',
        seed=None,
        export_matlab_dir=None,
        node_coords=None,
        node_names=None,
        subject_ids=None,
        exchange_blocks=None,
        add_intercept=False,
    ):
        """
        Classical NBS with permutation schemes:
        - One-sample: sign-flips.
        - General linear model (with or without nuisance covariates): Freedman–Lane permutations.
        - Optional gray-matter adjacency constrained permutations for spatially aware nulls.

        Parameters
        ----------
        matrices : array-like, shape (n, n, S) or (S, n, n)
            Stack of subject adjacency matrices.
        designmat_dict : dict[str, array-like]
            Mapping from covariate name to subject-level vector.
        nuisance : list[str] | None, optional
            Names inside `designmat_dict` to treat as nuisance regressors. Defaults
            to all keys except the regressor of interest.
        regress : str | None, optional
            Name of the predictor to test. If None or not present in
            `designmat_dict`, a one-sample sign-flip test is performed.
        permtest : {'freedman', 'gmadj'}, optional
            Null generation strategy. 'freedman' (default) performs Freedman–Lane
            permutations across subjects (sign-flips when no regressor is provided).
            'gmadj' generates null matrices by spatially shuffling node identities
            while only allowing swaps between gray-matter adjacent parcels.
        gm_adj : np.ndarray | None, optional
            Binary (n, n) adjacency describing which parcels are considered neighbors.
            Required when `permtest='gmadj'`.
        export_matlab_dir : str | None, optional
            Directory where MATLAB NBS input files will be written. If None (default),
            no files are exported.
        node_coords : array-like, shape (n, 3), optional
            XYZ coordinates per node; required when `export_matlab_dir` is provided.
        node_names : array-like, shape (n,), optional
            Node label strings; required when `export_matlab_dir` is provided.
        subject_ids : array-like, shape (S,), optional
            Subject identifiers used only for export; defaults to 1..S if omitted.
        exchange_blocks : array-like, shape (S,), optional
            Block labels defining exchangeability groups for permutations (e.g.,
            repeated-measures/paired designs). Residuals are permuted only within
            each block. Ignored when None.
        add_intercept : bool, optional
            Whether to include an intercept column (ones) in the design. Defaults
            to False so the design matches MATLAB runs without an intercept.
        Returns
        -------
        results : dict
            {
            'comp_pvals'   : list of p-values per observed component,
            'comp_edges'   : list of 1D index arrays (edges in iu) per component,
            'comp_masks'   : list of (N,N) boolean masks per component,
            'sig_mask'     : (N,N) boolean mask union of significant components (p<alpha),
            't_mat'        : (N,N) float edge-wise t-statistics,
            'iu'           : tuple of upper-tri indices,
            'null_max'     : (n_perms,) array of max component sizes,
            'labels_nodes' : node labels for observed supra graph (for debugging),
            'export_paths' : dict of MATLAB-NBS compatible exports if requested,
            }
        """
        rng = np.random.default_rng(seed)
        matrices = np.asarray(matrices)
        if matrices.ndim != 3:
            raise ValueError("`matrices` must be a 3D array of stacked adjacency matrices.")
        s0, s1, s2 = matrices.shape
        if s0 == s1:
            N, _, S = matrices.shape
        elif s1 == s2:
            S, N, _ = matrices.shape
            matrices = np.transpose(matrices, (1, 2, 0))
        else:
            raise ValueError("`matrices` must have shape (n, n, k) or (k, n, n).")
        if matrices.shape[:2] != (N, N):
            raise ValueError("`matrices` must contain square adjacency matrices.")
        N, _, S = matrices.shape

        export_dir = os.path.abspath(export_matlab_dir) if export_matlab_dir else None
        coords_arr = names_arr = subj_labels = None
        if export_dir:
            if node_coords is None:
                raise ValueError("`node_coords` must be provided when exporting MATLAB inputs.")
            coords_arr = np.asarray(node_coords, dtype=float)
            if coords_arr.shape != (N, 3):
                raise ValueError(f"`node_coords` must have shape ({N}, 3); got {coords_arr.shape}.")
            if node_names is None:
                raise ValueError("`node_names` must be provided when exporting MATLAB inputs.")
            names_arr = np.asarray(node_names)
            if names_arr.shape[0] != N:
                raise ValueError(f"`node_names` must have length {N}; got {names_arr.shape[0]}.")
            if subject_ids is None:
                subj_labels = np.arange(1, S + 1)
            else:
                subj_labels = np.asarray(subject_ids)
                if subj_labels.shape[0] != S:
                    raise ValueError(f"`subject_ids` must have length {S}; got {subj_labels.shape[0]}.")

        permtest = (permtest or "freedman").lower()
        if permtest not in {"freedman", "gmadj"}:
            raise ValueError("`permtest` must be either 'freedman' or 'gmadj'.")
        if tail not in {"both", "right", "left"}:
            raise ValueError("`tail` must be one of {'both','right','left'}.")

        # vectorize edges
        iu = np.triu_indices(N, 1)
        E  = len(iu[0])
        Y  = np.stack([matrices[:, :, i][iu] for i in range(S)], axis=0)  # (S, E)

        # parse covariates
        covariates = {k: np.asarray(v) for k, v in designmat_dict.items()}
        for key, arr in covariates.items():
            if arr.shape[0] != S:
                raise ValueError(f"Covariate `{key}` must have length {S}.")

        if nuisance is None:
            nuisance_keys = [k for k in covariates.keys() if k != regress]
        else:
            nuisance = [nuisance] if isinstance(nuisance, str) else list(nuisance)
            missing = [k for k in nuisance if k not in covariates]
            if missing:
                raise ValueError(f"Nuisance covariate(s) {missing} not found in design matrix.")
            if regress is not None and regress in nuisance:
                raise ValueError("Regressor of interest cannot also be listed as nuisance.")
            nuisance_keys = nuisance

        target = None
        if regress is not None:
            if regress not in covariates:
                raise ValueError(f"Regressor `{regress}` not found in design matrix.")
            target = covariates[regress].astype(float).reshape(S, 1)
            if np.allclose(target, target[0]):
                raise ValueError(f"Regressor `{regress}` is constant; cannot estimate effect.")

        # Build design matrices
        # Full design = [optional intercept] + nuisances + regressor (if provided)
        parts_full = []
        if add_intercept:
            X_int = np.ones((S, 1))
            parts_full.append(X_int)
        if nuisance_keys:
            X_nui = np.column_stack([covariates[k].astype(float) for k in nuisance_keys])
            parts_full.append(X_nui)
        else:
            X_nui = None
        group_col_idx = None
        if target is not None:
            group_col_idx = sum(p.shape[1] for p in parts_full)
            parts_full.append(target)
        X_full = np.column_stack(parts_full)  # (S, p_full)
        design_columns = []
        if add_intercept:
            design_columns.append("intercept")
        design_columns.extend(nuisance_keys)
        if target is not None:
            design_columns.append(regress)

        # Reduced design (intercept + nuisances), used for residualization
        parts_red = parts_full[:-1] if target is not None else parts_full
        X_red = np.column_stack(parts_red)

        # helper: compute edge-wise t for the group column of X_full
        def group_t_from_Y(Ymat, X_full, group_col_idx):
            # OLS: beta = (X^+ Y), residual E = Y - X beta
            Xp   = np.linalg.pinv(X_full)                  # (p, S)
            beta = Xp @ Ymat                               # (p, E)
            Yhat = X_full @ beta                           # (S, E)
            Eres = Ymat - Yhat                             # (S, E)
            dof  = X_full.shape[0] - X_full.shape[1]
            # variance: sigma^2 per edge
            s2   = (Eres**2).sum(axis=0) / max(dof, 1)
            XtX_inv = np.linalg.pinv(X_full.T @ X_full)
            se_g = np.sqrt(np.maximum(s2, 0) * XtX_inv[group_col_idx, group_col_idx])
            tval = beta[group_col_idx, :] / np.where(se_g == 0, np.inf, se_g)
            return tval

        tail_map = {'both': 'two-sided', 'right': 'greater', 'left': 'less'}

        def compute_t_stats(Ymat):
            if target is None:
                if nuisance_keys:
                    B = np.linalg.lstsq(X_red, Ymat, rcond=None)[0]
                    R = Ymat - X_red @ B
                else:
                    R = Ymat
                return ttest_1samp(R, popmean=0.0, axis=0, alternative=tail_map[tail])[0]
            if group_col_idx is None:
                raise RuntimeError("Regressor column index missing for GLM computation.")
            return group_t_from_Y(Ymat, X_full, group_col_idx)

        T_obs = compute_t_stats(Y)

        block_indices = None
        if exchange_blocks is not None:
            blocks_arr = np.asarray(exchange_blocks)
            if blocks_arr.shape[0] != S:
                raise ValueError(f"`exchange_blocks` must have length {S}; got {blocks_arr.shape[0]}.")
            unique_blocks = np.unique(blocks_arr)
            block_indices = [np.flatnonzero(blocks_arr == b) for b in unique_blocks]
            if any(idx.size == 0 for idx in block_indices):
                raise ValueError("All exchangeability blocks must contain at least one observation.")

        R_for_perm = None
        Yhat_red = None
        Rred = None
        if permtest == "freedman":
            if target is None:
                if nuisance_keys:
                    B = np.linalg.lstsq(X_red, Y, rcond=None)[0]
                    R_for_perm = Y - X_red @ B
                else:
                    R_for_perm = Y.copy()
            else:
                Bred = np.linalg.lstsq(X_red, Y, rcond=None)[0]
                Yhat_red = X_red @ Bred
                Rred = Y - Yhat_red

        if permtest == "gmadj":
            if gm_adj is None:
                raise ValueError("`gm_adj` must be provided when permtest='gmadj'.")
            gm_adj_arr = np.asarray(gm_adj)
            if gm_adj_arr.shape != (N, N):
                raise ValueError("`gm_adj` must have shape (n_nodes, n_nodes).")
            gm_adj_bool = gm_adj_arr.astype(bool)
            gm_adj_bool = gm_adj_bool | gm_adj_bool.T
            np.fill_diagonal(gm_adj_bool, False)
            gm_neighbors = [np.flatnonzero(gm_adj_bool[i]) for i in range(N)]
            if not any(nb.size for nb in gm_neighbors):
                raise ValueError("`gm_adj` must include at least one valid adjacency.")
            swap_steps = max(N * 4, 1)

        # threshold (respect tail)
        if tail == 'both':
            supra = np.abs(T_obs) >= t_thresh
        elif tail == 'right':
            supra = T_obs >= t_thresh
        else:
            supra = -T_obs >= t_thresh

        # observed supra graph and components
        A_obs = np.zeros((N, N), dtype=bool)
        A_obs[iu] = supra
        A_obs |= A_obs.T
        n_comp, labels_nodes = connected_components(A_obs.astype(int), directed=False, connection='weak')
        obs_clusters = []
        for comp_id in range(n_comp):
            nodes = np.where(labels_nodes == comp_id)[0]
            if nodes.size < 2:
                continue
            mask = np.isin(iu[0], nodes) & np.isin(iu[1], nodes) & supra
            edges = np.where(mask)[0]
            if edges.size:
                obs_clusters.append(edges)

        obs_sizes = np.array([len(c) for c in obs_clusters], int)

        # --- Permutation nulls ---
        if permtest == "freedman":
            def sample_permutation(rlocal):
                if block_indices is None:
                    return rlocal.permutation(S)
                P = np.arange(S)
                for idx in block_indices:
                    permuted = rlocal.permutation(idx)
                    P[idx] = permuted
                return P

            def max_cluster_size_for_perm(seed_i):
                rlocal = np.random.default_rng(seed_i)
                if target is None:
                    if R_for_perm is None:
                        raise RuntimeError("Residuals not initialized for sign-flip permutations.")
                    signs = rlocal.choice([1, -1], size=S).astype(float)[:, None]
                    Tp = ttest_1samp(R_for_perm * signs, popmean=0.0, axis=0,
                                     alternative=tail_map[tail])[0]
                else:
                    if Yhat_red is None or Rred is None:
                        raise RuntimeError("Reduced-model components missing for Freedman–Lane permutations.")
                    P = sample_permutation(rlocal)
                    Yp = Yhat_red + Rred[P, :]
                    Tp = group_t_from_Y(Yp, X_full, group_col_idx)

                if tail == 'both':
                    supr = np.abs(Tp) >= t_thresh
                elif tail == 'right':
                    supr = Tp >= t_thresh
                else:
                    supr = -Tp >= t_thresh

                if not np.any(supr):
                    return 0
                Ap = np.zeros((N, N), dtype=bool)
                Ap[iu] = supr
                Ap |= Ap.T
                ncp, labs = connected_components(Ap.astype(int), directed=False, connection='weak')
                max_sz = 0
                for cid in range(ncp):
                    nds = np.where(labs == cid)[0]
                    if nds.size < 2:
                        continue
                    mask = np.isin(iu[0], nds) & np.isin(iu[1], nds) & supr
                    sz = int(np.sum(mask))
                    if sz > max_sz:
                        max_sz = sz
                return max_sz
        else:
            def adjacency_permutation(rng):
                perm = np.arange(N)
                inv = np.arange(N)
                for _ in range(swap_steps):
                    u = rng.integers(0, N)
                    neigh = gm_neighbors[u]
                    if neigh.size == 0:
                        continue
                    v = neigh[rng.integers(0, neigh.size)]
                    if u == v:
                        continue
                    pu, pv = inv[u], inv[v]
                    perm[pu], perm[pv] = perm[pv], perm[pu]
                    inv[u], inv[v] = pv, pu
                return perm

            def max_cluster_size_for_perm(seed_i):
                rlocal = np.random.default_rng(seed_i)
                Yp = np.empty_like(Y)
                for subj in range(S):
                    perm = adjacency_permutation(rlocal)
                    subj_mat = matrices[:, :, subj]
                    perm_mat = np.take(np.take(subj_mat, perm, axis=0), perm, axis=1)
                    Yp[subj, :] = perm_mat[iu]
                Tp = compute_t_stats(Yp)

                if tail == 'both':
                    supr = np.abs(Tp) >= t_thresh
                elif tail == 'right':
                    supr = Tp >= t_thresh
                else:
                    supr = -Tp >= t_thresh

                if not np.any(supr):
                    return 0
                Ap = np.zeros((N, N), dtype=bool)
                Ap[iu] = supr
                Ap |= Ap.T
                ncp, labs = connected_components(Ap.astype(int), directed=False, connection='weak')
                max_sz = 0
                for cid in range(ncp):
                    nds = np.where(labs == cid)[0]
                    if nds.size < 2:
                        continue
                    mask = np.isin(iu[0], nds) & np.isin(iu[1], nds) & supr
                    sz = int(np.sum(mask))
                    if sz > max_sz:
                        max_sz = sz
                return max_sz

        seeds = (rng.integers(0, 2**31-1, size=n_perms) if seed is not None else np.arange(n_perms))
        parallel_kwargs = {"prefer": "threads"} if permtest == "gmadj" else {}
        null_max = np.array(
            Parallel(n_jobs=nthreads, **parallel_kwargs)(
                delayed(max_cluster_size_for_perm)(int(s)) for s in tqdm(seeds, desc="Permutations")
            )
        )

        # component-level p-values + masks
        comp_pvals, comp_masks = [], []
        for edges in obs_clusters:
            sz = len(edges)
            p = (np.sum(null_max >= sz) + 1) / (n_perms + 1)  # +1 smoothing
            comp_pvals.append(p)
            m = np.zeros((N, N), bool)
            m[iu[0][edges], iu[1][edges]] = True
            m |= m.T
            comp_masks.append(m)

        # union of significant components
        sig_mask = np.zeros((N, N), bool)
        for p, m in zip(comp_pvals, comp_masks):
            if p < alpha:
                sig_mask |= m

        # full t-matrix
        t_mat = np.zeros((N, N), float)
        t_mat[iu] = T_obs
        t_mat = t_mat + t_mat.T

        export_paths = {}
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            mat_dir = os.path.join(export_dir, "matrices")
            os.makedirs(mat_dir, exist_ok=True)
            matrices_float = matrices.astype(float, copy=False)
            matrix_files = []
            for idx in range(S):
                fname = os.path.join(mat_dir, f"subject{idx+1}.txt")
                np.savetxt(fname, matrices_float[:, :, idx], fmt="%.6f")
                matrix_files.append(fname)
            cog_path = os.path.join(export_dir, "COG.txt")
            np.savetxt(cog_path, coords_arr, fmt="%.6f")
            labels_path = os.path.join(export_dir, "nodeLabels.txt")
            with open(labels_path, "w", encoding="utf-8") as f:
                for name in names_arr:
                    f.write(f"{name}\n")
            design_path = os.path.join(export_dir, "designMatrix.txt")
            design_to_save = X_full if target is not None else X_red
            np.savetxt(design_path, design_to_save, fmt="%.6f")
            export_paths = {
                "export_dir": export_dir,
                "matrices": matrix_files,
                "cog": cog_path,
                "node_labels": labels_path,
                "design_matrix": design_path,
                "design_columns": design_columns,
                "subject_ids": subj_labels.tolist() if subj_labels is not None else None,
            }

        return {
            'comp_pvals'  : comp_pvals,
            'comp_edges'  : obs_clusters,
            'comp_masks'  : comp_masks,
            'sig_mask'    : sig_mask,
            't_mat'       : t_mat,
            'iu'          : iu,
            'null_max'    : null_max,
            'labels_nodes': labels_nodes,
            'export_paths': export_paths,
        }


    def bct_corr_z(self,matrices, thresh, y_vec, n_perms=1000, extent=True, verbose=False, nthreads=None):

        '''
        Performs the NBS for matrices [matrices] and vector [y_vec]  for a Pearson's r-statistic threshold of
        [thresh].

        Parameters
        ----------
        matrices : NxNxP np.ndarray
            matrix representing the correlation matrices population with P subjects. must be
            symmetric.

        y_vec : 1xP vector representing the behavioral/physiological values to correlate against

        thresh : float
            minimum Pearson's r-value used as threshold
        n_perms : int
            number of permutations used to estimate the empirical null
            distribution, recommended - 10000
        verbose : bool
            print some extra information each iteration. defaults value = False
        nthreads : int or None
            number of worker threads to use when estimating the null distribution.
            Defaults to the available CPU count.

        Returns
        -------
        pval : Cx1 np.ndarray
            A vector of corrected p-values for each component of the networks
            identified. If at least one p-value is less than thres, the omnibus
            null hypothesis can be rejected at alpha significance. The null
            hypothesis is that the value of the connectivity from each edge has
            equal mean across the two populations.
        adj : IxIxC np.ndarray
            an adjacency matrix identifying the edges comprising each component.
            edges are assigned indexed values.
        null : Kx1 np.ndarray
            A vector of K sampled from the null distribution of maximal component
            size.

        Notes
        -----
        ALGORITHM DESCRIPTION
        The NBS is a nonparametric statistical test used to isolate the
        components of an N x N undirected connectivity matrix that differ
        significantly between two distinct populations. Each element of the
        connectivity matrix stores a connectivity value and each member of
        the two populations possesses a distinct connectivity matrix. A
        component of a connectivity matrix is defined as a set of
        interconnected edges.

        The NBS is essentially a procedure to control the family-wise error
        rate, in the weak sense, when the null hypothesis is tested
        independently at each of the N(N-1)/2 edges comprising the undirected
        connectivity matrix. The NBS can provide greater statistical power
        than conventional procedures for controlling the family-wise error
        rate, such as the false discovery rate, if the set of edges at which
        the null hypothesis is rejected constitues a large component or
        components.

        The NBS comprises fours steps:
        1. Perform a Pearson r test at each edge indepedently to test the
        hypothesis that the value of connectivity between each edge and an
        external variable, corelates across all nodes.
        2. Threshold the Pearson r-statistic available at each edge to form a set of
        suprathreshold edges.
        3. Identify any components in the adjacency matrix defined by the set
        of suprathreshold edges. These are referred to as observed
        components. Compute the size of each observed component
        identified; that is, the number of edges it comprises.
        4. Repeat K times steps 1-3, each time randomly permuting the extarnal
        variable vector and storing the size of the largest component
        identified for each permutation. This yields an empirical estimate
        of the null distribution of maximal component size. A corrected
        p-value for each observed component is then calculated using this
        null distribution.

        [1] Zalesky A, Fornito A, Bullmore ET (2010) Network-based statistic:
            Identifying differences in brain networks. NeuroImage.
            10.1016/j.neuroimage.2010.06.041

        Adopted from the python implementation of the BCT - https://sites.google.com/site/bctnet/, https://pypi.org/project/bctpy/
        Credit for implementing the vectorized version of the code to Gideon Rosenthal

        '''

        ix, jx, nx = matrices.shape
        ny, = y_vec.shape

        if not ix == jx:
            raise ValueError('Matrices are not symmetrical')
        else:
            n = ix

        if nx != ny:
            raise ValueError('The [y_vec dimension must match the [matrices] third dimension')

        # only consider upper triangular edges
        ixes = np.where(np.triu(np.ones((n, n)), 1))

        # number of edges
        m = np.size(ixes, axis=1)

        # vectorize connectivity matrices for speed
        xmat = np.zeros((m, nx))
        for i in range(nx):
            xmat[:, i] = matrices[:, :, i][ixes].squeeze()
        del matrices

        # center the edge time series so correlations become fast dot products
        x_mean = xmat.mean(axis=1, keepdims=True)
        xmat -= x_mean
        x_norm = np.linalg.norm(xmat, axis=1)

        def compute_edge_z_scores(y_values):
            y_center = y_values - y_values.mean()
            y_norm = np.linalg.norm(y_center)
            if y_norm == 0:
                raise ValueError('Behavioral vector has zero variance')

            denom = x_norm * y_norm
            numerators = xmat @ y_center
            with np.errstate(divide='ignore', invalid='ignore'):
                r_vals = np.divide(
                    numerators,
                    denom,
                    out=np.zeros_like(numerators),
                    where=denom > 0,
                )

            r_vals = np.clip(r_vals, -0.999999, 0.999999)
            return 0.5 * np.log((1 + r_vals) / (1 - r_vals))

        # perform pearson corr test at each edge
        z_stat = compute_edge_z_scores(y_vec)
        debug.info('z_stat: ', z_stat)

        # threshold
        ind_r, = np.where(z_stat > thresh)

        if len(ind_r) == 0:
            raise ValueError("Unsuitable threshold")
        # suprathreshold adjacency matrix
        adj = np.zeros((n, n))
        adjT = np.zeros((n, n))

        if extent:
            adj[(ixes[0][ind_r], ixes[1][ind_r])] = 1
            adj = adj + adj.T  # make symmetrical
        else:
            adj[(ixes[0][ind_r], ixes[1][ind_r])] = 1
            adj = adj + adj.T  # make symmetrical
            adjT[(ixes[0], ixes[1])] = z_stat
            adjT = adjT + adjT.T  # make symmetrical
            adjT[adjT <= thresh] = 0
        a, sz = get_components(adj)
        # convert size from nodes to number of edges
        # only consider components comprising more than one node (e.g. a/l 1 edge)
        ind_sz, = np.where(sz > 1)
        ind_sz += 1
        nr_components = np.size(ind_sz)
        sz_links = np.zeros((nr_components,))
        for i in range(nr_components):
            nodes, = np.where(ind_sz[i] == a)
            if extent:
                sz_links[i] = np.sum(adj[np.ix_(nodes, nodes)]) / 2
            else:
                sz_links[i] = np.sum(adjT[np.ix_(nodes, nodes)]) / 2

            adj[np.ix_(nodes, nodes)] *= (i + 2)
        # subtract 1 to delete any edges not comprising a component
        adj[np.where(adj)] -= 1

        if np.size(sz_links):
            max_sz = np.max(sz_links)
        else:
            # max_sz=0
            raise ValueError('True matrix is degenerate')
        debug.info('max component size is %i' % max_sz)
        # estimate empirical null distribution of maximum component size by
        # generating n_perms independent permutations
        debug.info('estimating null distribution with %i permutations' % n_perms)

        null = np.zeros((n_perms,))
        hit = 0

        log_step = max(n_perms // 10, 1)

        if nthreads is None:
            nthreads = os.cpu_count() or 1
        nthreads = max(1, min(nthreads, n_perms))

        rng = np.random.default_rng()
        seeds = rng.integers(0, np.iinfo(np.uint32).max, size=n_perms, dtype=np.uint32)

        progress = None
        progress_task = None
        if debug.verbose:
            try:
                progress = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.completed}/{task.total} permutations"),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    transient=True,
                    console=debug.console,
                )
                progress.__enter__()
                progress_task = progress.add_task("Estimating null distribution", total=n_perms)
            except Exception:
                progress = None
                progress_task = None

        def permutation_worker(seed):
            rng_local = np.random.default_rng(int(seed))
            perm_indices = rng_local.permutation(ny)
            z_stat_perm = compute_edge_z_scores(y_vec[perm_indices])

            ind_r, = np.where(z_stat_perm > thresh)

            adj_perm = np.zeros((n, n))

            if extent:
                adj_perm[(ixes[0][ind_r], ixes[1][ind_r])] = 1
                adj_perm = adj_perm + adj_perm.T
            else:
                adj_perm[(ixes[0], ixes[1])] = z_stat_perm
                adj_perm = adj_perm + adj_perm.T
                adj_perm[adj_perm <= thresh] = 0

            a_perm, sz_perm = get_components(adj_perm)

            ind_sz_perm, = np.where(sz_perm > 1)
            if not np.size(ind_sz_perm):
                return 0.0

            ind_sz_perm += 1
            nr_components_perm = np.size(ind_sz_perm)
            sz_links_perm = np.zeros((nr_components_perm,))
            for i in range(nr_components_perm):
                nodes, = np.where(ind_sz_perm[i] == a_perm)
                sz_links_perm[i] = np.sum(adj_perm[np.ix_(nodes, nodes)]) / 2

            if np.size(sz_links_perm):
                return float(np.max(sz_links_perm))
            return 0.0

        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            for u, null_value in enumerate(executor.map(permutation_worker, seeds, chunksize=1)):
                null[u] = null_value

                if null_value >= max_sz:
                    hit += 1

                if progress and progress_task is not None:
                    progress.update(progress_task, advance=1, description=f"Permutation {u + 1}/{n_perms}")

                if verbose:
                    debug.info(
                        'permutation %i of %i.  Permutation max is %s.  Observed max is %s.  P-val estimate is %.3f'
                        % (u, n_perms, null_value, max_sz, hit / (u + 1))
                    )
                elif (u % log_step == 0 or u == n_perms - 1):
                    debug.info(
                        'permutation %i of %i.  p-value so far is %.3f'
                        % (u, n_perms, hit / (u + 1))
                    )

        if progress:
            progress.__exit__(None, None, None)

        pvals = np.zeros((nr_components,))
        # calculate p-vals
        for i in range(nr_components):
            pvals[i] = np.size(np.where(null >= sz_links[i])) / n_perms

        return pvals, adj, null
    
    @staticmethod
    def compare_rich_club(
        data_ctrl: np.ndarray,         # (n,n,kc)
        data_pat:  np.ndarray,         # (n,n,kp)
        bin_ref:   np.ndarray | None = None,  # (n,n) binary reference adjacency; if None it will be inferred
        *,
        deg_th: float = 80,            # hub selection threshold: if >=1 => absolute degree; if 0<deg_th<1 => percentile
        weight_th: float | None = None,# when bin_ref is None, threshold for binarizing the reference (see notes below)
        trim_prop: float = 0.10,       # trimmed mean proportion (two-sided)
        n_perm: int = 10000,           # permutations for label-based test
        n_boot: int = 2000,            # bootstraps for 95% CI of group mean difference
        random_state: int | None = 0,  # RNG seed
    ) -> dict:
        """
        Returns a compact dictionary summarizing RC/Feeder/Peripheral group differences.

        Reference binary graph (for degrees / hub selection):
        - If `bin_ref` is provided, it's used directly (assumed symmetric, 0/1, zero diagonal).
        - Else we compute the mean control matrix W = mean_k(data_ctrl[...,k]) and binarize:
            * If weight_th is None: bin_ref = (W > 0)
            * If 0 < weight_th < 1: keep top `weight_th` fraction of *upper-triangle* weights (density-based)
            * If weight_th >= 1:    bin_ref = (W >= weight_th) (value cutoff)
        Hub selection:
        - If deg_th >= 1: hubs = {i : degree_i >= deg_th}
        - If 0 < deg_th < 1: hubs = {i : degree_i >= quantile(degree, q=1-deg_th)}  (e.g., deg_th=0.2 => top 20%)

        The dict includes, for each class (rc / feeder / peripheral):
        - per-subject means (arrays for ctrl/pat),
        - robust summaries (mean, median, trimmed mean),
        - Welch t-test (t, p),
        - Cohen's d (Hedges' g-like correction for unequal n),
        - permutation p-value on the difference of means,
        - bootstrap 95% CI on the difference of means.
        Also returns BH-FDR q-values across the three class p-values (ttest and perm).
        """
        # -------------------- helpers --------------------
        def upper_triangle_indices(n):
            return np.triu_indices(n, k=1)
        def make_bin_ref_from_ctrl_mean(W, weight_th):
            n = W.shape[0]
            iu = upper_triangle_indices(n)
            A = np.zeros_like(W, dtype=int)
            if weight_th is None:
                mask = (W[iu] > 0)
            elif 0 < weight_th < 1:
                # keep top density fraction
                k_keep = int(np.round(weight_th * len(W[iu])))
                if k_keep < 1:
                    mask = np.zeros_like(W[iu], dtype=bool)
                else:
                    thresh = np.partition(W[iu], -k_keep)[-k_keep]
                    mask = (W[iu] >= thresh)
            else:
                mask = (W[iu] >= float(weight_th))
            A[iu] = mask.astype(int)
            A[(iu[1], iu[0])] = A[iu]  # symmetrize
            np.fill_diagonal(A, 0)
            return A
        def pick_hubs(A, deg_th):
            deg = A.sum(axis=0)
            if 0 < deg_th < 1:
                thr = np.quantile(deg, 1 - deg_th)
                hubs = np.where(deg >= thr)[0]
            else:
                hubs = np.where(deg >= deg_th)[0]
            return hubs, deg
        def class_mask(n, hubs):
            all_nodes = np.arange(n)
            nh = np.setdiff1d(all_nodes, hubs, assume_unique=True)
            rc = np.zeros((n, n), dtype=int)
            fd = np.zeros((n, n), dtype=int)
            pe = np.zeros((n, n), dtype=int)
            rc[np.ix_(hubs, hubs)] = 1
            fd[np.ix_(hubs, nh)] = 1
            fd[np.ix_(nh, hubs)] = 1
            pe[np.ix_(nh, nh)] = 1
            np.fill_diagonal(rc, 0); np.fill_diagonal(fd, 0); np.fill_diagonal(pe, 0)
            return rc, fd, pe
        def per_subject_means(arr, mask, iu):
            # arr: (n,n,k)
            # mask: (n,n) 0/1
            idx = mask[iu] == 1
            # collect mean of the masked edges for each subject
            return np.array([m[iu][idx].mean() if np.any(idx) else np.nan
                            for m in np.rollaxis(arr, 2)])
        def trimmed_mean(x, p):
            if p <= 0: return float(np.mean(x))
            if p >= 0.5: return float(np.median(x))
            x = np.sort(np.asarray(x))
            k = int(np.floor(p * len(x)))
            if 2*k >= len(x): return float(np.median(x))
            return float(np.mean(x[k:len(x)-k]))
        def cohens_d(x, y):
            x = np.asarray(x); y = np.asarray(y)
            nx, ny = len(x), len(y)
            vx, vy = x.var(ddof=1), y.var(ddof=1)
            s = np.sqrt(((nx - 1)*vx + (ny - 1)*vy) / (nx + ny - 2))
            if s == 0: return 0.0
            return (x.mean() - y.mean()) / s
        def welch_t_p(x, y):
            # falls back to manual Welch test to avoid external deps
            x = np.asarray(x); y = np.asarray(y)
            nx, ny = len(x), len(y)
            mx, my = x.mean(), y.mean()
            vx, vy = x.var(ddof=1), y.var(ddof=1)
            se2 = vx/nx + vy/ny
            if se2 == 0:
                return np.nan, 1.0
            t = (mx - my) / np.sqrt(se2)
            # Welch-Satterthwaite df
            df = (se2**2) / ((vx**2)/((nx**2)*(nx-1)) + (vy**2)/((ny**2)*(ny-1)))
            # two-sided p-value via Student-t CDF (approx using survival function of normal if df large)
            from math import erf, sqrt
            if df > 200:  # normal approx
                p = 2 * (1 - 0.5*(1 + erf(abs(t)/sqrt(2))))
            else:
                try:
                    from scipy.stats import t as tdist
                    p = 2 * tdist.sf(abs(t), df)
                except Exception:
                    # very rough fallback normal approx
                    p = 2 * (1 - 0.5*(1 + erf(abs(t)/sqrt(2))))
            return float(t), float(p)
        def perm_pvalue(x, y, n_perm, rng):
            x = np.asarray(x); y = np.asarray(y)
            nx = len(x)
            pool = np.r_[x, y]
            obs = x.mean() - y.mean()
            cnt = 0
            for _ in range(n_perm):
                rng.shuffle(pool)
                d = pool[:nx].mean() - pool[nx:].mean()
                if abs(d) >= abs(obs): cnt += 1
            return (cnt + 1) / (n_perm + 1)
        def boot_ci_mean_diff(x, y, n_boot, rng, alpha=0.05):
            x = np.asarray(x); y = np.asarray(y)
            nx, ny = len(x), len(y)
            diffs = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                xb = x[rng.integers(0, nx, nx)]
                yb = y[rng.integers(0, ny, ny)]
                diffs[b] = xb.mean() - yb.mean()
            lo, hi = np.quantile(diffs, [alpha/2, 1-alpha/2])
            return float(lo), float(hi)
        def bh_fdr(pvals):
            p = np.asarray(pvals, float)
            m = len(p)
            order = np.argsort(p)
            ranks = np.empty_like(order); ranks[order] = np.arange(1, m+1)
            q = p * m / ranks
            # enforce monotonicity
            q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
            q_final = np.empty_like(q); q_final[order] = np.clip(q_sorted, 0, 1)
            return q_final.tolist()
        # -------------------- prep --------------------
        rng = np.random.default_rng(random_state)
        n = data_ctrl.shape[0]
        assert data_ctrl.shape[:2] == (n, n) and data_pat.shape[:2] == (n, n), "Matrices must be (n,n,k)."
        iu = upper_triangle_indices(n)
        # Reference binary graph
        if bin_ref is None:
            W = np.mean(np.rollaxis(data_ctrl, 2), axis=0)  # mean control weights
            bin_ref = make_bin_ref_from_ctrl_mean(W, weight_th)
        else:
            bin_ref = np.array(bin_ref, dtype=int)
            assert bin_ref.shape == (n, n), "bin_ref must be (n,n)."
            np.fill_diagonal(bin_ref, 0)
            bin_ref = ((bin_ref + bin_ref.T) > 0).astype(int)  # ensure symmetric 0/1
        # Hubs and masks
        hubs, deg = pick_hubs(bin_ref, deg_th)
        rc_mask, fd_mask, pe_mask = class_mask(n, hubs)
        # Per-subject means
        rc_ctrl = per_subject_means(data_ctrl, rc_mask, iu)
        rc_pat  = per_subject_means(data_pat,  rc_mask, iu)
        fd_ctrl = per_subject_means(data_ctrl, fd_mask, iu)
        fd_pat  = per_subject_means(data_pat,  fd_mask, iu)
        pe_ctrl = per_subject_means(data_ctrl, pe_mask, iu)
        pe_pat  = per_subject_means(data_pat,  pe_mask, iu)
        # Summaries & tests for each class
        results = {}
        for name, x, y in [
            ("rc", rc_ctrl, rc_pat),
            ("feeder", fd_ctrl, fd_pat),
            ("peripheral", pe_ctrl, pe_pat),
        ]:
            # drop NaNs (can happen if a class has zero edges)
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if len(x) == 0 or len(y) == 0:
                results[name] = {
                    "ctrl": {"n": int(len(x)), "mean": np.nan, "median": np.nan, "trimmed_mean": np.nan},
                    "pat":  {"n": int(len(y)), "mean": np.nan, "median": np.nan, "trimmed_mean": np.nan},
                    "stats": {"t": np.nan, "p_t": 1.0, "cohens_d": 0.0, "p_perm": 1.0, "diff_mean": np.nan,
                            "diff_ci95": [np.nan, np.nan]}
                }
                continue
            t, p_t = welch_t_p(x, y)
            d = cohens_d(x, y)
            p_perm = perm_pvalue(x, y, n_perm=n_perm, rng=rng)
            lo, hi = boot_ci_mean_diff(x, y, n_boot=n_boot, rng=rng)
            results[name] = {
                "ctrl": {
                    "n": int(len(x)),
                    "mean": float(np.mean(x)),
                    "median": float(np.median(x)),
                    "trimmed_mean": float(trimmed_mean(x, trim_prop)),
                },
                "pat": {
                    "n": int(len(y)),
                    "mean": float(np.mean(y)),
                    "median": float(np.median(y)),
                    "trimmed_mean": float(trimmed_mean(y, trim_prop)),
                },
                "stats": {
                    "t": float(t),
                    "p_t": float(p_t),
                    "cohens_d": float(d),
                    "p_perm": float(p_perm),
                    "diff_mean": float(np.mean(x) - np.mean(y)),
                    "diff_ci95": [float(lo), float(hi)],
                },
            }
        # BH-FDR across the three classes (ttest and permutation separately)
        p_t_list   = [results[c]["stats"]["p_t"]   for c in ("rc","feeder","peripheral")]
        p_perm_list= [results[c]["stats"]["p_perm"]for c in ("rc","feeder","peripheral")]
        q_t   = bh_fdr(p_t_list)
        q_perm= bh_fdr(p_perm_list)
        for i, c in enumerate(("rc","feeder","peripheral")):
            results[c]["stats"]["q_t_BH"]    = float(q_t[i])
            results[c]["stats"]["q_perm_BH"] = float(q_perm[i])
        out = {
            "n_nodes": int(n),
            "hubs": {
                "indices": hubs.tolist(),
                "n_hubs": int(len(hubs)),
                "degree": deg.tolist(),
                "deg_threshold_input": deg_th,
            },
            "bin_ref_summary": {
                "density": float(bin_ref[iu].mean()),
                "num_edges": int(bin_ref[iu].sum()),
            },
            "class_results": results,
            "params": {
                "deg_th": deg_th,
                "weight_th": weight_th,
                "trim_prop": trim_prop,
                "n_perm": n_perm,
                "n_boot": n_boot,
                "random_state": random_state,
            },
        }
        return out
    
