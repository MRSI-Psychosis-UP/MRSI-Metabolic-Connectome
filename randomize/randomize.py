import numpy as np
from copy import copy
from tools.debug import Debug
debug = Debug()

class Randomize:
    def __init__(self,mriData,space,metabolites=None,option=None):
        """
        Prepare randomized metabolite volumes and noise maps for perturbation sampling.

        Parameters
        ----------
        mriData : MRIData
            Data accessor used to fetch signals/noise.
        space : str
            Target space of the MRSI data (e.g., "orig").
        metabolites : list, optional
            Metabolites to include; defaults to mriData.metabolites.
        option : str, optional
            Additional descriptor passed to MRIData when loading signal volumes.
        """
        self.signal3D_dict = dict()
        self.noise3D_dict  = dict()
        if metabolites==None:
            self.metabolites   = mriData.metabolites
        else:
            self.metabolites   = metabolites
        for metabolite in self.metabolites:
            # self.signal3D_dict[metabolite] = mriData.get_mrsi_volume(metabolite,space).get_fdata().squeeze()
            self.signal3D_dict[metabolite] = mriData.get_mri_nifti(modality="mrsi",space=space,desc="signal",
                                                                   met=metabolite,option=option).get_fdata().squeeze()

            # self.noise3D_dict[metabolite]  = mriData.get_mrsi_volume(f"{metabolite}-crlb","orig").get_fdata().squeeze()
            self.noise3D_dict[metabolite]  = mriData.get_mri_nifti(modality="mrsi",space=space,desc="crlb",
                                                                   met=metabolite).get_fdata().squeeze()

            # self.mask                      = mriData.get_mrsi_volume("brainmask",space).get_fdata().squeeze()
            self.mask                      = mriData.get_mri_nifti(modality="mrsi",space=space,
                                                                   desc="brainmask").get_fdata().squeeze()

    def perturbate(self,image3D,crlb3D,sigma_scale=2):
        """
        Add Gaussian noise scaled by CRLB and mask to brain region.
        """
        sigma = image3D * crlb3D / 100
        MAX   = image3D.mean() + 3*image3D.std()
        noisy_image3D = np.random.normal(image3D, sigma * sigma_scale)
        noisy_image3D = np.clip(noisy_image3D, 0, MAX)  # Clip values to stay within 0 and MAX
        noisy_image3D[self.mask==0]=0
        return noisy_image3D


    def subset(self, metabolites):
        """Return a lightweight clone restricted to the provided metabolites."""
        subset = copy(self)
        subset.metabolites = list(metabolites)
        subset.signal3D_dict = {met: self.signal3D_dict[met] for met in subset.metabolites}
        subset.noise3D_dict = {met: self.noise3D_dict[met] for met in subset.metabolites}
        return subset

    
    def sample_noisy_img4D(self,sigma=2):
        """
        Sample a 4D noisy metabolite volume using stored signal and CRLB maps.
        """
        if len(self.metabolites) == 0:
            raise ValueError("No metabolites available to sample.")
        shape_ref_met = self.metabolites[0]
        shape4d       = (len(self.metabolites),) + self.signal3D_dict[shape_ref_met].shape
        noisy_image4D = np.zeros(shape4d)
        for idm,metabolite in enumerate(self.metabolites):
            image3D            = self.signal3D_dict[metabolite]
            crlb3D             = self.noise3D_dict[metabolite]
            noisy_image4D[idm] = self.perturbate(image3D,crlb3D,sigma_scale=sigma)
        return noisy_image4D

if __name__=="__main__":
    pass


            
