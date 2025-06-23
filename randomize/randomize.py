import numpy as np


class Randomize:
    def __init__(self,mriData,space,metabolites=None,option=None):
        self.signal3D_dict = dict()
        self.noise3D_dict  = dict()
        if metabolites==None:
            self.metabolites   = mriData.metabolites
        else:
            self.metabolites   = metabolites
        for metabolite in mriData.metabolites:
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
        sigma = image3D * crlb3D / 100
        MAX   = image3D.mean() + 3*image3D.std()
        noisy_image3D = np.random.normal(image3D, sigma * sigma_scale)
        noisy_image3D = np.clip(noisy_image3D, 0, MAX)  # Clip values to stay within 0 and MAX
        noisy_image3D[self.mask==0]=0
        return noisy_image3D

   
    def sample_noisy_img4D(self,sigma=2):
        shape4d       = (len(self.metabolites),) + self.signal3D_dict["Ins"].shape
        noisy_image4D = np.zeros(shape4d)
        for idm,metabolite in enumerate(self.metabolites):
            image3D            = self.signal3D_dict[metabolite]
            crlb3D             = self.noise3D_dict[metabolite]
            noisy_image4D[idm] = self.perturbate(image3D,crlb3D,sigma_scale=sigma)
        return noisy_image4D

if __name__=="__main__":
    pass


            
