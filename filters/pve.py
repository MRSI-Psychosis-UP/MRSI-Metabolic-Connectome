import numpy   as np
import nibabel as nib
import nilearn as nil
from nilearn.image import resample_img, resample_to_img

import tempfile, os

from os.path import join, exists, split
from registration.registration import Registration
from tools.filetools import FileTools
from tools.debug import Debug
from tools.mridata import MRIData
import subprocess

debug    = Debug()
reg      = Registration()
ftools   = FileTools()


class PVECorrection:
    def __init__(self):
        self._tissue_list = [None, "GM","WM","CSF"]
        pass
    

    def create_3tissue_pev(self,mridb,space="mrsi"):


        mrsi_path   = mridb.get_mri_filepath("mrsi","orig","signal","Ins","filtbiharmonic")
        met_img     = nib.load(mrsi_path); 
        header_mrsi = met_img.header 
        #
        cat12_dirPath = split(mridb.find_nifti_paths("_desc-p1_T1w"))[0]
        p1_img  = nib.load(mridb.find_nifti_paths("_desc-p1_T1w"))
        p2_img  = nib.load(mridb.find_nifti_paths("_desc-p2_T1w"))
        p3_img  = nib.load(mridb.find_nifti_paths("_desc-p3_T1w"))
        # p5_img  = nib.load(p5_path)
        # p6_img  = nib.load(p6_path)


        # Correct T1W space
        t1w_img         = nib.load(mridb.find_nifti_paths("desc-brain_T1w"))
        affine          = t1w_img.affine
        self.p1_img     = resample_img(p1_img, target_affine=affine, target_shape=t1w_img.shape)
        self.p2_img     = resample_img(p2_img, target_affine=affine, target_shape=t1w_img.shape)
        self.p3_img     = resample_img(p3_img, target_affine=affine, target_shape=t1w_img.shape)



        if space=="anat" or space.lower()=="t1w":
            filename = f"{mridb.prefix}_space-orig_desc-4Dtissue_T1w.nii.gz"
            outpath  = join(cat12_dirPath,filename)
            header   = t1w_img.header
        elif space=="mrsi":
            transform_list  = mridb.get_transform("inverse","mrsi")
            self.p1_img     = reg.transform(met_img,p1_img,transform=transform_list).to_nibabel()
            self.p2_img     = reg.transform(met_img,p2_img,transform=transform_list).to_nibabel()
            self.p3_img     = reg.transform(met_img,p3_img,transform=transform_list).to_nibabel()
            filename = f"{mridb.prefix}_space-mrsi_desc-4Dtissue_T1w.nii.gz"
            outpath  = join(cat12_dirPath,filename)
            header   = met_img.header
            


        # Create 4D nifti
        tissue_4d          = np.zeros(self.p1_img.shape+(3,))
        tissue_4d[:,:,:,0] = self.p1_img.get_fdata() 
        tissue_4d[:,:,:,1] = self.p2_img.get_fdata()
        tissue_4d[:,:,:,2] = self.p3_img.get_fdata()
        ftools.save_nii_file(tissue_4d,header=header,outpath=outpath)
        self.tissue4D_path = outpath
        return outpath

    def proc(self,mridb, mrsi_path,tissue_mask_space="t1w",tissue="GM",nthreads=16):
        os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(nthreads)
        input_space = mridb.extract_metadata(mrsi_path)["space"]
        tissue_mask_space = "mrsi" if input_space=="orig" else input_space
        out_dict                           = dict()
        #
        bids_suffixes = mridb.extract_metadata(mrsi_path)
        metabolite    = bids_suffixes["met"]
        filt          = bids_suffixes["filt"]
        filter        = f"filt{filt}"

        tissue4D_path = self.create_3tissue_pev(mridb,
                                                space=tissue_mask_space)


        if not exists(mrsi_path):
            debug.error("MRSI path does not exist", mrsi_path)
            return
        if not exists(tissue4D_path):
            debug.error("tissue4D_path does not exist or not created")
            debug.error("Run create_3tissue_pev() first")
            return
        
        mrsi_corr_path = mrsi_path.replace("_mrsi.nii.gz","_pvcorr_mrsi.nii.gz")

        command = [
            "petpvc",
            "-i", mrsi_path,
            "-m", self.tissue4D_path,
            "-p", "RBV",
            "-x", "5",
            "-y", "5",
            "-z", "5",
            "-o", mrsi_corr_path
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            debug.error("Command failed with error code: {}".format(e.returncode))
            debug.error("STDOUT: " + e.stdout)
            debug.error("STDERR: " + e.stderr)
            return
        except FileNotFoundError:
            debug.error("petpvc command not found. Make sure it is installed and in PATH.")
            return
        except Exception as e:
            debug.error(f"Unexpected error occurred: {str(e)}")
            return
        # debug.success("petpvc DONE")
        ################## Postprocess PV corrected output ##################
        met_uncorr_img        = nib.load(mrsi_path)
        mask_space_mrsi_np    = mridb.get_mri_nifti("mrsi","orig","brainmask").get_fdata().astype(bool)

        # Apply brain mask
        img_corr = nib.load(mrsi_corr_path)
        img_np   = img_corr.get_fdata()
        img_np[img_np>2*met_uncorr_img.get_fdata()] = 0 # remove signal spikes
        img_np[img_np<0] = 0

        # Weight by tissue contribution
        for tissue in self._tissue_list:
            pvcorr_str = f"pvcorr_{tissue}" if tissue is not None else f"pvcorr"
            tissue_out_path = mrsi_corr_path.replace("_pvcorr",f"_{pvcorr_str}")
            if tissue=="GM":
                img_np_tissue = img_np * self.p1_img.get_fdata()
            elif tissue=="WM":
                img_np_tissue = img_np * self.p2_img.get_fdata()
            elif tissue=="CSF":
                img_np_tissue = img_np * self.p3_img.get_fdata()
            elif tissue is None:
                img_np_tissue   = img_np
                tissue_out_path = mrsi_corr_path

            # Save background masked image in t1w space
            ftools.save_nii_file(img_np_tissue.astype(np.float32),
                                 outpath=tissue_out_path,
                                 header=img_corr.header)


            # Transform to MRSI space T1 res
            if "space-t1w" in mrsi_path.lower():
                transform_list                = mridb.get_transform("inverse", "mrsi")
                met_img_corr_space_mrsi_t1res = reg.transform(met_uncorr_img, tissue_out_path, transform_list).to_nibabel()
                tissue_out_path_t1res         = mridb.get_mri_filepath(modality="mrsi",space="orig",
                                                                    desc="signal",met=metabolite,
                                                                    option=f"{filter}_{pvcorr_str}_t1res")        
                ftools.save_nii_file(met_img_corr_space_mrsi_t1res,outpath=tissue_out_path_t1res)



                # Transform to MRSI space native res
                met_og_res_path = mridb.get_mri_filepath(modality="mrsi",space="orig",
                                                        desc="signal",met=metabolite,option=filter)
                met_img_corr_space_mrsi_ogres     = reg.transform(met_og_res_path, tissue_out_path, transform_list).to_nibabel()
                met_img_corr_space_mrsi_ogres_np  = met_img_corr_space_mrsi_ogres.get_fdata()
                met_img_corr_space_mrsi_ogres_np[~mask_space_mrsi_np] = 0
                mrsi_header                       = met_img_corr_space_mrsi_ogres.header
                met_img_corr_space_mrsi_ogres     = ftools.numpy_to_nifti(met_img_corr_space_mrsi_ogres_np,mrsi_header)

                tissue_out_path_ogres             = mridb.get_mri_filepath(modality="mrsi",space="orig",
                                                            desc="signal",met=metabolite,
                                                            option=f"{filter}_{pvcorr_str}")  
                
                ftools.save_nii_file(met_img_corr_space_mrsi_ogres,outpath=tissue_out_path_ogres)

                out_dict[f"met-{metabolite}_{pvcorr_str}_space-orig_res-t1"] = tissue_out_path_t1res
                out_dict[f"met-{metabolite}_{pvcorr_str}_space-orig_res-og"] = tissue_out_path_ogres
                out_dict[f"met-{metabolite}_{pvcorr_str}_space-t1w_res-og"]  = tissue_out_path
                # debug.success(f"petpvc {tissue} DONE")

        return out_dict



if __name__=="__main__":
    metabolite = "GPCPCh"
    pv_corr_str = "pvcorr"
    METABOLITES         = ["NAANAAG", "Ins", "GPCPCh", "GluGln", "CrPCr"]
    mridata = MRIData(group="LPN-Project",subject_id="CHUVA015",session="V5")

    pve = PVECorrection()
    p1_path      = mridata.find_nifti_paths("_desc-p1_T1w")
    p2_path      = mridata.find_nifti_paths("_desc-p2_T1w")
    p3_path      = mridata.find_nifti_paths("_desc-p3_T1w")
    debug.info(p1_path)
    debug.info(p2_path)
    debug.info(p3_path)

    # mridata = MRIData(group="Mindfulness-Project",subject_id="S047",session="V1")

    # metab_path = mridata.get_mri_filepath(modality="mrsi",space="t1w",
    #                                              desc="signal",met=metabolite,option="filtbiharmonic")
    for met in METABOLITES:
        metab_path = mridata.get_mri_filepath(modality="mrsi",space="orig",
                                                    desc="signal",met=met,option="filtbiharmonic")
        debug.info("sub-CHUVA015_ses-V5_space-orig_met-NAANAAG_desc-signal_filtbiharmonic_mrsi.nii.gz")
        debug.info(exists(metab_path),split(metab_path)[1])
        # _,label_path = mridata.get_parcel("orig","LFMIHIFIS",scale=3)
        # p1_path      = mridata.find_nifti_paths("desc-p0_T1w")

        # pve.create_3tissue_pev(mridata,space="mrsi")
        # debug.info(exists(metab_path),"metab_path","\t",metab_path)
        out_dict = pve.proc(mridata, metab_path, tissue_mask_space="mrsi")
        # for k,v in out_dict.items():
        #     debug.info(exists(v),k,"\t",v)



