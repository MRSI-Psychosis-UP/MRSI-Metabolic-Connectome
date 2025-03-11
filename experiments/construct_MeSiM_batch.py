import os, sys, subprocess, time
import numpy as np
import json
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from os.path import split, join, exists
from tools.filetools import FileTools
from tools.debug import Debug


dutils   = DataUtils()
debug    = Debug()
pb       = ProgressBar()


###############################################################################
GROUP       = "Mindfulness-Project"
#GROUP       = "LPN-Project"

# ATLAS_ARR   = ["LFMIHIFIF-3","LFMIHIFIF-2","LFMIHIFIF-4","aal","geometric_cubeK18mm","mist-197"]
ATLAS_ARR   = ["schaefer-200"]
# ATLAS_ARR   = ["LFMIIIFIF-3","LFMIIIFIF-2","LFMIIIFIF-4"]


ftools      = FileTools(GROUP)
# N_PERT      = 1
N_PERT_LIST = [50]
EXEC_PATH = join(dutils.DEVANALYSEPATH,"connectomics","metabolic_simmatrix_subject.py")
################################################################################
############ List all subjects ##################
recording_list = np.array(ftools.list_recordings())
duration       = 35
for atlas in ATLAS_ARR:
    for N_PERT in N_PERT_LIST:
        for ids, recording in enumerate(recording_list):
            subject_id,session = recording
            start = time.time()
            debug.title(f"Processing {subject_id}-{session} with atlas {atlas}")
            try:
                subprocess.run(["python3",EXEC_PATH,"--atlas",atlas,"--n_pert",str(N_PERT),
                                "--subject_id",subject_id,"--session",session,"--group",GROUP,
                                "--nthreads","30","--leave_one_out","0","--overwrite","1","--show_plot","0"])
            except subprocess.CalledProcessError:
                debug.error("An error occurred during execution.")
                # thread.join()
                continue
            except KeyboardInterrupt:
                debug.error("KeyboardInterrupt")
                # thread.join()
                sys.exit()
            duration = round(time.time()-start)
            debug.success(f"Done in {duration} seconds")
        
