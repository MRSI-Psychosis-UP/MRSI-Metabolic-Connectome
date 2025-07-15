import os, sys, subprocess, time
import numpy as np
import json, argparse
import pandas as pd
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from os.path import split, join, exists, abspath
from tools.debug import Debug
from tools.mridata import MRIData


dutils   = DataUtils()
debug    = Debug()
pb       = ProgressBar()


###############################################################################
parser = argparse.ArgumentParser(description="Process some input parameters.")
# Parse arguments
parser.add_argument('--group', type=str,default="Mindfulness-Project") 
parser.add_argument('--nthreads'  , type=int, default = 4,help="Number of CPU threads [default=4]")
parser.add_argument('--overwrite' , type=int, default=0, choices = [1,0],help="Overwrite coregistered maps (default: 0)")
parser.add_argument('--overwrite_pve' , type=int, default=0, choices = [1,0],help="Overwrite partial volume correction (default: 0)")
parser.add_argument('--overwrite_filt' , type=int, default=0, choices = [1,0],help="Overwrite filtered MRSI raw orig space (default: 0)")
parser.add_argument('--t1pattern' , type=str, default=None,help="Anatomical T1w file pattern")
parser.add_argument('--participants', type=str, default=None,
                    help="Path to TSV file containing list of participant IDs and sessions to include. If not specified, process all.")
parser.add_argument('--b0'        , type=float, default = 3,choices=[3,7],help="MRI B0 field strength in Tesla [default=3]")


args               = parser.parse_args()
group              = args.group
overwrite          = args.overwrite
participants_file  = args.participants
nthreads           = args.nthreads
t1pattern          = args.t1pattern
B0_strength        = str(args.b0)
overwrite_pvcorr   = str(args.overwrite_pve)
overwrite_filt     = str(args.overwrite_filt)

###############################################################################
EXEC_PATH = join(dutils.DEVANALYSEPATH,"experiments","scripts","preprocess.py")
################################################################################
if participants_file is None:
    participant_session_list = join(dutils.BIDSDATAPATH,group,"participants_allsessions.tsv")
    df                       = pd.read_csv(participant_session_list, sep='\t')
    df = df[df.session_id != "V2BIS"]
else:
    df = pd.read_csv(participants_file, sep='\t')
    
subject_id_list = df.participant_id.to_list()
session_id_list = df.session_id.to_list()

############ Process all subjects ##################
os.system("clear")
for i,(subject_id,session) in enumerate(zip(subject_id_list,session_id_list)):
    debug.title(f"Processing sub-{subject_id}_ses-{session} ---- {i}/{len(session_id_list)}")
    if t1pattern is not None:
        mrsiData = MRIData(subject_id,session,group)
        t1_path  = mrsiData.find_nifti_paths(t1pattern)
        if t1_path is not None:
            debug.info("Found t1 image",t1_path) 
        else:
            debug.error(group,subject_id,session,f"Could not find a t1 image matching the pattern [{t1pattern}]. Skip...")
            continue
    else:   
        t1_path = ""
    # Start space transformation
    start = time.time()
    try:
        subprocess.run(["python3",EXEC_PATH,"--t1",t1_path,
                        "--subject_id",str(subject_id),
                        "--session",session,
                        "--group",group,
                        "--b0",B0_strength,
                        "--overwrite",str(overwrite),
                        "--overwrite_pve",overwrite_pvcorr,
                        "--overwrite_filt",overwrite_filt,
                        "--nthreads",str(nthreads),
                        ])
    except subprocess.CalledProcessError:
        debug.error("An error occurred during execution.")
        continue
    except KeyboardInterrupt:
        debug.error("KeyboardInterrupt")
        sys.exit()
    duration = round(time.time()-start)
    debug.success(f"Done in {duration} seconds")
    debug.separator()

