import os, sys, subprocess, time
import numpy as np
import json, argparse
import pandas as pd
from tools.datautils import DataUtils
from os.path import split, join, exists, abspath
from tools.debug import Debug
from tools.mridata import MRIData
from pathlib import Path


dutils   = DataUtils()
debug    = Debug()


###############################################################################
parser = argparse.ArgumentParser(description="Process some input parameters.")
# Parse arguments
parser.add_argument('--group', type=str,default="Mindfulness-Project") 
parser.add_argument('--nthreads'  , type=int, default = 4,help="Number of CPU threads [default=4]")
parser.add_argument('--ref_met'   , type=str, default = "CrPCr",help="Reference metabolite to be coregistered with T1 [CrPCr]")
parser.add_argument('--overwrite' , type=int, default=0, choices = [1,0],help="Overwrite existing parcellation (default: 0)")
parser.add_argument('--t1pattern' , type=str, default=None,help="Anatomical T1w file pattern")
parser.add_argument('--participants', type=str, default=None,
                    help="Path to TSV file containing list of participant IDs and sessions to include. If not specified, process all.")


args               = parser.parse_args()
group              = args.group
overwrite          = args.overwrite
ref_met            = args.ref_met
participants_file  = args.participants
nthreads           = args.nthreads
t1pattern          = args.t1pattern



###############################################################################

# Define the base directory to start the search (e.g., DEVANALYSEPATH)
base_dir = Path(dutils.DEVANALYSEPATH)

# Use rglob to search for the file recursively (this pattern is case-sensitive)
matches = list(base_dir.rglob("registration_mrsi_to_t1.py"))

if matches:
    EXEC_PATH = str(matches[0])  # choose the first match, or add logic if multiple are found
else:
    raise FileNotFoundError(f"registration_mrsi_to_t1.py not found ")
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
for subject_id,session in zip(subject_id_list,session_id_list):
    debug.title(f"Processing sub-{subject_id}_ses-{session}ref_met-{ref_met}")
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

    start = time.time()
    
    try:
        subprocess.run(["python3",EXEC_PATH,"--ref_met",ref_met,"--t1",t1_path,
                        "--subject_id",subject_id,"--session",session,"--group",group,
                        "--overwrite",str(overwrite),"--nthreads",str(nthreads),
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

