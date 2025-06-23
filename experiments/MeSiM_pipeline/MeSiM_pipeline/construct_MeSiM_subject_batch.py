import os, sys, subprocess, time
import numpy as np
import json, argparse
import pandas as pd
from tools.progress_bar import ProgressBar
from tools.datautils import DataUtils
from os.path import split, join, exists, abspath
from tools.filetools import FileTools
from tools.debug import Debug
from tools.mridata import MRIData


dutils   = DataUtils()
debug    = Debug()
pb       = ProgressBar()


###############################################################################
def parse_parc(arg):
    """
    Parse a comma-separated list of parcellation scheme names and check they are valid.
    """
    valid_choices = ['LFMIHIFIS', 'LFMIHIFIF','LFIIIIFIS']
    # Split the string by comma and remove any extra spaces
    values = [v.strip() for v in arg.split(',')]
    # Check each value
    for v in values:
        if v not in valid_choices:
            raise argparse.ArgumentTypeError(f"Invalid choice: {v}. Must be one of {valid_choices}.")
    return values

def parse_list_int(arg):
    """
    Parse a comma-separated list of integers.
    """
    try:
        # Split string by comma, strip spaces and convert each piece to int.
        return [int(x.strip()) for x in arg.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected a comma-separated list of integers, got: {arg}")

###############################################################################
parser = argparse.ArgumentParser(description="Process some input parameters.")

# Use our custom parser functions as the type
parser.add_argument('--parc', type=parse_parc, default=["LFMIHIFIS"],
                    help='Chimera parcellation scheme(s) (comma separated), valid choices: LFMIHIFIS, LFMIHIFIF. Default: LFMIHIFIS')
parser.add_argument('--scale', type=parse_list_int, default=[3],
                    help="Cortical parcellation scale(s) as comma-separated integers (default: 3)")
parser.add_argument('--nthreads', type=int, default=4,
                    help="Number of parallel threads (default=4)")
parser.add_argument('--npert', type=parse_list_int, default=[50],
                    help='Number of perturbations as comma-separated integers (default: 50)')
parser.add_argument('--group', type=str, default="Dummy-Project")
parser.add_argument('--preproc', type=str, default="filtbiharmonic",
                    help="Preprocessing of orig MRSI files (default: filtbiharmonic)")
parser.add_argument('--leave_one_out', type=int, default=0, choices=[0, 1],
                    help="Leave-one-metabolite-out (default: 0)")
parser.add_argument('--overwrite', type=int, default=0, choices=[0, 1],
                    help="Overwrite existing parcellation (default: 0)")
parser.add_argument('--participants', type=str, default=None,
                    help="Path to TSV file containing list of participant IDs and sessions to include. If not specified, process all.")
parser.add_argument('--t1maskpattern' , type=str, default=None,help="Anatomical T1w mask file pattern")

args = parser.parse_args()

# The custom-parsed arguments are now lists:
group       = args.group
parc_list   = args.parc       # a list of parcellations, e.g. ['LFMIHIFIS'] or ['LFMIHIFIS', 'LFMIHIFIF']
scale_list  = args.scale      # a list of integers, e.g. [3]
npert_list  = args.npert      # a list of integers, e.g. [50]
nthreads    = args.nthreads
preproc     = args.preproc
lomo_bool   = args.leave_one_out
overwrite   = args.overwrite
participants_file = args.participants
t1maskpattern   = args.t1maskpattern

###############################################################################

# N_PERT      = 1
EXEC_PATH = join(dutils.DEVANALYSEPATH,"experiments","MeSiM_pipeline","construct_MeSiM_subject.py")
################################################################################
if participants_file is None:
    participant_session_list = join(dutils.BIDSDATAPATH,group,"participants_allsessions.tsv")
    df                       = pd.read_csv(participant_session_list, sep='\t')
    df = df[df.session_id != "V2BIS"]
else:
    df = pd.read_csv(participants_file, sep='\t')

subject_id_list = df.participant_id.to_list()
session_id_list = df.session_id.to_list()

# sys.exit()


############ List all subjects ##################
for parc in parc_list:
    for scale in scale_list:
        for npert in npert_list:
            for subject_id,session in zip(subject_id_list,session_id_list):
                start = time.time()
                debug.title(f"Processing sub-{subject_id}_ses-{session}_atlas-chimera{parc}_scale-{scale}_npert-{npert}")
                try:
                    if t1maskpattern is not None:
                        mrsiData = MRIData(subject_id,session,group)
                        t1_mask_path  = mrsiData.find_nifti_paths(t1maskpattern)
                        if t1_mask_path is not None:
                            debug.info("Found t1 image",t1_mask_path) 
                        else:
                            debug.error(group,subject_id,session,f"Could not find a t1 image matching the pattern [{t1maskpattern}]. Skip...")
                            continue
                    else:   
                        t1_path = ""
                    subprocess.run(["python3",EXEC_PATH,"--parc",parc,"--scale",str(scale),"--npert",str(npert),
                                    "--subject_id",subject_id,"--session",session,"--group",group,"--preproc",preproc,
                                    "--nthreads",str(nthreads),"--leave_one_out",str(lomo_bool),"--t1mask",t1_mask_path,
                                    "--overwrite",str(overwrite)])
                except subprocess.CalledProcessError:
                    debug.error("An error occurred during execution.")
                    continue
                except KeyboardInterrupt:
                    debug.error("KeyboardInterrupt")
                    sys.exit()
                duration = round(time.time()-start)
                debug.success(f"Done in {duration} seconds")
                debug.separator()
        
