import os, sys, subprocess, time
import numpy as np
import json, argparse
import pandas as pd
from tools.datautils import DataUtils
from os.path import split, join, exists, abspath
from tools.debug import Debug
from tools.mridata import MRIData


dutils    = DataUtils()
debug     = Debug()
LOG_DIR   = dutils.ANALOGPATH
EXEC_PATH = join(dutils.DEVANALYSEPATH,"experiments","scripts","registration_mrsi_to_t1.py")
log_file_path = join(LOG_DIR,split(EXEC_PATH)[1].replace(".py",""),"full_run.log")
os.makedirs(split(log_file_path)[0],exist_ok=True)
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
parser.add_argument('--b0'        , type=float, default = 3,choices=[3,7],help="MRI B0 field strength in Tesla [default=3]")


args               = parser.parse_args()
group              = args.group
overwrite          = args.overwrite
ref_met            = args.ref_met
participants_file  = args.participants
nthreads           = args.nthreads
t1pattern          = args.t1pattern
B0_strength        = str(args.b0)



###############################################################################

################################################################################
if participants_file is None:
    participant_session_list = join(dutils.BIDSDATAPATH,group,"participants_allsessions.tsv")
    df                       = pd.read_csv(participant_session_list, sep='\t')
    df = df[df.session_id != "V2BIS"]
else:
    df = pd.read_csv(participants_file, sep='\t')
    
subject_id_list = df.participant_id.to_list()
session_id_list = df.session_id.to_list()
n_to_process    = len(subject_id_list)

############ Process all subjects ##################
os.system("clear")
with open(log_file_path, 'a') as logfile:
    logfile.write(f"\n{'#' * 60}\nStarting new run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'#' * 60}\n")

    for subject_id, session in zip(subject_id_list, session_id_list):
        header = f"\n--- Processing sub-{subject_id}_ses-{session} ---\n"
        debug.title(header.strip())
        logfile.write(header)

        if t1pattern is not None:
            mrsiData = MRIData(subject_id, session, group)
            t1_path = mrsiData.find_nifti_paths(t1pattern)
            if t1_path is not None:
                msg = f"Found t1 image: {t1_path}"
                debug.info(msg)
                logfile.write(f"{msg}\n")
            else:
                msg = f"Could not find a t1 image matching the pattern [{t1pattern}]. Skipping..."
                debug.error(group, subject_id, session, msg)
                logfile.write(f"{msg}\n")
                continue
        else:
            t1_path = ""

        start = time.time()

        try:
            cmd = [
                "python3", EXEC_PATH,
                "--ref_met", ref_met,
                "--t1", t1_path,
                "--subject_id", str(subject_id),
                "--session", session,
                "--group", group,
                "--overwrite", str(overwrite),
                "--nthreads", str(nthreads)
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output to both shell and log
            for line in process.stdout:
                print(line, end='')        # show in terminal
                logfile.write(line)        # write to log

            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)

        except subprocess.CalledProcessError as e:
            msg = "An error occurred during execution."
            debug.error(msg)
            logfile.write(f"{msg}\n{str(e)}\n")
            continue
        except KeyboardInterrupt:
            msg = "KeyboardInterrupt detected. Exiting..."
            debug.error(msg)
            logfile.write(f"{msg}\n")
            sys.exit()

        duration = round(time.time() - start)
        msg = f"Done in {duration} seconds"
        debug.success(msg)
        debug.separator()
        logfile.write(f"{msg}\n{'-' * 40}\n")