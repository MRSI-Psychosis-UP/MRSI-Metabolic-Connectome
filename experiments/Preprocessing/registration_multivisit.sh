#!/bin/bash

# This script performs registration for multivisit MRI data using ANTs.
set -o pipefail

display_help() {
    echo "Usage: $0 -i <BIDSDIR>"
    echo
    echo "Options:"
    echo "  -i, --input     Path to the BIDS directory containing multivisit MRI data."
    echo "  -s,    Process only the specified subject (e.g., S001 -> sub-S001)."
    echo "      --t1        Glob pattern (without subject/ses prefix) to locate T1 files."
    echo "      --mni       Path to the MNI reference brain."
    echo "  -h, --help      Display this help message."
    echo
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_info() {
    local message="$1"
    local ts
    ts=$(timestamp)
    echo "[$ts] [INFO] ${message}"
    [[ -n "${LOG_FILE:-}" ]] && echo "[$ts] [INFO] ${message}" >> "${LOG_FILE}"
}

log_error() {
    local message="$1"
    local ts
    ts=$(timestamp)
    echo "[$ts] [ERROR] ${message}" >&2
    [[ -n "${LOG_FILE:-}" ]] && echo "[$ts] [ERROR] ${message}" >> "${LOG_FILE}"
}

run_step() {
    local description="$1"
    shift
    log_info "START: ${description}"
    "$@"
    local status=$?
    if [[ ${status} -ne 0 ]]; then
        log_error "FAILED: ${description} (exit code ${status})"
        exit ${status}
    fi
    log_info "DONE: ${description}"
}

BIDSDIR=""
SUBJECT_FILTER=""
T1_PATTERN="*_desc-brain_T1w.nii.gz"
MNI_TEMPLATE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    -h | --help)
        display_help
        exit 0
        ;;
    -i | --input)
        if [[ -z $2 || $2 == -* ]]; then
            echo "Error: --input requires a non-empty argument."
            exit 1
        fi
        BIDSDIR="$2"
        shift 2
        ;;
    -s | --subject)
        if [[ -z $2 || $2 == -* ]]; then
            echo "Error: --subject requires a non-empty argument."
            exit 1
        fi
        SUBJECT_FILTER="$2"
        shift 2
        ;;
    --t1)
        if [[ -z $2 || $2 == -* ]]; then
            echo "Error: --t1 requires a non-empty argument."
            exit 1
        fi
        T1_PATTERN="$2"
        shift 2
        ;;
    --mni)
        if [[ -z $2 || $2 == -* ]]; then
            echo "Error: --mni requires a non-empty argument."
            exit 1
        fi
        MNI_TEMPLATE="$2"
        shift 2
        ;;
    *)
        echo "Unknown option: $1"
        display_help
        exit 1
        ;;
    esac
done

if [[ -z "${BIDSDIR}" ]]; then
    echo "Error: --input <BIDSDIR> is required."
    exit 1
fi

if [[ -z "${MNI_TEMPLATE}" ]]; then
    echo "Error: --mni <path> is required."
    exit 1
fi

if [[ ! -f "${MNI_TEMPLATE}" ]]; then
    echo "Error: MNI reference not found at ${MNI_TEMPLATE}"
    exit 1
fi

if [[ ! -d "${BIDSDIR}" ]]; then
    echo "Error: BIDSDIR not found at ${BIDSDIR}"
    exit 1
fi

LOG_DIR="${BIDSDIR}/derivatives/logs"
mkdir -p "${LOG_DIR}" || { echo "Unable to create log directory at ${LOG_DIR}"; exit 1; }
LOG_FILE="${LOG_DIR}/registration_multivisit_$(date +%Y%m%d_%H%M%S).log"
touch "${LOG_FILE}" || { echo "Unable to write to log file ${LOG_FILE}"; exit 1; }

log_info "Starting registration for ${BIDSDIR}"

shopt -s nullglob

# Go to directory with T1w skulstripped images
T1W_DIR="${BIDSDIR}/derivatives/skullstrip"

if [[ ! -d "${T1W_DIR}" ]]; then
    log_error "Directory not found: ${T1W_DIR}"
    exit 1
fi

run_step "Change directory to ${T1W_DIR}" cd "${T1W_DIR}"

subject_dirs=()
if [[ -n "${SUBJECT_FILTER}" ]]; then
    if [[ "${SUBJECT_FILTER}" != sub-* ]]; then
        SUBJECT_FILTER="sub-${SUBJECT_FILTER}"
    fi
    SUBJECT_PATH="${T1W_DIR}/${SUBJECT_FILTER}"
    if [[ ! -d "${SUBJECT_PATH}" ]]; then
        log_error "Specified subject directory not found: ${SUBJECT_FILTER}"
        exit 1
    fi
    subject_dirs=( "${SUBJECT_PATH}" )
else
    for SUBJECT_DIR in "${T1W_DIR}"/sub-*; do
        [[ -d "${SUBJECT_DIR}" ]] && subject_dirs+=( "${SUBJECT_DIR}" )
    done
    if [[ ${#subject_dirs[@]} -eq 0 ]]; then
        log_error "No subject directories found in ${T1W_DIR}"
        exit 1
    fi
fi

# Loop through each subject
for SUBJECT_DIR in "${subject_dirs[@]}"; do
    [[ -d "${SUBJECT_DIR}" ]] || continue
    SUBJECT_ID=$(basename "${SUBJECT_DIR}")
    log_info "Processing subject: ${SUBJECT_ID}"

    run_step "Enter subject directory ${SUBJECT_DIR}" cd "${SUBJECT_DIR}"
    run_step "Ensure ses-all directory exists for ${SUBJECT_ID}" mkdir -p ses-all

    # Loop through each visit for the subject
    for VISIT_DIR in ${SUBJECT_DIR}/ses-*; do
        [[ -d "${VISIT_DIR}" ]] || continue
        VISIT_ID=$(basename "${VISIT_DIR}")
        if [[ "${VISIT_ID}" == "ses-all" ]]; then
            continue
        fi
        log_info "Found visit ${VISIT_ID} for ${SUBJECT_ID}"
        visit_files=( ${VISIT_DIR}/${SUBJECT_ID}_${VISIT_ID}_${T1_PATTERN} )
        if [[ ${#visit_files[@]} -eq 0 ]]; then
            log_error "No T1w file found for ${SUBJECT_ID} ${VISIT_ID}; skipping visit."
            continue
        fi
        for VISIT_FILE in "${visit_files[@]}"; do
            run_step "Copy $(basename "${VISIT_FILE}") to ses-all for ${SUBJECT_ID} ${VISIT_ID}" cp "${VISIT_FILE}" "ses-all/"
        done
    done

    run_step "Enter ses-all directory for ${SUBJECT_ID}" cd ses-all

    ses_all_files=( ${SUBJECT_ID}_ses-*_${T1_PATTERN} )
    if [[ ${#ses_all_files[@]} -eq 0 ]]; then
        log_error "No T1w files were aggregated for ${SUBJECT_ID}; skipping subject."
        run_step "Return to subject directory ${SUBJECT_DIR}" cd "${SUBJECT_DIR}"
        run_step "Return to ${T1W_DIR}" cd "${T1W_DIR}"
        continue
    fi

    # Perform registrations using ANTs
    run_step "Construct template for ${SUBJECT_ID}" \
        antsMultivariateTemplateConstruction2.sh -d 3 -i 4 -g 0.2 -k 1 -r 1 -v 16 -o "${SUBJECT_ID}_template_" "${ses_all_files[@]}"

    run_step "Register ${SUBJECT_ID} template to MNI" \
        antsRegistrationSyN.sh -d 3 -f "${MNI_TEMPLATE}" \
        -m "${SUBJECT_ID}_template_template0.nii.gz" -o "${SUBJECT_ID}_ses-all_desc-template_to_mni" -t s -n 16

    # Move registration to template in good directory
    run_step "Ensure transforms directory for ${SUBJECT_ID}/ses-all" mkdir -p "${BIDSDIR}/derivatives/transforms/ants/${SUBJECT_ID}/ses-all/anat"

    for VISIT_DIR in ${SUBJECT_DIR}/ses-*; do
        [[ -d "${VISIT_DIR}" ]] || continue
        VISIT_ID=$(basename "${VISIT_DIR}")
        if [[ "${VISIT_ID}" == "ses-all" ]]; then
            continue
        fi
        run_step "Ensure transforms directory for ${SUBJECT_ID}/${VISIT_ID}" mkdir -p "${BIDSDIR}/derivatives/transforms/ants/${SUBJECT_ID}/${VISIT_ID}/anat"

        affine_files=( ${SUBJECT_ID}_template*_${VISIT_ID}_*-0GenericAffine.mat )
        warp_files=( ${SUBJECT_ID}_template*_${VISIT_ID}_*-1Warp.nii.gz )
        invwarp_files=( ${SUBJECT_ID}_template*_${VISIT_ID}_*-1InverseWarp.nii.gz )

        if [[ ${#affine_files[@]} -eq 0 || ${#warp_files[@]} -eq 0 || ${#invwarp_files[@]} -eq 0 ]]; then
            log_error "Missing transform files for ${SUBJECT_ID} ${VISIT_ID}; skipping move for this visit."
            continue
        fi

        run_step "Move affine transform for ${SUBJECT_ID} ${VISIT_ID}" \
            mv "${affine_files[0]}" "${BIDSDIR}/derivatives/transforms/ants/${SUBJECT_ID}/${VISIT_ID}/anat/${SUBJECT_ID}_${VISIT_ID}_desc-t1w_to_template.affine.mat"
        run_step "Move warp transform for ${SUBJECT_ID} ${VISIT_ID}" \
            mv "${warp_files[0]}" "${BIDSDIR}/derivatives/transforms/ants/${SUBJECT_ID}/${VISIT_ID}/anat/${SUBJECT_ID}_${VISIT_ID}_desc-t1w_to_template.syn.nii.gz"
        run_step "Move inverse warp transform for ${SUBJECT_ID} ${VISIT_ID}" \
            mv "${invwarp_files[0]}" "${BIDSDIR}/derivatives/transforms/ants/${SUBJECT_ID}/${VISIT_ID}/anat/${SUBJECT_ID}_${VISIT_ID}_desc-t1w_to_template.syn_inv.nii.gz"
    done

    template_affine=( ${SUBJECT_ID}_ses-all_desc-template_to_mni0GenericAffine.mat )
    template_warp=( ${SUBJECT_ID}_ses-all_desc-template_to_mni1Warp.nii.gz )
    template_invwarp=( ${SUBJECT_ID}_ses-all_desc-template_to_mni1InverseWarp.nii.gz )

    if [[ ${#template_affine[@]} -eq 0 || ${#template_warp[@]} -eq 0 || ${#template_invwarp[@]} -eq 0 ]]; then
        log_error "Missing template-to-MNI transform files for ${SUBJECT_ID}"
        exit 1
    fi

    run_step "Move template-to-MNI affine for ${SUBJECT_ID}" \
        mv "${template_affine[0]}" "${BIDSDIR}/derivatives/transforms/ants/${SUBJECT_ID}/ses-all/anat/${SUBJECT_ID}_ses-all_desc-template_to_mni.affine.mat"
    run_step "Move template-to-MNI warp for ${SUBJECT_ID}" \
        mv "${template_warp[0]}" "${BIDSDIR}/derivatives/transforms/ants/${SUBJECT_ID}/ses-all/anat/${SUBJECT_ID}_ses-all_desc-template_to_mni.syn.nii.gz"
    run_step "Move template-to-MNI inverse warp for ${SUBJECT_ID}" \
        mv "${template_invwarp[0]}" "${BIDSDIR}/derivatives/transforms/ants/${SUBJECT_ID}/ses-all/anat/${SUBJECT_ID}_ses-all_desc-template_to_mni.syn_inv.nii.gz"

    run_step "Removing temp files ${SUBJECT_ID}" \
        rm -rf "intermediateTemplates"

    run_step "Removing temp files ${SUBJECT_ID}" \
        find . -maxdepth 1 -type f ! -name "${SUBJECT_ID}_template_template0*" -delete
    

    run_step "Return to subject directory ${SUBJECT_DIR}" cd "${SUBJECT_DIR}"
    run_step "Return to ${T1W_DIR}" cd "${T1W_DIR}"
done

log_info "Registration completed for all subjects in ${BIDSDIR}"
