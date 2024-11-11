# ================ define paths ================
_cwd=$(pwd)
_paper_dir=$(dirname "${_cwd}")
_this_path="${_cwd}/settings.sh"  # the location of this file
echo "[*] Using environment settings from ${_this_path}."

export PROJECT_DIR=$_paper_dir
export OUTPUT_DIR=${PROJECT_DIR}/output

