#!/usr/bin/env bash
# download_model_weights.sh
#
# Usage:
#   bash download_model_weights.sh               # Download into ./tool_weights
#   bash download_model_weights.sh /path/to/dir  # Custom download root directory
#
# The final directory structure will look like:
#   MODELS_ROOT/
#     af2/
#       params_model_{1..5}.npz
#       params_model_{1..5}_ptm.npz
#       params_model_{1..5}_multimer_v3.npz
#       LICENSE
#     mpnn/
#       ca_model_weights/
#       soluble_model_weights/
#       vanilla_model_weights/
#
# After downloading, update pxdbench/globals.py to point to these locations.

set -euo pipefail

# Set root directory for storing all model weights
MODELS_ROOT="${1:-$(pwd)/tool_weights}"

AF2_DIR="${MODELS_ROOT}/af2"
MPNN_DIR="${MODELS_ROOT}/mpnn"

echo "Model root directory: ${MODELS_ROOT}"
mkdir -p "${AF2_DIR}" "${MPNN_DIR}"
########################################
# 1. AlphaFold2 parameters
########################################
echo "==> Downloading AlphaFold2 parameters ..."

AF2_TAR="alphafold_params_2022-12-06.tar"
AF2_URL="https://storage.googleapis.com/alphafold/${AF2_TAR}"

# If AF2 parameters already exist, skip download
if compgen -G "${AF2_DIR}/params_model_1*.npz" > /dev/null; then
  echo "  AlphaFold2 params appear to already exist — skipping download."
else
  tmp_tar="${MODELS_ROOT}/${AF2_TAR}"
  echo "  Downloading from: ${AF2_URL}"
  curl -L "${AF2_URL}" -o "${tmp_tar}"

  echo "  Extracting to: ${AF2_DIR}"
  tar -xf "${tmp_tar}" -C "${AF2_DIR}"
  rm -f "${tmp_tar}"

  echo "  AlphaFold2 parameters downloaded successfully."
fi


########################################
# 2. ProteinMPNN weights
########################################
echo "==> Downloading ProteinMPNN weights ..."

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

echo "  Cloning dauparas/ProteinMPNN (shallow clone)..."
git clone --depth 1 https://github.com/dauparas/ProteinMPNN.git "${TMP_DIR}"

# Copy each weight directory
for subdir in ca_model_weights soluble_model_weights vanilla_model_weights; do
  src="${TMP_DIR}/${subdir}"
  dst="${MPNN_DIR}/${subdir}"

  if [ -d "${dst}" ]; then
    echo "  ${subdir} already exists — skipping."
  else
    echo "  Copying ${subdir} → ${dst}"
    mkdir -p "${MPNN_DIR}"
    cp -r "${src}" "${dst}"
  fi
done

echo "  ProteinMPNN weights are ready in: ${MPNN_DIR}"

########################################
# 3. CCD cache (PXDesign release_data)
########################################
echo "==> Downloading CCD cache ..."

CCD_DIR="${1:-$(pwd)/release_data/ccd_cache}"
mkdir -p "${CCD_DIR}"

CCD_COMPONENTS_URL="https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif"
CCD_RDKIT_URL="https://pxdesign.tos-cn-beijing.volces.com/release_data/components.v20240608.cif.rdkit_mol.pkl"
PDB_CLUSTER_URL="https://pxdesign.tos-cn-beijing.volces.com/release_data/clusters-by-entity-40.txt"

CCD_COMPONENTS_FILE="${CCD_DIR}/components.v20240608.cif"
CCD_RDKIT_FILE="${CCD_DIR}/components.v20240608.cif.rdkit_mol.pkl"
PDB_CLUSTER_FILE="${CCD_DIR}/clusters-by-entity-40.txt"

download_if_missing() {
  local url="$1"
  local out="$2"
  if [ -f "$out" ]; then
    echo "  $(basename "$out") already exists — skipping."
  else
    echo "  Downloading $(basename "$out")"
    curl -L -C - "$url" -o "$out"
  fi
}

download_if_missing "${CCD_COMPONENTS_URL}" "${CCD_COMPONENTS_FILE}"
download_if_missing "${CCD_RDKIT_URL}"     "${CCD_RDKIT_FILE}"
download_if_missing "${PDB_CLUSTER_URL}"   "${PDB_CLUSTER_FILE}"

echo "  CCD cache is ready in: ${CCD_DIR}"

########################################

echo "==> All downloads completed."
echo "Model weight directories:"
echo "  AF2:      ${AF2_DIR}"
echo "  MPNN:     ${MPNN_DIR}"
echo "CCD cache:  ${CCD_DIR}"
