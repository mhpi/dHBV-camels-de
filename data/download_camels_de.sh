#!/usr/bin/env bash
# Download the CAMELS-DE 1.0 archive from Zenodo (~3 GB).
# Zenodo record: https://zenodo.org/records/13837553
set -euo pipefail

OUT_DIR="${1:-./raw}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

BASE="https://zenodo.org/records/13837553/files"
FILES=(
    "CAMELS_DE_attributes.zip"
    "CAMELS_DE_climatic_attributes.csv"
    "CAMELS_DE_hydrometeorology.zip"
    "CAMELS_DE_humaninfluence_attributes.csv"
    "CAMELS_DE_hydrologic_attributes.csv"
    "CAMELS_DE_landcover_attributes.csv"
    "CAMELS_DE_soil_attributes.csv"
    "CAMELS_DE_topographic_attributes.csv"
)

for f in "${FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "Downloading $f ..."
        wget -q --show-progress "$BASE/$f"
    else
        echo "Already have $f, skipping."
    fi
done

for z in CAMELS_DE_attributes.zip CAMELS_DE_hydrometeorology.zip; do
    if [[ -f "$z" && ! -d "${z%.zip}" ]]; then
        echo "Unzipping $z ..."
        unzip -q "$z"
    fi
done

echo "Done. Raw CAMELS-DE is in $OUT_DIR"
