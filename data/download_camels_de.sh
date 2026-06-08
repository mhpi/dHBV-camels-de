#!/usr/bin/env bash
# Download the CAMELS-DE 1.0 archive from Zenodo (~2.2 GB).
# Zenodo record: https://zenodo.org/records/13837553
# The record ships a single archive `camels_de.zip` that expands to
#   timeseries/                            (per-catchment hydromet CSVs)
#   timeseries_simulated/                  (Hargreaves PET in discharge_sim CSVs)
#   CAMELS_DE_climatic_attributes.csv
#   CAMELS_DE_topographic_attributes.csv
#   CAMELS_DE_soil_attributes.csv
#   CAMELS_DE_landcover_attributes.csv
#   ... (other attribute CSVs)
# preprocess.py reads from this expanded layout under --de-root.
set -euo pipefail

OUT_DIR="${1:-./raw}"
mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

URL="https://zenodo.org/records/13837553/files/camels_de.zip"
ZIP="camels_de.zip"

if [[ ! -f "$ZIP" ]]; then
    echo "Downloading $ZIP from $URL ..."
    wget -q --show-progress "$URL"
else
    echo "Already have $ZIP, skipping download."
fi

# Unpack into the same directory. The zip's top-level layout matches what
# preprocess.py expects when passed --de-root pointing at this directory.
echo "Unzipping $ZIP ..."
unzip -q -o "$ZIP"

echo "Done. Raw CAMELS-DE is in $OUT_DIR"
echo "Next: python data/preprocess.py --de-root $OUT_DIR --selected data/selected_catchments_1347.csv --out-dir data/"
