# Data preparation

This bundle does **not** ship the raw CAMELS-DE data (it is hundreds of MB
and is governed by the dataset's own license). Use the two scripts in this
directory to fetch and preprocess it.

```bash
bash download_camels_de.sh          # ~3 GB into ./raw/
python preprocess.py --raw ./raw --out .
```

This produces three files used by `reproduce.py`:

| File | Used by | Shape |
|---|---|---|
| `camels_de.pkl`               | dHBV forward | tuple (forcings 1347×14975×3, target 1347×14975×1, attrs 1347×15) |
| `gage_info.npy`               | dHBV forward | 1347 catchment IDs in dataset order |
| `data_test_CAMELS_DE1.00.csv` | LSTM forward | catchment_name, Date, P, T, PET, Q rows for 2010-01-01 .. 2020-12-31 |

## Conventions

- **Time window**: 1980-01-02 to 2020-12-31 (14,975 days)
- **Catchments**: 1,347 (subset listed in `selected_catchments_1347.csv`)
- **Forcings**: daily P (mm/d), T_mean (°C), PET (mm/d); PET is taken from
  CAMELS-DE's `timeseries_simulated/CAMELS_DE_discharge_sim_<id>.csv`
  (`pet_hargreaves` column, i.e. Hargreaves PET as supplied with the dataset)
- **Attributes (15)**: p_mean, pet_mean, aridity, p_seasonality, frac_snow,
  high_prec_freq, high_prec_dur, low_prec_freq, low_prec_dur, elev_mean,
  area_gages2, frac_forest, sand_frac, silt_frac, clay_frac — computed
  from the CAMELS-DE static and climatic-indices tables

## CAMELS-DE source

CAMELS-DE 1.0 (Loritz et al., Zenodo). The `download_camels_de.sh` script
fetches the archive directly from the Zenodo record used by this bundle.
