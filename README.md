# Reservoir FDC Alteration and HAPI

Analysis code for the manuscript *Large-sample assessment of reservoir-associated flow-duration-curve alteration and a hydrological alteration prioritization index across the United States, with a preliminary comparison in Brazil*.

## Scope

The repository contains the scripts for data screening, flow-duration-curve metrics, functional clustering, information-theoretic summaries, association analyses, HAPI scoring, sensitivity analyses, figures, and the lightweight reproduction checks. HAPI is a transparent hydrological screening score for prioritizing reservoirs for follow-up assessment. It is not an ecological-benefit validation or an operating prescription.

## Data

The primary input archives are public Zenodo deposits:

- ResOpsUS+CARS: https://doi.org/10.5281/zenodo.15978041
- ResOpsBR+CARS: https://doi.org/10.5281/zenodo.16096623

The repository does not include the large raw archives or derived source-data tables. Downloaded files should be placed under the project `data/raw/` directory and processed outputs under `data/processed/`.

Google Earth Engine scripts require an Earth Engine account and are separate from the local core pipeline. The manuscript submission package contains the derived remote-sensing products used in the paper.

## Environment

Install the Python dependencies listed in `requirements.txt`. The core analysis uses Python 3, pandas, NumPy, SciPy, scikit-learn, matplotlib, geopandas, and related scientific packages. `econml`, `dbfread`, and Earth Engine are used only by the corresponding analysis steps.

## Reproduction

Run the scripts from the repository root after placing the public input archives in the expected data directories. The main local entry point is:

```bash
python3 code/00_utils/reproduce_full_pipeline.py
```

This entry point does not download data automatically. The saved remote-sensing products are treated as external Google Earth Engine outputs, as described in the manuscript.

## License

Code is released under the MIT License. Third-party datasets retain their original licenses and terms.
