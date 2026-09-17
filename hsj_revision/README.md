# Temporal persistence and reservoir sample eligibility

Version 2.0.0 contains the code, derived numerical tables and figures for the revised decadal flow-duration-shape benchmark. The historical v1.0.2 release remains available for the earlier analysis. This archive does not represent acceptance or publication by a journal.

## Main results

Persistence MAE is 0.07730 across 197 US test reservoirs and 0.05084 across 91 Brazilian reservoirs. A source change retains 154 US reservoirs and excludes 43 with larger errors on the unchanged CARS targets. Fixed-cohort comparisons separate this composition effect from changes in source values. Full model and sensitivity results, including unsuccessful models, are retained.

## Reproduction using saved results

From this directory, install requirements.txt, then run:

```sh
python code/tests/test_revision_phase2.py
python code/03_analysis/plot_hsj_main_figures.py
python code/03_analysis/plot_hsj_population_benchmark.py
```

The tests cover signature definitions, window coverage and prediction chronology. Figure scripts read saved predictions and case tables, without downloads or refitting. All 16 chronological MAEs were checked against saved predictions at tolerance 1e-10. The requirements record the reporting environment; it is not a reconstructed historical training environment.

## Recomputing from original inputs

Source datasets are distributed by their providers under their own terms:

- ResOpsUS+CARS v1.0: https://doi.org/10.5281/zenodo.15978041
- ResOpsBR+CARS v1.0: https://doi.org/10.5281/zenodo.16096623
- Upstream ResOpsUS Version 2: https://doi.org/10.5281/zenodo.6612040
- BasinATLAS: https://www.hydrosheds.org/hydroatlas

Raw flow archives and BasinATLAS polygons are not redistributed. Place the CARS directories under data/raw/ResOpsUS+CARS_v10/v1.0 and data/raw/ResOpsBR+CARS_v10/v1.0. Place the upstream ZIP at data/raw/phase2_resops_original/ResOpsUS 2.zip and the geodatabase at data/enhancement_probe/basinatlas/BasinATLAS_v10.gdb.

Run revision_temporal_validation.py, revision_phase2.py, cohort_source_sensitivity.py, gap_processing_sensitivity.py, persistence_failure_warning.py and storage_state_warning.py in that order from code/03_analysis. Use a working copy because the analysis writes its output tables. The saved cohort and folds allow comparisons with the archived results. Processing reconstruction and case scripts provide additional audits. The two upstream inventory tables are metadata accompanying the source archive and retain its source attribution.

A fresh-input, clean-environment run has not been completed for this release. Saved-result checks and processing-replay results have narrower scopes. Existing provenance files describe the original runs. The original dated scientific protocol is preserved for its checksum; its initial volume-share terminology was subsequently narrowed to archived flow-rate-sum shares in the manuscript.

## Files and rights

Analysis code is released under the repository MIT license. Derived summary results and original figure assets are provided under CC BY 4.0; upstream metadata retains the source dataset terms. Scientific-use citations should also acknowledge the original source datasets. SHA256SUMS.json records the packaged files.
