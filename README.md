# Reservoir flow-duration benchmarks

Version **v2.0.0** accompanies *Temporal persistence as a benchmark for reservoir-associated flow-duration alteration*.

The current analysis, saved predictions, source and gap sensitivities, warning experiments, figures and reproduction instructions are in [hsj_revision](hsj_revision/README.md). It examines whether historical distributional shape predicts a later decade and separates source-dependent sample eligibility from changes in predictions.

Start in `hsj_revision/`, install its requirements and run `python code/tests/test_revision_phase2.py`. Saved-result checks do not download data. The README distinguishes those checks from a complete raw-data rerun and identifies the source archives.

The root `code/` and `requirements.txt` retain the historical workflow. The [v1.0.2 release](https://github.com/yuelangmanle/resops-fdc-hapi/releases/tag/v1.0.2) and DOI [10.5281/zenodo.22240369](https://doi.org/10.5281/zenodo.22240369) describe that earlier version, not the revised results. The current version does not use the earlier HAPI analysis as its main contribution.

Code is MIT licensed. Derived summary results and original figure assets in `hsj_revision/` are CC BY 4.0; source metadata retains its original dataset terms. Raw flow archives and BasinATLAS polygons are obtained from their providers.
