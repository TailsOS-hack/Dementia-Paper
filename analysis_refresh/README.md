# Reproduce the current refresh

Install `requirements.txt` in a virtual environment. All commands run from the repository root. Models use Git LFS. No command here trains or selects a model.

1. Download the Mendeley archive from the URL in `datasets/mendeley_xx9zzz6t54_v1/README.md`, verify its recorded SHA-256, and extract it to a local data folder.
2. Audit against the original project's legacy manifests with `python datasets/mendeley_xx9zzz6t54_v1/audit_source.py --source-project /path/to/original/project`. The recorded reference hashes are in `legacy_hash_reference.csv` beside the audit.
3. Run `python analysis_refresh/evaluate_mendeley.py --data /path/to/extracted` to evaluate all ten frozen stage classifiers on the supplied test images. This is explicitly reused-data evaluation.
4. Run `python analysis_refresh/refresh_results.py --clinical-predictions /path/to/private/patient_predictions.json` to recompute the retained clinical results, plot the stage predictions and redraw all twelve raw training histories. Clinical patient-level predictions must be obtained from the authorized local source; they are not public repository files.
5. Check `results/current/verification.json` and visually inspect every generated plot. Public aggregate values are checked against the retained original scenario table at strict tolerance.

Current clinical and stage endpoints remain separate. OASIS access is pending; its intake tool does not authorize training. No recorded model is promoted or replaced by this refresh.
