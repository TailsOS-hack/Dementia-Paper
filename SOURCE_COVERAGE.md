# Current source coverage

Current research tables and regenerated plots are under `results/current/`. The STS Results manuscript and reproducible historical training/full-test/supporting-benchmark audits are under `results/sts/`, with source code in `analysis/sts_results/`. The active clinical scope contains two reused cohorts and zero fresh external clinical cohorts. Historical snapshots remain in Git history. Retired exact-cohort comparison scripts and combined artifacts were removed from the active tree; the generic statistical helpers and current validated runner replace them.

The Mendeley release is recorded as an alias of already-present image content, not additional training data. OASIS intake is implemented but actual access, acquisition, preprocessing and evaluation remain pending. Private clinical images, participant records, access receipts, credentials and operational material are not redistributed. Existing historical dementia-stage evidence and trained checkpoints remain included.

The six Round 1 runs completed 305 epochs. Their 5,450-image replay remains a post-hoc descriptive supplement containing the original 1,058-image primary comparison and nested 756-image sensitivity subset. Source clinical endpoints and the reused 100-image benchmark retain their separate interpretations. The October 18 training implementation and corrected loss-plot labels are separate from those historical measurements; new experiment outputs must identify their own run and checkpoint provenance.

The repository manifest includes new public code and outputs, excludes private/raw-data working bundles, and uses payload SHA256/size for LFS files. Local pointer identities do not constitute an independent rehash of unavailable model binaries.
