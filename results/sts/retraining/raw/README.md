# Verified October 18 retraining source evidence

Byte-identical public metadata, full histories, aggregate per-epoch evaluation JSON and run curves for six completed runs. The packaging step reran the aggregate verifier against all original checkpoint binaries before copying. Selected/last checkpoint hashes in the raw metadata identify the original resumable files, which remain local. Inference-state derivatives, when included, have separate hashes and explicit original-to-derivative mappings in `models/sts_retrain_20261018/provenance.json`. They must not be substituted for originals during binary verification.

No training images, participant records, private manifests, input inventory, logs, credentials or resumable binaries are included. Runtime and protocol records contain only reviewed scientific provenance. This local packaging operation does not publish remotely. Historical benchmark results are unchanged; this evidence contains no new held-out inference.
