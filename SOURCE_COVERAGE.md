# Source coverage and omissions

The source is `TailsOS-hack/Neuro-Radiology-Database`. This standalone repository contains the complete October 9 dementia-only evidence package, the October 11 scientific handoff payload, both publication rounds, additional already-public dementia documentation and audits, dementia training/evaluation/figure code, Round 3 code and tests, thirteen trained dementia-related checkpoints, the recorded initialization weights, and the historical original100 benchmark assets.

The source tree was compared with the live public GitHub tree on October 17, 2026. All thirteen trained checkpoint pointers and the included historical evaluation images/ground-truth/reader files already appear as identical blobs in that public tree. Newly copied material includes repository documentation, previously delivered historical figure panels, source inventories and additional research documentation. No raw newly acquired clinical MRI or private participant records are added.

## Material intentionally kept outside this public repository

- Full raw training-image collection: sources and manifests are included; the large dataset is not duplicated.
- Newly acquired clinical DICOM/NIfTI, participant rows, private clinical predictions and original clinical manifests: aggregate results and reviewed scientific figures are supplied instead. Exact clinical replay requires appropriately obtained originals.
- Private access applications/attestations and personal correspondence: not manuscript evidence for public release.
- Cloud credentials, connection records, billing and operational logs: excluded.
- Tumor data/checkpoints, routing/classification GUI, mixed-disease applications and unrelated multimodal work: outside this paper.

These omissions mean that “complete” refers to the available shareable scientific handoff and code, not unrestricted redistribution of every source dataset. Study documents explicitly record unavailable data, blocked checks and pending future work. No missing real outcomes are replaced by synthetic results.

## Provenance and integrity

Source bytes are retained wherever possible. Handoff entry-document paths were adapted to the standalone layout. Generic attribution helpers were extracted without changing the scientific function bodies. The repository documentation explains the archive; it does not report newly measured results. Scientific source manifests retain original package provenance. `REPOSITORY_MANIFEST.json` is the authoritative inventory for this standalone copy.
