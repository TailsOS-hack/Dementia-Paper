# Private clinical development transfer

Ready archive: `data/external/clinical_development_20261011.tar.gz` (30,927,689 bytes).

SHA256: `bc5dca72277b299c8a4eff5d7e4ddf35727e0e8bb101836a223f882260b8d531`.

It contains 397 regular TAR members: one private integrity manifest and 396 payloads. Payloads are exactly 288 training PNGs, 99 validation PNGs, and nine explicitly allowed support files. No held-out image, clinical prediction, research checkpoint or prior outcome file is present. The patient manifest and archive are private; do not publish them.

Upload the archive and the standalone `clinical_transfer.py` installer separately. Installer SHA256 is `26d83baf2bc777ba9fd52eb83469b9df682e2aef91a0cb2ba5e45e241dc0fe23`. Verify both hashes at the remote destination before running the installer. Root owns SSH upload and all cloud actions.

```sh
python3 /workspace/clinical_transfer.py install \
  --archive /workspace/clinical_development_20261011.tar.gz \
  --sha256 bc5dca72277b299c8a4eff5d7e4ddf35727e0e8bb101836a223f882260b8d531 \
  --destination /tmp/Neuro-Radiology-Database \
  --evidence /workspace/clinical_input_verification
```

The stdlib-only installer validates archive SHA, regular safe member paths, exact development membership/counts, all per-file SHA hashes, and frozen source pins. It preflights every destination conflict before writing. It creates only missing files; existing files must be byte-identical. It does not replace the original stage `bundle_manifest.json` or `SHA256SUMS`. All installed files are independently re-read and verified, with timestamped evidence outside the repository. A partial or conflicting install is not a verified input package.

After successful installation, run the discarded smoke using the already provisioned training environment from the current Runpod handoff:

```sh
/tmp/kevin-round2-venv/bin/python \
  /tmp/Neuro-Radiology-Database/docs/kevin_paper_20261011/round2_clinical/smoke_clinical_cuda.py \
  --root /tmp/Neuro-Radiology-Database \
  --prepared /tmp/Neuro-Radiology-Database/docs/kevin_paper_20261011/round2_clinical/prepared \
  --evidence /workspace/clinical_smoke_discarded \
  --research-output /workspace/kevin_clinical_results
```

The smoke uses four TRAIN patients only and a separate diagnostic seed. It reconstructs a fresh ImageNet model for each phase, checks a single head-only step and a single final-three-block step, exact patient-logit aggregation, finite nonzero gradients, frozen BatchNorm/other tensors, and parameter updates in the intended blocks. It writes zero checkpoints and proves the research output directory and frozen source files remain unchanged. Its training loss is a smoke diagnostic, not a performance result. Re-running requires a new evidence directory; do not overwrite earlier evidence.

Smoke helper SHA256: `6cef04f11ed10e1cf0c683bf2cb090790dc8af3b9c6ac045a228809856046c6f`.

Only after root confirms the smoke passed and the existing GPU is available:

```sh
/tmp/kevin-round2-venv/bin/python \
  /tmp/Neuro-Radiology-Database/docs/kevin_paper_20261011/round2_clinical/clinical_bag.py run \
  --root /tmp/Neuro-Radiology-Database \
  --prepared /tmp/Neuro-Radiology-Database/docs/kevin_paper_20261011/round2_clinical/prepared \
  --output /workspace/kevin_clinical_results --device cuda --workers 2
```

Local evidence: `verification/development_bundle.json` and `verification/transfer_tests.json`. Twelve installer tests passed, including real private archive installation/readback, byte-identical repeat installation, old bundle metadata preservation, conflict preflight, destination symlink, wrong archive checksum, traversal/absolute names, archive symlink/hardlink, duplicate members, image corruption, and undeclared payload rejection. Standalone isolated system Python verified the archive without any non-stdlib dependencies. The CUDA smoke was prepared, not executed locally. Neither the frozen trainer nor its protocol was changed.
