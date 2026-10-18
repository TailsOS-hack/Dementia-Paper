# Mendeley xx9zzz6t54 v1 — acquired and audited

Source: [sufian, abu (2025), Advancing Alzheimer’s Disease Detection in Clinical Settings: MRI Image Data, version 1](https://doi.org/10.17632/xx9zzz6t54.1), listed CC BY 4.0.

Downloaded on October 18, 2026. The archive contains 6,400 JPEGs: 5,121 train and 1,279 test. All actual images are 176 × 208 pixels. The archive contains no CSVs or participant/demographic records, despite the landing-page description of 256 × 256 images and label metadata. Source diagnosis/origin claims were not independently verified.

**Every one of the 6,400 images is an exact RGB pixel match to an image already in the existing legacy collection.** The match hashes include image dimensions and decoded RGB bytes. Both supplied splits overlap existing data. This is another distribution of already-present image content, not 6,400 additional independent cases. The download is not merged into training, and the supplied test split is not treated as external validation.

`audit.json` records archive integrity and counts. `image_inventory.csv` records each image's hash, supplied split/class, and existing-data overlap. The source is integrated as a dataset alias and a frozen-model reused-image diagnostic. Current predictions, all ten stage-model statistics, and regenerated figures are in `results/current/mendeley/`. There are no clinical patient-level claims from this dataset.

Raw files remain local outside the public repository. Download URL observed on the provider page:
https://data.mendeley.com/public-files/datasets/xx9zzz6t54/files/7358251b-5411-478a-813c-2d45746f5a6b/file_downloaded
