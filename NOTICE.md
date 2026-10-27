# Attribution and licensing

Original replay code and documentation are MIT licensed. This grant does not replace the licenses of source datasets, metadata or third-party libraries.

Source-derived numerical data, frozen predictions, cached features and selected trained-head examples in `06_Portable_Replay/` and `05_Review_Only/source_data/` are distributed under Creative Commons Attribution 4.0 International (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/ . These experimental public data contain no new patient data. Original source metadata snapshots and citations are retained under `06_Portable_Replay/source-attribution/`.

- Chicken source: Shuyuan Yang, My Le, Kyle Golobish, Juan Carlos Beaver and Zonghe Chua; v1; source publication date 24 July 2024; Zenodo record https://zenodo.org/records/21041361 . Original article DOI: https://doi.org/10.1142/S2424905X24400087 .
- SinglePSM source: Zonghe Chua, Anthony M. Jarc and Allison M. Okamura; v1; source publication date 18 October 2021; Zenodo record https://zenodo.org/records/21038878 . Original article DOI: https://doi.org/10.1109/ICRA48506.2021.9560945 .

Both source metadata records declared CC BY 4.0 when checked on 27 October 2026. Adaptations: recorded sensor-norm targets and selected bag timestamps were exported from existing immutable analysis caches; frozen predictions were added. Cached descriptors and trained-head parameters were selected by fixed row identity for the inference examples. Data are losslessly text-encoded or stored as existing NumPy/PyTorch arrays. No source raw video, force components or pretrained backbone weights are redistributed. Source owners do not endorse this analysis.

NumPy and PyTorch are optional external dependencies, not vendored software; their own licenses apply. The GRU architecture citations are in `06_Portable_Replay/source-attribution/gru-primary-bib.txt`.

AI assistance: OpenAI Codex assisted code checking, numerical organization and documentation. Human authors remain responsible for the analysis and interpretations.
