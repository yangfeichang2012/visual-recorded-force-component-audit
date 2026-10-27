# Visual recorded-force component audit — D76 v2.4

Frozen replay code for **Source-dependent gains and tail errors in visual recorded-force prediction for surgical robotics**. Software version: **v2.4.0**. This release is a computational artifact, not a claim of clinical safety or a complete raw-video training pipeline.

## Run from the repository root

Python 3.12 or later; the first three commands need only its standard library. No command installs dependencies, accesses the network, writes results, or fits/tunes a model.

```sh
python -B 06_Portable_Replay/verify_endpoint_replay.py
python -B 05_Review_Only/verify_d76_tables.py
python -B 05_Review_Only/verify_revision_tables.py
```

Optional cached-input inference (requires the tested NumPy/PyTorch environment in `06_Portable_Replay/environment.json`):

```sh
python -B 06_Portable_Replay/verify_frozen_inference.py
```

The optional command checks three fixed rows for each of three cases. It uses CPU, one thread, FP64 heads and the original cached descriptor precision; its absolute tolerance is 1e-12 N. It does not decode video or rerun the image encoder. RBF examples include exact training descriptors; GRU files contain a state dictionary without optimizer state. Never load arbitrary external checkpoints.

## What can be reproduced

- 35,303 unique frozen 5-Hz target rows across 83 included source/view recordings, all 18 original method outputs, and the V1 GRU outputs.
- Per-record MAE, RMSE, P95, signed bias, training-Q90 high-force error and eligible-grid peak shortfall, followed by the original group/record hierarchy and complete gates.
- D204 failed GRU comparison and D205 internal anchor-cap stopping diagnostic, including 49,410 repeated role/cap prediction rows. These are not additional independent observations.
- Three representative cached-feature/head inference examples, not statistical validation.

`05_Review_Only` is a retained legacy folder name for the public numerical tables and verifiers; it contains no peer reviews. The public revision verifier retains the original numerical checks but omits checks of unpublished manuscript wording and the unrelated whole-submission manifest. This change is documented in `PUBLIC-PROVENANCE.json`.

## Interpretation and limits

Fusion versus fixed temporal MAE was 0.8813 → 0.8123 N in chicken V1 (7.83%), 0.8286 → 0.8220 N in V2 (0.80%), and 0.9813 → 0.8221 N in SinglePSM (16.23%). Only SinglePSM passed the complete original gate. The SinglePSM motion branch B had lower MAE than fusion. V1 fusion reduced MAE against B while increasing its P95. Failed GRU and size-diagnostic results remain included.

The sources were previously exposed and retrained separately. They do not establish blind external validation or direct weight transfer. Proxy IDs/acquisition conditions are not verified independent biological specimens. Targets are recorded three-axis sensor norms in source-declared N; calibration, physical zero, drift and hardware synchronization were not established. Five-Hz outputs do not establish fast dynamic response or safe feedback use. No population confidence interval or frame-level significance claim is supplied.

`singlepsm-all46-flow.csv` retains all exclusions: one permanent pilot and two unresolved metadata quarantines. Chicken V2 is retained as negative evidence. Quantiles use linear interpolation; high-force thresholds come from the relevant training role. Empty high-force support remains null, not zero. Eligibility, row identities and original hashes are supplied.

## Scope and provenance

No private author contact files, unpublished manuscript, internal review reports, raw recordings or pretrained backbone weights are included. Historical training launchers with machine-specific paths are not a runnable public training interface and are omitted. This archive supports frozen-output and selected cached-feature replay; **full training reproduction is not claimed**. See `NOTICE.md` for source attribution and separate data licensing, `CITATION.cff` for citation metadata, and `PUBLIC-MANIFEST.json` for file hashes.

Lang Yang, Chuan-Xin Duan and Xixi Gao contributed equally and share first authorship, listed in that order; Xiao Chen is the corresponding manuscript author. Funding: Elite Medical Professionals Project of China-Japan Friendship Hospital (ZRJY2025-QMPY67). This code publication does not certify final approval of a journal submission by all authors. No journal submission has been performed.
