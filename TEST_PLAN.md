# Proposed local validation after approval

These commands were inspected only. Run them in a disposable working copy with approved dependencies. The test outputs can be discarded after review.

## Small offline checks

From the candidate root, use the selected Python environment.

```sh
python -B tools/confidence_intervals.py
python -B common/anchors.py
python -B common/bridge.py
python -B T1190_aggregation/scripts/numpy_scorer.py
python -B T1486_sessions/scripts/04_chisquare_corpus.py --self-test
```

Compare the confidence rows to Table II. The NumPy scorer self-test only confirms the stand-in math. The file-corpus self-test processes small synthetic fixtures and opens a locally generated ZIP. It does not invoke OpenSSL. Real file-corpus runs use OpenSSL. Next, validate the T1070 detector against a tiny synthetic pair of forwarded/local record lists and the T1593 grader against synthetic offline events. Prepare and review those fixtures first.

## Dataset-backed checks after input and environment approval

```sh
python -B selection/attestation.py /path/to/approved/enterprise-attack-v19.2.json
python -B T1021_anchor/scripts/01_make_auth_data.py --help
python -B T1021_anchor/scripts/02_run_experiment.py --help
python -B T1021_anchor/scripts/04_pool_windows.py --tags lanl_d8_b1000 lanl_d26_b1000
python -B T1486_sessions/scripts/05_overwrite_detector.py --help
```

The pooling command requires the two full local detector summaries. The public evidence excerpts omit fields required by the pooler. Check the two LANL windows and their individual budgets before pooling. Confirm the T1190 corpus manifest and group split before computing either operating point. Approve the RanSMAP input size and version before extracting features and verifying the held-out application, hardware and mixed-run results. Historical live and chain runs remain evidence-only until suitable inputs and a reviewable offline protocol are approved.
