ATT&CK Enterprise version: 19.2 (modified 2026-08-05)
Live groups: 176; groups with at least one mapped technique (N): 173

| Technique | Groups (direct) | Groups (+campaigns) | % of N |
|---|---|---|---|
| T1593 | 11 | 13 | 7.5% |
| T1190 | 46 | 47 | 27.2% |
| T1021 | 62 | 63 | 36.4% |
| T1486 | 19 | 19 | 11.0% |
| T1070 | 49 | 50 | 28.9% |
| T1055 (excluded by the observability condition) | 30 | 34 | 19.7% |
| T1685 (log clearing, moved out of T1070 in v19) | 44 | 46 | 26.6% |

T1070 including the v19 log-clearing sub-techniques T1685.005/.006: 54 groups

Co-attestation of consecutive stages (expected under independence = n1*n2/N):

| Pair | Observed | Expected | Lift |
|---|---|---|---|
| T1593 + T1190 | 6 | 3.5 | 1.70 |
| T1190 + T1021 | 36 | 17.1 | 2.10 |
| T1021 + T1486 | 14 | 6.9 | 2.02 |
| T1486 + T1070 | 9 | 5.5 | 1.64 |

Attested for all five: APT41, Sandworm Team
