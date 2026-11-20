# One Technique per Stage

This research artifact accompanies the IEEE paper on weighting the APT attack tree for detection. It provides defensive analysis code and selected aggregate evidence for five detection studies, technique selection, and cross-stage context. The artifact supports inspection and partial verification. Full reproduction has open requirements in [RESULTS.md](RESULTS.md).

## Start here

1. Read [DATA.md](DATA.md) before acquiring anything. Keep inputs and intermediate outputs local.
2. Use an isolated Python environment. Python 3.10 or newer is required by the source syntax. Confirm package resolution before treating the environment as reproducible.
3. Follow [TEST_PLAN.md](TEST_PLAN.md) for proposed local checks once execution is approved.
4. Use [MANIFEST.md](MANIFEST.md) for the exact release contents.

For a later approved setup, create a virtual environment with `python3 -m venv .venv`. Install the provisional dependencies with `.venv/bin/python -m pip install -r requirements.txt` once package installation is authorized. No packages are needed to inspect the Markdown or JSON.

## Environment

The working dependency specification is [requirements.txt](requirements.txt). Recorded T1190 metadata reports scikit-learn 1.9.1 and NumPy 2.5.3 for both saved runs. Those version strings are retained as historical evidence. Package availability, compatibility, Python version, SciPy, pandas, and python-evtx versions for the original run remain unverified. The file-corpus run records OpenSSL 3.0.2. The T1190 NumPy scorer is a stand-in and its results must be labeled separately. No binary environments or dependencies are bundled.

## Evidence

The JSON files under `evidence/` contain numeric excerpts of saved results. The detector scripts expect their original local input and output layouts. The excerpts support checking claims and should not be fed back as detector inputs. The adapted confidence script reads those excerpts and uses the overwrite operating point for A4. The existing draft license is provisional. See [RESULTS.md](RESULTS.md) for study populations, archived result conventions, and reproduction limits.
