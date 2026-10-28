# CRCT code release (v1.0.0)

Code for:

**Few-shot composition extrapolation of thermoelectric performance curves via chemistry-conditioned residual curve transport (CRCT)**

This package contains the model code, experiment scripts, tests, and figure-generation scripts. Processed data and result files are in the companion data release.

## Contents

```text
src/                 model, data, evaluation and plotting code
scripts/             experiment and figure scripts
tests/               test suite
requirements.txt     minimal dependencies
requirements-lock.txt  exact versions used for the manuscript
PROJECT_README.md    original project README
README_FIGURES.md    figure inventory
```

## Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements-lock.txt
```

If `lightgbm` fails to install, use the version for your platform from
https://lightgbm.readthedocs.io/ and keep the remaining versions unchanged.

## Data

The processed inputs are in the companion data release. Place them at:

```text
data/processed/starrydata2_curves_long_clean.parquet
data/processed/starrydata2_samples_clean.csv
```

## Run tests

```bash
python -m pytest tests -q
```

## Reproduce figures

Figure scripts are under `figures/` in the development repository and under
`scripts/` where applicable. The manuscript figure files are PDF/SVG vector
exports.

## License

`[choose a license before publishing the repository; MIT is a common choice]`

## Contact

`[corresponding author name and email to be added]`
