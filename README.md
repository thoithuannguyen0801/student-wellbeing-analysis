# Student Wellbeing & Success Analysis

An integrated analysis across **three datasets** investigating how study habits, mental
health and socioeconomic background shape student performance and long-run outcomes.
Built in Python with a self-contained interactive dashboard and a written report.

## Live outputs

| Output | File |
|--------|------|
| Interactive dashboard | [`dashboard/index.html`](dashboard/index.html) — open in any browser |
| Written report | [`report/index.html`](report/index.html) |

Both are self-contained — double-click to open, or host free on GitHub Pages.

## What's inside

```
src/analysis.py            # end-to-end, reproducible pipeline across all 3 datasets
notebooks/analysis.ipynb   # narrated notebook version
data/raw/                  # 3 source datasets
data/processed/            # cleaned versions of each dataset
figures/                   # exported charts (PNG)
dashboard/index.html       # interactive dashboard (Chart.js)
dashboard/data.json        # summary metrics produced by analysis.py
report/index.html          # written analytical report
```

## Datasets

| # | Dataset | n | Used for |
|---|---------|---|----------|
| D1 | University mental-health survey | 101 | Prevalence, treatment gap, depression drivers |
| D2 | Student habits & performance | 1,000 | What drives exam scores |
| D3 | Socioeconomic outcomes | 2,000 | Income by education level |

## Key findings

- **Study time dominates performance:** r = 0.825 with exam score — far above any other habit.
- **Wellbeing gap:** ~63% of surveyed students reported some distress; only 5.9% sought treatment.
- **Depression drivers are not academic:** marital status and anxiety lead; CGPA is negligible.
- **Education pays:** postgraduate mean income (~$147k) sits ~26% above high-school-only.
- Classifiers reach 76–79% accuracy predicting depression (small sample, indicative).

## Reproduce

```bash
pip install -r requirements.txt
python src/analysis.py     # regenerates data/processed, figures and dashboard/data.json
```
