"""
Student Wellbeing & Success — integrated analysis across three datasets.

  D1  university mental-health survey (prevalence + treatment gap, depression drivers)
  D2  student habits vs academic performance (what actually moves exam scores)
  D3  socioeconomic outcomes by education level (long-run payoff)

Exports cleaned data, figures, and dashboard/data.json with the headline metrics.
Run:  python src/analysis.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
DASH = ROOT / "dashboard"
for d in (PROC, FIG, DASH):
    d.mkdir(parents=True, exist_ok=True)

RS = 42
PALETTE = ["#E0533D", "#2E7C84", "#F0A35E", "#1C3A5E", "#8FB996"]
YESNO = {"Yes": 1, "No": 0}


def clean_d1():
    df = pd.read_csv(RAW / "d1_mental_health_survey.csv")
    df.columns = ["timestamp", "gender", "age", "course", "year", "cgpa",
                  "marital", "depression", "anxiety", "panic", "treatment"]
    df = df.drop(columns=["timestamp"])
    df["age"] = df["age"].fillna(df["age"].median())
    for c in ["depression", "anxiety", "panic", "treatment"]:
        df[c] = df[c].map(YESNO)
    df["marital_bin"] = (df["marital"].str.strip().str.lower() == "yes").astype(int)
    df["cgpa"] = df["cgpa"].str.strip()
    df.to_csv(PROC / "d1_clean.csv", index=False)
    return df


def clean_d2():
    df = pd.read_csv(RAW / "d2_habits_performance.csv")
    df.to_csv(PROC / "d2_clean.csv", index=False)
    return df


def clean_d3():
    df = pd.read_csv(RAW / "d3_socioeconomic.csv")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.to_csv(PROC / "d3_clean.csv", index=False)
    return df


def models_d1(df: pd.DataFrame) -> dict:
    """Predict depression from demographics + comorbid symptoms."""
    feats = pd.DataFrame({
        "age": df["age"],
        "anxiety": df["anxiety"],
        "panic": df["panic"],
        "marital": df["marital_bin"],
        "gender": LabelEncoder().fit_transform(df["gender"].astype(str)),
        "cgpa": LabelEncoder().fit_transform(df["cgpa"].astype(str)),
    })
    y = df["depression"]
    Xtr, Xte, ytr, yte = train_test_split(feats, y, test_size=0.33, random_state=RS, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=RS),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    acc = {}
    for name, m in models.items():
        m.fit(Xtr_s, ytr)
        acc[name] = round(float(accuracy_score(yte, m.predict(Xte_s))) * 100, 1)
    # logistic coefficients as a rough driver ranking
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(feats), y)
    drivers = sorted(
        [{"feature": f, "weight": round(float(w), 3)} for f, w in zip(feats.columns, lr.coef_[0])],
        key=lambda d: abs(d["weight"]), reverse=True,
    )
    return {"accuracy": acc, "drivers": drivers}


def main():
    d1, d2, d3 = clean_d1(), clean_d2(), clean_d3()

    # ---------- D1 prevalence + treatment gap ----------
    n1 = len(d1)
    prev = {
        "depression": round(d1["depression"].mean() * 100, 1),
        "anxiety": round(d1["anxiety"].mean() * 100, 1),
        "panic": round(d1["panic"].mean() * 100, 1),
    }
    any_distress = round(((d1[["depression", "anxiety", "panic"]].sum(axis=1) > 0)).mean() * 100, 1)
    treatment = round(d1["treatment"].mean() * 100, 1)
    md = models_d1(d1)

    # ---------- D2 correlations with exam score ----------
    num = d2.select_dtypes("number")
    corr = num.corr()["exam_score"].drop("exam_score").sort_values(key=abs, ascending=False)
    corr_top = {k: round(float(v), 3) for k, v in corr.items()}
    r_study = round(float(num["study_hours_per_day"].corr(num["exam_score"])), 3)

    # ---------- D3 income by education ----------
    inc = d3.groupby("education")["income"].mean().round(0).sort_values(ascending=False)
    income_by_edu = {k: int(v) for k, v in inc.items()}

    # ---------- figures ----------
    # study vs exam scatter
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d2["study_hours_per_day"], d2["exam_score"], s=8, alpha=.35, color=PALETTE[0])
    z = np.polyfit(d2["study_hours_per_day"], d2["exam_score"], 1)
    xs = np.linspace(d2["study_hours_per_day"].min(), d2["study_hours_per_day"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color=PALETTE[3], lw=2)
    ax.set_xlabel("Study hours / day"); ax.set_ylabel("Exam score")
    ax.set_title(f"Study time vs exam score (r = {r_study})")
    fig.tight_layout(); fig.savefig(FIG / "study_vs_exam.png", dpi=130); plt.close(fig)

    # prevalence vs treatment
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ["Depression", "Anxiety", "Panic", "Any distress", "Sought\ntreatment"]
    vals = [prev["depression"], prev["anxiety"], prev["panic"], any_distress, treatment]
    cols = [PALETTE[0]]*4 + [PALETTE[1]]
    ax.bar(bars, vals, color=cols)
    ax.set_ylabel("% of students"); ax.set_title("Prevalence vs treatment-seeking")
    fig.tight_layout(); fig.savefig(FIG / "prevalence_gap.png", dpi=130); plt.close(fig)

    # income by education
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ie = inc.sort_values()
    ax.barh(ie.index, ie.values, color=PALETTE[2])
    ax.set_xlabel("Mean income"); ax.set_title("Income by education level")
    fig.tight_layout(); fig.savefig(FIG / "income_by_education.png", dpi=130); plt.close(fig)

    payload = {
        "meta": {"n_d1": int(n1), "n_d2": int(len(d2)), "n_d3": int(len(d3))},
        "prevalence": prev,
        "any_distress": any_distress,
        "treatment": treatment,
        "model_accuracy": md["accuracy"],
        "depression_drivers": md["drivers"],
        "r_study_exam": r_study,
        "exam_correlations": corr_top,
        "income_by_education": income_by_edu,
    }
    (DASH / "data.json").write_text(json.dumps(payload, indent=2))

    print("D1 n=%d  prevalence=%s  any_distress=%.1f%%  treatment=%.1f%%" % (n1, prev, any_distress, treatment))
    print("model accuracy:", md["accuracy"])
    print("r(study,exam)=", r_study, " top exam corr:", list(corr_top.items())[:4])
    print("income by edu:", income_by_edu)


if __name__ == "__main__":
    main()
