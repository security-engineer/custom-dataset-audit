#!/usr/bin/env python3
"""audit_shortcuts.py — dataset-agnostic shortcut audit for binary text benchmarks.

Given two JSONL files (one human/class-0, one AI/class-1), each line a JSON object
with a text field, this reports whether a classifier's high AUROC could come from
dataset artifacts (shortcuts) rather than real authorship signal.

Battery (each reported as separability AUROC = max(a, 1-a), 0.5 = no signal, 1.0 = perfect giveaway):
  1. Lexical TF-IDF probe   (word 1-2gram → logistic regression, CV)   [needs scikit-learn]
  2. Register / stopword probe (stopwords-only TF-IDF → logistic)       [needs scikit-learn]
  3. Length-only            (word count, single feature, rank AUROC)
  4. Surface single-features (space-padded punctuation, em-dash, non-ascii, markdown
                              bullet, missing-terminal-punct, digit ratio)
  5. Opener monopoly        (top first-3-words share per class)

Pure-python for everything except the two probes; if scikit-learn is missing, those are
skipped with a note and the surface/length battery still runs.

Usage:
    python audit_shortcuts.py --human human.jsonl --ai ai.jsonl [--text-field text] [--json out.json]
"""

import argparse
import json
import re
import sys
from collections import Counter


# ---------- io ----------
def load_texts(path, field):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            v = obj.get(field)
            if isinstance(v, str) and v.strip():
                out.append(v)
    return out


# ---------- rank-based AUROC (pure python, tie-aware) ----------
def auroc(scores0, scores1):
    """AUROC for label-1 (scores1) over label-0 (scores0) via rank-sum; 0.5=chance."""
    n0, n1 = len(scores0), len(scores1)
    if n0 == 0 or n1 == 0:
        return float("nan")
    paired = [(s, 0) for s in scores0] + [(s, 1) for s in scores1]
    paired.sort(key=lambda x: x[0])
    # average ranks for ties
    ranks = [0.0] * len(paired)
    i = 0
    while i < len(paired):
        j = i
        while j + 1 < len(paired) and paired[j + 1][0] == paired[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(ranks[k] for k in range(len(paired)) if paired[k][1] == 1)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return u1 / (n0 * n1)


def separability(scores0, scores1):
    a = auroc(scores0, scores1)
    if a != a:  # nan
        return a, "n/a"
    return (a if a >= 0.5 else 1 - a), ("ai_higher" if a >= 0.5 else "ai_lower")


# ---------- surface features (per text scalar) ----------
_BULLET = re.compile(r"^\s*[\*\-•·]")
_TERMINAL = ".!?\"'）)】」』”’"


def feat(text):
    n_chars = len(text)
    words = text.split()
    n_words = max(1, len(words))
    nonascii = sum(1 for c in text if ord(c) > 127)
    digits = sum(1 for c in text if c.isdigit())
    return {
        "length_words": float(len(words)),
        "space_padded_punct": (
            text.count(" , ")
            + text.count(" . ")
            + text.count(" ( ")
            + text.count(" ) ")
        )
        / n_words,
        "em_dash": (text.count("—") + text.count(" - ")) / n_words,
        "nonascii_ratio": nonascii / max(1, n_chars),
        "md_bullet": 1.0 if _BULLET.match(text) else 0.0,
        "ends_no_terminal": 0.0
        if (text.rstrip() and text.rstrip()[-1] in _TERMINAL)
        else 1.0,
        "digit_ratio": digits / max(1, n_chars),
    }


def opener_top_share(texts, k=3):
    openers = Counter(" ".join(t.lower().split()[:k]) for t in texts if t.split())
    if not openers:
        return 0.0, ""
    top, cnt = openers.most_common(1)[0]
    return cnt / len(texts), top


# ---------- sklearn probes ----------
def sklearn_probe(human, ai, mode):
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score, roc_curve
    except Exception:
        return None
    X = human + ai
    y = np.array([0] * len(human) + [1] * len(ai))
    if mode == "lexical":
        vec = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True
        )
    else:  # register: stopwords only
        sw = list(ENGLISH_STOP_WORDS)
        vec = TfidfVectorizer(vocabulary=sw)
    try:
        Xt = vec.fit_transform(X)
        if Xt.shape[1] == 0:
            return None
        clf = LogisticRegression(max_iter=1000, C=1.0)
        n = min(5, len(human), len(ai))
        if n < 2:
            return None
        proba = cross_val_predict(clf, Xt, y, cv=n, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba)
        fpr, tpr, _ = roc_curve(y, proba)
        tpr_at_1 = max([t for f, t in zip(fpr, tpr) if f <= 0.01], default=0.0)
        return {
            "auroc": round(float(auc), 4),
            "tpr_at_1pct_fpr": round(float(tpr_at_1), 4),
        }
    except Exception as e:
        return {"error": str(e)}


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Dataset shortcut audit for binary text benchmarks."
    )
    ap.add_argument("--human", required=True, help="JSONL of class-0 (human) texts")
    ap.add_argument("--ai", required=True, help="JSONL of class-1 (AI) texts")
    ap.add_argument(
        "--text-field",
        default="text",
        help="JSON field holding the text (default: text)",
    )
    ap.add_argument(
        "--json", default=None, help="optional path to write the report as JSON"
    )
    args = ap.parse_args()

    human = load_texts(args.human, args.text_field)
    ai = load_texts(args.ai, args.text_field)
    if not human or not ai:
        sys.exit(
            f"ERROR: empty input (human={len(human)}, ai={len(ai)}). Check --text-field '{args.text_field}'."
        )

    report = {"n_human": len(human), "n_ai": len(ai), "flags": []}

    # surface single-features
    fh = [feat(t) for t in human]
    fa = [feat(t) for t in ai]
    surface = {}
    for key in fh[0]:
        s, direction = separability([d[key] for d in fh], [d[key] for d in fa])
        surface[key] = {"separability_auroc": round(s, 4), "direction": direction}
    report["surface"] = surface
    report["length_only_auroc"] = surface["length_words"]["separability_auroc"]

    # opener monopoly
    sh, th = opener_top_share(human)
    sa, ta = opener_top_share(ai)
    report["opener_monopoly"] = {
        "human": {"top_share": round(sh, 4), "opener": th},
        "ai": {"top_share": round(sa, 4), "opener": ta},
    }

    # probes
    report["lexical_tfidf"] = sklearn_probe(human, ai, "lexical")
    report["register_stopword"] = sklearn_probe(human, ai, "register")

    # ---- verdict ----
    flags = report["flags"]
    lex = report["lexical_tfidf"]
    if isinstance(lex, dict) and "auroc" in lex:
        if lex["auroc"] >= 0.98:
            flags.append(
                f"CEILING lexical shortcut (TF-IDF AUROC {lex['auroc']}) — vocabulary alone separates classes; real-signal claims unsafe."
            )
        elif lex["auroc"] >= 0.90:
            flags.append(f"High lexical separability (TF-IDF AUROC {lex['auroc']}).")
    reg = report["register_stopword"]
    if isinstance(reg, dict) and reg.get("auroc", 0) >= 0.90:
        flags.append(
            f"Register shortcut (stopword-only AUROC {reg['auroc']}) — classes differ in style/register, not necessarily authorship."
        )
    if report["length_only_auroc"] >= 0.80:
        flags.append(
            f"Length confound (length-only AUROC {report['length_only_auroc']})."
        )
    for key, v in surface.items():
        if key == "length_words":
            continue
        if v["separability_auroc"] >= 0.80:
            flags.append(
                f"Surface giveaway '{key}' (single-feature AUROC {v['separability_auroc']}, {v['direction']})."
            )
    om = report["opener_monopoly"]
    # Opener is a discriminative shortcut only when ASYMMETRIC: the two classes lead
    # with DIFFERENT openers and at least one dominates. A shared opener at equal rate
    # is a corpus property, not a between-class giveaway, so it must not trip the verdict.
    if (
        om["human"]["opener"] != om["ai"]["opener"]
        and max(om["human"]["top_share"], om["ai"]["top_share"]) >= 0.20
    ):
        flags.append(
            f"Opener asymmetry: human top '{om['human']['opener']}' {om['human']['top_share']:.0%} vs ai top '{om['ai']['opener']}' {om['ai']['top_share']:.0%}."
        )

    report["verdict"] = (
        "SHORTCUTS DETECTED — audit before trusting numbers"
        if flags
        else "No strong shortcut detected by this battery"
    )

    # ---- print ----
    print(f"\n=== Dataset Shortcut Audit ===  human={len(human)}  ai={len(ai)}")
    if isinstance(lex, dict) and "auroc" in lex:
        print(
            f"  [1] Lexical TF-IDF probe AUROC : {lex['auroc']}  (TPR@1%FPR {lex['tpr_at_1pct_fpr']})"
        )
    else:
        print(
            f"  [1] Lexical TF-IDF probe       : skipped ({'scikit-learn not installed' if lex is None else lex})"
        )
    if isinstance(reg, dict) and "auroc" in reg:
        print(f"  [2] Register (stopword) AUROC  : {reg['auroc']}")
    else:
        print(f"  [2] Register (stopword) probe  : skipped")
    print(f"  [3] Length-only AUROC          : {report['length_only_auroc']}")
    print(f"  [4] Surface single-features (separability AUROC):")
    for key, v in surface.items():
        if key == "length_words":
            continue
        print(f"        {key:20s} {v['separability_auroc']:.4f}  ({v['direction']})")
    print(
        f"  [5] Opener monopoly            : human '{th}' {sh:.1%} | ai '{ta}' {sa:.1%}"
    )
    print(f"\n  VERDICT: {report['verdict']}")
    for fl in flags:
        print(f"    ⚠ {fl}")
    if not flags:
        print(
            "    (battery is necessary, not sufficient — pair with leakage/split audit; see SKILL.md)"
        )
    print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  JSON report → {args.json}\n")


if __name__ == "__main__":
    main()
