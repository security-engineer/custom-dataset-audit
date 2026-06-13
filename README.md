# custom-dataset-audit

A Claude Code plugin marketplace for **auditing binary text benchmarks for dataset shortcuts** — before you trust (or publish) a suspiciously high AUROC from a human-vs-AI detector, authorship classifier, or any binary text benchmark.

High classifier scores are usually a *mix* of construction artifacts (formatting relics, leaked tokens), brittle-but-real signals (LLM-isms, length policy), and robust signal. This plugin helps you **decompose** the score instead of trusting it.

## Install (in Claude Code)

```text
/plugin marketplace add security-engineer/custom-dataset-audit
/plugin install dataset-shortcut-audit@custom-dataset-audit
```

Then the **`auditing-dataset-shortcuts`** skill activates whenever a text benchmark reports a suspiciously high score, and the bundled CLI is available.

## What's inside

- **`auditing-dataset-shortcuts` skill** — the 7-family auditing methodology (length, format markers, unicode, surface stats, lexical, human-side relics, signal location) plus iron rules (match the protocol, verify the split before claiming leakage, compare rates not counts) and the normalize-and-retest workflow.
- **`audit_shortcuts.py` CLI** — a dataset-agnostic shortcut battery you can run directly.

## Use the CLI directly

Two JSONL files, one per class, each line a JSON object with a text field:

```bash
python plugins/dataset-shortcut-audit/skills/auditing-dataset-shortcuts/scripts/audit_shortcuts.py \
    --human human.jsonl --ai ai.jsonl --text-field text --json report.json
```

Reports **separability AUROC** (0.5 = no signal, 1.0 = perfect giveaway) for:

1. Lexical TF-IDF probe (word 1–2gram → logistic, CV) + TPR@1%FPR
2. Register / stopword-only probe
3. Length-only
4. Surface single-features: space-padded punctuation, em-dash, non-ASCII ratio, markdown bullet, missing terminal punctuation, digit ratio
5. Opener asymmetry (lead trigram differs between classes)

…then prints a **verdict** flagging ceiling lexical shortcuts, register confounds, length confounds, surface giveaways, and opener asymmetry.

**Dependencies:** `scikit-learn` and `numpy` for the two probes (`pip install scikit-learn numpy`). The surface/length battery runs on pure stdlib; the probes are skipped gracefully if those packages are absent.

> The CLI is a fast first pass. It is *necessary, not sufficient* — pair it with the manual split/leakage and signal-location checks in the skill, which an automated battery cannot do.

## License

MIT
