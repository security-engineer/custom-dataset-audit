---
name: auditing-dataset-shortcuts
description: Use when a text classifier (human-vs-AI detector, authorship, or any binary text benchmark) reports suspiciously high AUROC/accuracy, before trusting or publishing benchmark results, or when asked whether detection numbers reflect real signal vs dataset artifacts.
---

# Auditing Dataset Shortcuts

## Overview
High classifier scores on text benchmarks are usually a MIX of: ① construction artifacts (formatting relics, preprocessing tokens), ② brittle-but-real signals (LLM-isms, length policy), ③ robust signal. Your job is to decompose the score, not just find "a problem". One dead-giveaway byte can carry AUROC 0.99 alone (real case: 100% of one corpus's human texts started with "– " — first-character classification).

## Quick start — bundled CLI (run this first)
A dataset-agnostic battery ships with this skill. Given two JSONL files (one per class, each line a JSON object with a text field):

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/auditing-dataset-shortcuts/scripts/audit_shortcuts.py" \
    --human human.jsonl --ai ai.jsonl --text-field text [--json report.json]
```

It reports separability AUROC (0.5 = no signal, 1.0 = perfect giveaway) for: lexical TF-IDF probe, register/stopword probe, length-only, surface single-features (space-padded punctuation, em-dash, non-ASCII, markdown bullet, missing terminal punctuation, digit ratio), and opener asymmetry — then prints a verdict with flags. Needs `scikit-learn`+`numpy` for the two probes; the surface/length battery runs on pure stdlib and the probes degrade gracefully if those are absent. **The CLI is the fast first pass — the manual checklist below catches what an automated battery cannot (split/leakage logic, prompt-induced absences, signal location).**

## Iron rules
1. **Match the suspect protocol exactly** (same row filters, same train/test split, same folds) before comparing any baseline number to the model's.
2. **Verify the actual split before claiming leakage.** Read the training script. "Human texts repeat across cells" is only a leak if the split lets the same source cross train/test. Overclaiming leakage discredits the audit.
3. **Compare rates, not counts**, when one class is deduplicated and the other repeats (per-text presence %, per-1k-words frequency).
4. Single-feature AUROC for every candidate giveaway; family-combined logistic for each family; report next to the model's number.

## The 7-family checklist (run ALL — partial audits miss the killers)
| Family | What to compute | Real-case finding |
|---|---|---|
| 1. Length | words/chars/sentences-only AUROC | 0.51–0.997 by fold |
| 2. Format markers | leading/trailing bytes, markdown, headers, leaked prompt/preproc tokens (`[Evidence`, `NEWLINE_CHAR`), URLs, HTML entities | "– " prefix = 0.99 alone |
| 3. Unicode | smart quotes/dashes/non-ASCII by class — **check per domain; direction can REVERSE across domains** (normalize globally, not per-class) | humans ASCII-pure in 2 domains, opposite in the 3rd |
| 4. Surface stats | 13-stat battery (avg word len, punctuation rates, stopword ratio…), with and without length features | 0.94 / 0.91 |
| 5. Lexical | log-odds top words per class (LLM-isms vs content?), opener templates (first-trigram one-hot) | "aims" 48% AI vs 0% human |
| 6. Human-side relics | class-specific source-cleaning leftovers (citations, legalese enumeration, attribution verbs) — classify each as natural style vs **prompt-induced absence** (read the generation prompts' ban lists; ablate to test) | subsection refs 0.834 |
| 7. Signal location | TF-IDF on first sentence / last sentence / middle-50% / full — distributed vs ending-artifact | middle≈full ⇒ distributed |

## Then: normalize and re-test (the audit's product)
Build a normalization recipe from findings (strip giveaway prefixes, fold unicode both classes, drop leaked-token rows, unescape HTML, replace URLs), optionally equal-length truncation, then **retrain the suspect model on the normalized data**. Report the three-tier table: raw → normalized → normalized+length-controlled. The surviving number is the defensible one.

## Common mistakes
- Auditing only length/TF-IDF and stopping (misses families 2,3,6,7).
- Claiming leakage without reading the split code (rule 2).
- Normalizing only the class where artifacts were found (introduces a NEW artifact).
- Treating LLM-isms as "artifacts" — they're real but brittle signal; tier ② not ①.
- Comparing dedup'd-class counts to repeated-class counts (rule 3).
