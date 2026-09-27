# Business Entity Resolution
 
Matches every Source 1 business entity to its duplicates in the Source 2 and Source 3
reference pools, optimising macro-averaged F0.5.
 
## Layout
 
```
code/business_entity_resolution/
├── requirements.txt
├── README.md
├── src/
│   ├── utils.py        normalization, IO, record lookups
│   ├── blocking.py     IDF-weighted sparse token index -> top-k candidates
│   ├── features.py     pairwise string/token/number similarity features
│   ├── model.py        LightGBM matcher + F0.5-optimal decision rule
│   ├── evaluation.py   macro F0.5 metric (parity-tested against evaluator.py)
│   └── pipeline.py     train / evaluate / predict CLI
└── tests/              unit tests (pytest)
utils/
├── validate_submission.py     submission schema + content checks
└── make_synthetic_dataset.py  small schema-compatible dataset for smoke runs
```
 
## Approach
 
1. **Normalization** — NFKC, lowercase, punctuation stripped, whitespace collapsed; a
   `name_core` variant drops legal suffixes (`inc`, `llc`, `pvt`, …) and digit runs are
   extracted from the address (street numbers, unit numbers, postcodes).
2. **Blocking** — every reference record becomes a bag of blocking tokens: name tokens,
   address tokens, digit runs, the sorted name-token signature, and name-token×number
   compounds. Tokens appearing in more than `--max-doc-freq` of the bucket are dropped
   (they are not discriminative and would otherwise dominate the candidate lists, which
   is what capped the notebook blockers at ~50% recall with ~1,700 candidates/record).
   Candidates are the reference records sharing a surviving token, ranked by TF-IDF
   cosine similarity and truncated to `--top-k`. Buckets are per country. Tokens are
   hashed into 2^22 slots instead of a vocabulary dict, and both the index build and the
   queries are chunked, so scoring 1.7M records against 10M references fits in ~7 GB.
3. **Features** — 24 pairwise features: rapidfuzz ratios (plain / token sort / token set /
   partial / Jaro-Winkler) on name and address, token Jaccard and containment, address
   number overlap, first-number match, country agreement, length ratios, the blocking
   score and its rank/ratio to the record's best candidate.
4. **Matcher** — LightGBM binary classifier over the blocked pairs, trained with the
   ground truth as labels (the blocking output supplies the hard negatives).
5. **Decision rule** — F0.5 rewards precision 4:1 over recall, so the predicted set is not
   a plain threshold: a candidate is kept when its probability clears `threshold` **and**
   lies within `margin` of the record's best candidate, capped at `max_k`; optionally the
   single best candidate is kept when it clears `best_threshold`. The four parameters are
   grid-searched on the frozen validation split to maximise macro F0.5.
 
## Running
 
```bash
pip install -r code/business_entity_resolution/requirements.txt
 
# full run: train -> evaluate -> predict -> validate
bash run_pipeline.sh
 
# or step by step, from code/business_entity_resolution/
python -m src.pipeline train --train-dir ../../dataset/train \
    --val-ids ../../validation_artifacts/val_s1_ids.txt --model-dir ../../artifacts
python -m src.pipeline evaluate --train-dir ../../dataset/train \
    --val-ids ../../validation_artifacts/val_s1_ids.txt --model-dir ../../artifacts \
    --output-dir ../../output
python -m src.pipeline predict --test-dir ../../dataset/test \
    --model-dir ../../artifacts --output-dir ../../output
 
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
pytest code/business_entity_resolution/tests
```
 
Useful knobs: `--top-k` (candidates per record, default 30), `--max-doc-freq` /
`--max-doc-count` (blocking token cut-offs, default 0.002 and 3,000), `--train-sample` /
`--tune-sample` / `--eval-sample` (how many Source 1 records to use — full 2.2M training
is unnecessary and slow), `--chunk-size` (Source 1 records per streamed batch, memory vs.
speed), `--no-country-bucket`.
 
## Outputs
 
| file | columns |
| --- | --- |
| `output/candidate_pairs.tsv` | `source1_entity_id`, `candidate_entity_ids` (comma-separated) |
| `output/matching_results.tsv` | `source1_entity_id`, `matched_entity_ids` (comma-separated) |
| `artifacts/metrics.json` | training stats, feature importance, validation macro F0.5 |
| `output/validation_report.json` | macro F0.5, candidate recall, TP/FP/FN diagnostics |
 
Both output files carry exactly one row per test Source 1 entity, in the order of
`test_source1.tsv`.
 
## Notes on scale
 
The reference pool is ~10.3M records and Source 1 ~2.2M (1.7M at test time). Sources are
read and normalized in chunks straight into Arrow-backed string columns, so the raw
object columns of a 5M-row file never exist all at once. Memory is then dominated by the
per-bucket hashed index (~86M postings for the largest country); `--chunk-size` bounds
each similarity product, and `predict` streams candidates → features → scores → output
rows batch by batch instead of materialising 1.7M × k pairs. Measured peak RSS for the
full test run is under 7 GB.
 
Training on a sample of 30k Source 1 records already yields 1.5M labelled pairs and is
enough for the matcher to converge; the tuned run scores macro F0.5 0.8865 on 30k frozen
validation records.