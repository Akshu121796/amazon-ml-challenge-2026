"""Generate a small synthetic dataset with the competition schema.
 
Used to exercise the pipeline end-to-end without the full 12M-row data:
 
    python utils/make_synthetic_dataset.py --out-dir dataset --n-train 4000 --n-test 800
"""
 
from __future__ import annotations
 
import argparse
import random
from pathlib import Path
from typing import List, Tuple
 
import pandas as pd
 
FIRST = ["orelee", "prime", "montebello", "aditya", "holloway", "wilford", "moncada", "greenline",
         "sunrise", "blue harbor", "northgate", "silver oak", "riverbend", "cedar", "lakeview"]
SECOND = ["barbershop", "money", "retail", "seafood", "consultants", "properties", "marketing",
          "logistics", "traders", "foods", "textiles", "motors", "clinic", "studios"]
SUFFIX = ["Inc", "LLC", "Pvt Ltd", "Private Limited", "Co", "Corp", "LLP", ""]
STREETS = ["Westchester Drive", "Ellis Road", "Montebello Avenue", "Elm St", "Gulmohar Colony",
           "Park Lane", "Market Street", "Ring Road", "Hill Avenue", "Station Road"]
CITIES = [("High Point", "NC", "US"), ("Tahlequah", "OK", "US"), ("Phoenix", "AZ", "US"),
          ("New Delhi", "Delhi", "India"), ("Bhopal", "Madhya Pradesh", "India"),
          ("Morganton", "NC", "US"), ("Pune", "Maharashtra", "India")]
 
 
def _perturb(text: str, rng: random.Random) -> str:
    """Realistic source noise: abbreviation, case, typos, dropped tokens."""
    result = text
    if rng.random() < 0.35:
        result = result.replace("Street", "St").replace("Road", "Rd").replace("Avenue", "Ave")
    if rng.random() < 0.25:
        result = result.upper()
    if rng.random() < 0.25 and len(result) > 6:
        cut = rng.randrange(1, len(result) - 1)
        result = result[:cut] + result[cut + 1 :]
    if rng.random() < 0.2:
        tokens = result.split()
        if len(tokens) > 2:
            tokens.pop(rng.randrange(len(tokens)))
            result = " ".join(tokens)
    return result
 
 
def _make_entity(rng: random.Random) -> Tuple[str, str, str]:
    name = f"{rng.choice(FIRST).title()} {rng.choice(SECOND).title()} {rng.choice(SUFFIX)}".strip()
    city, region, country = rng.choice(CITIES)
    address = f"{rng.randrange(1, 9999)} {rng.choice(STREETS)}, {city}, {region}"
    return name, address, country
 
 
def build_split(n_s1: int, seed: int, id_offset: int) -> Tuple[pd.DataFrame, ...]:
    rng = random.Random(seed)
    s1_rows, s2_rows, s3_rows, gt_rows = [], [], [], []
    next_id = id_offset
 
    for i in range(n_s1):
        name, address, country = _make_entity(rng)
        s1_id = f"S1-{id_offset + i}"
        s1_rows.append((s1_id, name, address, country))
 
        matches: List[str] = []
        for source, rows in (("S2", s2_rows), ("S3", s3_rows)):
            for _ in range(rng.randrange(0, 3)):
                next_id += 1
                entity_id = f"{source}-{next_id}"
                rows.append((entity_id, _perturb(name, rng), _perturb(address, rng), country))
                matches.append(entity_id)
        if not matches:  # every S1 record has at least one match, as in the real ground truth
            next_id += 1
            entity_id = f"S2-{next_id}"
            s2_rows.append((entity_id, _perturb(name, rng), _perturb(address, rng), country))
            matches.append(entity_id)
 
        # distractors that share a blocking key but are not matches
        for source, rows in (("S2", s2_rows), ("S3", s3_rows)):
            if rng.random() < 0.7:
                next_id += 1
                other_name, _, _ = _make_entity(rng)
                rows.append((f"{source}-{next_id}", other_name, _perturb(address, rng), country))
 
        gt_rows.append((s1_id, ",".join(sorted(matches))))
 
    columns = ["entity_id", "business_name", "business_address", "country"]
    return (
        pd.DataFrame(s1_rows, columns=columns),
        pd.DataFrame(s2_rows, columns=columns),
        pd.DataFrame(s3_rows, columns=columns),
        pd.DataFrame(gt_rows, columns=["source1_entity_id", "matched_entity_ids"]),
    )
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--n-train", type=int, default=4000)
    parser.add_argument("--n-test", type=int, default=800)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
 
    train_dir, test_dir = args.out_dir / "train", args.out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
 
    s1, s2, s3, gt = build_split(args.n_train, args.seed, 1_000_000)
    s1.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    s2.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    s3.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    gt.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)
 
    t1, t2, t3, tgt = build_split(args.n_test, args.seed + 1, 5_000_000)
    t1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    t2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    t3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)
    tgt.to_csv(test_dir / "test_ground_truth.tsv", sep="\t", index=False)
 
    print(f"train: {len(s1)} S1 / {len(s2)} S2 / {len(s3)} S3")
    print(f"test:  {len(t1)} S1 / {len(t2)} S2 / {len(t3)} S3")
 
 
if __name__ == "__main__":
    main()