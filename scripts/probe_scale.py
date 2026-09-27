"""One-off probe: memory/time of loading + normalizing the real reference pool."""
 
from __future__ import annotations
 
import resource
import sys
import time
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "business_entity_resolution"))
 
from src.utils import add_normalized_columns, read_source  # noqa: E402
 
ROOT = Path("/home/ubuntu/work/ber/dataset")
 
 
def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
 
 
def main() -> None:
    start = time.time()
    s2 = read_source(ROOT / "train" / "train_source2.tsv")
    print(f"read s2 {len(s2):,} rows in {time.time() - start:.0f}s, peak rss {rss_gb():.2f} GB")
    print(s2["country"].value_counts().head(10))
    start = time.time()
    s2 = add_normalized_columns(s2)
    print(f"normalized in {time.time() - start:.0f}s, peak rss {rss_gb():.2f} GB")
 
 
if __name__ == "__main__":
    main()