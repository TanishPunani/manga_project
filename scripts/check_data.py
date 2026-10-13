import argparse, json
from pathlib import Path

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--sample", required=True)
    a = p.parse_args()
    seqs = json.loads(Path(a.sequences).read_text(encoding="utf-8"))
    labels = read_jsonl(a.labels)
    sample = read_jsonl(a.sample)
    print(f"Manifest entries: {len(seqs)}")
    print(f"Development labels: {len(labels)}")
    print(f"Test template rows: {len(sample)}")
    print("Manifest split counts:", {s: sum(x.get("split") == s for x in seqs) for s in ("development", "test")})
    label_ids = {x["sequence_id"] for x in labels}
    dev_ids = {x["sequence_id"] for x in seqs if x.get("split") == "development"}
    test_ids = {x["sequence_id"] for x in seqs if x.get("split") == "test"}
    sample_ids = {x["sequence_id"] for x in sample}
    print("Labels missing from development manifest:", sorted(label_ids - dev_ids))
    print("Development sequences missing labels:", sorted(dev_ids - label_ids))
    print("Test IDs missing from sample template:", sorted(test_ids - sample_ids))
    print("Sample template IDs not marked test:", sorted(sample_ids - test_ids))
    assert len(labels) == 80 and len(sample) == 15
    assert all(len(x["pages"]) == 3 for x in labels + sample)
    assert label_ids == dev_ids
    assert sample_ids == test_ids
    print("Basic structural checks passed.")

if __name__ == "__main__":
    main()
