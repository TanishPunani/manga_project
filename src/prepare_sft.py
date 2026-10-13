"""Create a sequence-level instruction-tuning manifest from development labels.

This prepares records only; it does not train a model and does not touch test labels.
"""
import argparse, json
from pathlib import Path

PROMPT = (
    "Read these three consecutive English manga pages in intended reading order. "
    "Transcribe dialogue, internal thoughts, narration, vocalisations, and punctuation-only "
    "speech/thought balloons. Exclude visual sound effects, signs, credits, and text inside "
    "documents/interfaces. Keep speaker labels consistent across the three pages. Use the exact "
    "speaker label NARRATION only for narrator text. Return ONLY JSON with key pages, a list of "
    "three page lists; each line must have speaker and text."
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--out", default="sft_manifest.jsonl")
    args = ap.parse_args()
    labels = [json.loads(x) for x in Path(args.labels).read_text(encoding="utf-8").splitlines() if x.strip()]
    manifest = json.loads(Path(args.sequences).read_text(encoding="utf-8"))
    by_id = {x["sequence_id"]: x for x in manifest}
    with Path(args.out).open("w", encoding="utf-8") as f:
        for row in labels:
            entry = by_id[row["sequence_id"]]
            if entry.get("split") != "development":
                raise ValueError(f"Unexpected non-development sequence: {row['sequence_id']}")
            out = {
                "sequence_id": row["sequence_id"],
                "images": entry["images"],
                "prompt": PROMPT,
                "target": {"pages": row["pages"]}
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"Wrote {len(labels)} labelled training records to {args.out}")
    print("These records are a preparation artifact, not a trained adapter.")

if __name__ == "__main__":
    main()
