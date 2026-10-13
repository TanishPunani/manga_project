"""Run local open-weight Qwen2.5-VL inference on the three images per sequence."""
import argparse, json, re
from pathlib import Path
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

PROMPT = """Read these three consecutive English manga pages in intended reading order.
Include dialogue, internal thoughts, narration, clear spoken screams/grunts/vocalisations, and
punctuation-only speech/thought balloons. Exclude visual sound effects and translated captions,
tiny unclear reactions, titles/logos/credits/page numbers/ads/watermarks, labels, notes, signs,
clothes/objects, and text inside letters, diaries, phone messages, news pages, or interfaces.
Read in intended visual order (usually right-to-left). Thoughts belong to the thinking character.
Use the exact label NARRATION only for narrator text. Use consistent invented speaker labels across
all three pages. Return ONLY valid JSON in this exact shape:
{"pages":[[{"speaker":"char1","text":"..."}],[],[]]}
The pages array must contain exactly three lists. Do not add explanations."""

def resolve_image(root, rel):
    p = root / rel
    if p.exists():
        return p
    # ZIPs supplied by the user have paths like images/<sequence_id>/01.png.
    candidate = root / Path(rel).parent.name / Path(rel).name
    if candidate.exists():
        return candidate
    candidate = root / "images" / Path(rel).parent.name / Path(rel).name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find image {rel}; tried {p} and common extracted layouts.")

def extract_json(text):
    text = text.strip()
    # Remove common model-added fences, then recover the outermost JSON object.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model response did not contain a JSON object: " + text[:300])
    obj = json.loads(text[start:end+1])
    pages = obj.get("pages")
    if not isinstance(pages, list) or len(pages) != 3:
        raise ValueError("Model output must contain exactly three page lists")
    cleaned = []
    for page in pages:
        if not isinstance(page, list):
            raise ValueError("Each page must be a list")
        lines = []
        for item in page:
            if not isinstance(item, dict) or not isinstance(item.get("speaker"), str) or not isinstance(item.get("text"), str):
                continue
            speaker, txt = item["speaker"].strip(), item["text"].strip()
            if speaker and txt:
                lines.append({"speaker": speaker, "text": txt})
        cleaned.append(lines)
    return cleaned

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--sequences", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=1800)
    ap.add_argument("--limit", type=int, default=0, help="Debug only: process first N sequences")
    args = ap.parse_args()

    seqs = json.loads(Path(args.sequences).read_text(encoding="utf-8"))
    seq_by_id = {x["sequence_id"]: x for x in seqs}
    template = [json.loads(x) for x in Path(args.sample).read_text(encoding="utf-8").splitlines() if x.strip()]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model)
    image_root = Path(args.image_root)

    out_rows = []
    todo = template[:args.limit] if args.limit else template
    for row in todo:
        sid = row["sequence_id"]
        if sid not in seq_by_id:
            raise KeyError(f"Sequence {sid} missing from sequences manifest")
        entry = seq_by_id[sid]
        content = [{"type": "text", "text": PROMPT}]
        for rel in entry["images"]:
            img_path = resolve_image(image_root, rel)
            content.append({"type": "image", "image": Image.open(img_path).convert("RGB")})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            pages = extract_json(response)
        except Exception as e:
            raise RuntimeError(f"Invalid output for {sid}: {e}") from e
        out_rows.append({"sequence_id": sid, "pages": pages})
        print(f"{len(out_rows)}/{len(todo)} completed: {sid} ({sum(map(len, pages))} lines)")
    with Path(args.output).open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out_rows)} sequences to {args.output}")

if __name__ == "__main__":
    main()
