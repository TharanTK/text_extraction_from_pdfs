import os
import json
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

# Load multilingual model
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Load sentences from JSONL
def load_jsonl_sentences(filepath):
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("paragraph_content") or obj.get("paragraph") or obj.get("text", "")
                text = text.strip()

                if text:
                    sentences.append(text)
            except json.JSONDecodeError as e:
                print(f"⚠️ Skipping invalid line in {filepath}: {e}")
    return sentences

# Encode with prefix
def encode_with_prefix(sentences):
    return model.encode([f"query: {s}" for s in sentences], convert_to_tensor=True)

# Directory with input files
input_dir = "cleaned_jsonl/new"

# Group files by base name
file_groups = defaultdict(dict)

for filename in os.listdir(input_dir):
    if not filename.endswith(".jsonl"):
        continue
    if "-" not in filename:
        print(f"⚠️ Skipping file (missing '-'): {filename}")
        continue
    try:
        base, lang_ext = filename.rsplit("-", 1)
        lang = lang_ext.replace(".jsonl", "")
        if lang in {"e", "t", "s"}:
            file_groups[base][lang] = os.path.join(input_dir, filename)
        else:
            print(f"⚠️ Skipping file (invalid language code): {filename}")
    except Exception as e:
        print(f"⚠️ Error processing file {filename}: {e}")

# Process each file group
for base_name, paths in file_groups.items():
    if not all(lang in paths for lang in ("e", "t", "s")):
        print(f"⚠️ Skipping {base_name}: missing one or more language files")
        continue

    print(f"\n📄 Processing group: {base_name}")

    english_sentences = load_jsonl_sentences(paths["e"])
    tamil_sentences = load_jsonl_sentences(paths["t"])
    sinhala_sentences = load_jsonl_sentences(paths["s"])

    print("🔄 Encoding sentences...")
    eng_embeddings = encode_with_prefix(english_sentences)
    tam_embeddings = encode_with_prefix(tamil_sentences)
    sin_embeddings = encode_with_prefix(sinhala_sentences)

    aligned_triplets = []
    used_tam = set()
    used_sin = set()

    print("🔍 Mapping sentences with highest similarity...")

    for i, eng_emb in enumerate(eng_embeddings):
        eng_text = english_sentences[i]

        # English–Tamil similarity
        sim_et = util.pytorch_cos_sim(eng_emb, tam_embeddings)[0]
        for idx in used_tam:
            sim_et[idx] = -1
        best_et_idx = torch.argmax(sim_et).item()
        best_et_score = sim_et[best_et_idx].item()
        tam_text = tamil_sentences[best_et_idx]
        tam_emb = tam_embeddings[best_et_idx]

        # Sinhala similarity (average of English-Sinhala and Tamil-Sinhala)
        sim_es = util.pytorch_cos_sim(eng_emb, sin_embeddings)[0]
        sim_ts = util.pytorch_cos_sim(tam_emb, sin_embeddings)[0]
        combined_sim = (sim_es + sim_ts) / 2.0
        for idx in used_sin:
            combined_sim[idx] = -1
        best_es_idx = torch.argmax(combined_sim).item()
        best_es_score = sim_es[best_es_idx].item()
        best_ts_score = sim_ts[best_es_idx].item()
        sin_text = sinhala_sentences[best_es_idx]

        aligned_triplets.append({
            "english": eng_text,
            "tamil": tam_text,
            "sinhala": sin_text,
            "similarity": {
                "eng_tam": round(best_et_score, 4),
                "eng_sin": round(best_es_score, 4),
                "tam_sin": round(best_ts_score, 4),
                "combined_avg": round((best_et_score + best_es_score + best_ts_score) / 3, 4)
            }
        })

        used_tam.add(best_et_idx)
        used_sin.add(best_es_idx)

    # Save aligned triplets
    output_path = f"aligned_triplets_{base_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aligned_triplets, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(aligned_triplets)} aligned triplets to '{output_path}'")
