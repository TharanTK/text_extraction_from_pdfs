import json
from laserembeddings import Laser
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def extract_lines_from_jsonl(jsonl_path, key='text'):
    lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if key in data and data[key].strip():
                lines.append(data[key].strip())
    return lines

def align_sentences(base_lines, other_lines, laser, base_lang='en', other_lang='xx', threshold=0.75):
    base_embeddings = laser.embed_sentences(base_lines, lang=base_lang)
    other_embeddings = laser.embed_sentences(other_lines, lang=other_lang)

    similarity_matrix = cosine_similarity(base_embeddings, other_embeddings)
    alignment = []

    for i, row in enumerate(similarity_matrix):
        best_match_idx = np.argmax(row)
        best_score = row[best_match_idx]
        if best_score >= threshold:
            alignment.append((base_lines[i], other_lines[best_match_idx], best_score))
        else:
            alignment.append((base_lines[i], None, best_score))

    return alignment

def build_multilang_dataset_from_jsonl(eng_path, tam_path, sin_path, key='text'):
    laser = Laser()

    eng_lines = extract_lines_from_jsonl(eng_path, key)
    tam_lines = extract_lines_from_jsonl(tam_path, key)
    sin_lines = extract_lines_from_jsonl(sin_path, key)

    # Align Tamil with English
    tam_alignment = align_sentences(eng_lines, tam_lines, laser, base_lang='en', other_lang='ta')
    aligned_eng_tam = [(eng, tam) for eng, tam, score in tam_alignment if tam]

    # Align Sinhala with English
    sin_alignment = align_sentences(eng_lines, sin_lines, laser, base_lang='en', other_lang='si')
    aligned_eng_sin = [(eng, sin) for eng, sin, score in sin_alignment if sin]

    # Merge based on English sentence
    dataset = []
    for eng, tam in aligned_eng_tam:
        sin = next((s for e, s in aligned_eng_sin if e == eng), None)
        if sin:
            dataset.append((eng, tam, sin))

    df = pd.DataFrame(dataset, columns=["English", "Tamil", "Sinhala"])
    return df

# Example usage
eng_jsonl = 'output/appropriation_en.jsonl'
tam_jsonl = 'output/appropriation_ta.jsonl'
sin_jsonl = 'output/appropriation_si.jsonl'

df = build_multilang_dataset_from_jsonl(eng_jsonl, tam_jsonl, sin_jsonl)
df.to_csv('aligned_sentences_appropriation_jsonl.csv', index=False)
print(df.head())
