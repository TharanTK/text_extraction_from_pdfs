from sentence_transformers import SentenceTransformer, util

# Load the multilingual model
model = SentenceTransformer("intfloat/multilingual-e5-base")

english = "The lion roared loudly in the jungle."
tamil = "நான் இப்போது பாஸ்தாவை சமைக்கிறேன்."  # I'm cooking pasta now.
sinhala = "අපි හෙට පසුගිය වසරේ වාර්තාවක් ඉදිරිපත් කරනවා."  # We are presenting last year's report tomorrow.

# Add E5-specific prefix
inputs = [f"query: {s}" for s in [english, tamil, sinhala]]
embeddings = model.encode(inputs, convert_to_tensor=True)

# Calculate cosine similarity matrix
sim_matrix = util.pytorch_cos_sim(embeddings, embeddings)

# Extract scores
sim_et = float(sim_matrix[0][1])  # English–Tamil
sim_es = float(sim_matrix[0][2])  # English–Sinhala
sim_ts = float(sim_matrix[1][2])  # Tamil–Sinhala

# Print results
print("🔎 Similarity Scores (Expected to be Low for Unrelated Sentences):")
print(f"English–Tamil:   {sim_et:.4f}")
print(f"English–Sinhala: {sim_es:.4f}")
print(f"Tamil–Sinhala:   {sim_ts:.4f}")
