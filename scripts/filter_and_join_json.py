import os
import json

# Set your folder path here
input_folder = "aligned_triplets"
output_file = "filtered_output.json"

# Threshold
threshold = 0.85

# List to hold all valid entries
filtered_entries = []

# Loop through all files in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # If it's a list of objects in one file, iterate through
                if isinstance(data, list):
                    entries = data
                else:
                    entries = [data]
                for entry in entries:
                    sim = entry.get("similarity", {})
                    if (
                        sim.get("eng_tam", 0) > threshold and
                        sim.get("eng_sin", 0) > threshold and
                        sim.get("tam_sin", 0) > threshold
                    ):
                        filtered_entries.append({
                            "english": entry.get("english", ""),
                            "tamil": entry.get("tamil", ""),
                            "sinhala": entry.get("sinhala", ""),
                        })
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse {filename}")

# Write the filtered entries to a new JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_entries, f, ensure_ascii=False, indent=2)

print(f"Filtered {len(filtered_entries)} entries written to '{output_file}'")
