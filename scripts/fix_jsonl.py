import os
from dotenv import load_dotenv
import google.generativeai as genai

# Setup Gemini
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_TRANSLEGAL")
genai.configure(api_key=GEMINI_API_KEY)

# Load a JSONL file
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return lines

# Clean JSONL using Gemini
def clean_jsonl_with_gemini(lines, filename=None):
    input_text = "".join(lines)
    prompt = (
        f"I have a JSONL file that may contain english or tamil or sinhala text that looks like this:\n"
        f"{input_text}\n\n"
        f"Please clean this content by:\n"
        "- Removing all escaped newline characters (\\n) inside the text values.\n"
        "- Ensuring each JSON object is on a new line.\n"
        "- Adding one blank line between each JSON object for readability.\n"
        "- Do not change the original content apart from cleaning the formatting.\n"
        "Return only the cleaned JSONL content."
    )

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text

# Save cleaned content
def save_cleaned_jsonl(output_path, cleaned_text):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text.strip())

# Process all JSONL files in a folder
def clean_jsonl_folder(input_folder, output_folder):
    jsonl_files = [f for f in os.listdir(input_folder) if f.endswith(".jsonl")]
    print(f"Found {len(jsonl_files)} JSONL files to clean.")

    for jsonl_file in jsonl_files:
        input_path = os.path.join(input_folder, jsonl_file)
        output_path = os.path.join(output_folder, jsonl_file)

        print(f"Cleaning {jsonl_file}...")
        try:
            lines = load_jsonl(input_path)
            cleaned_text = clean_jsonl_with_gemini(lines, jsonl_file)
            save_cleaned_jsonl(output_path, cleaned_text)
            print(f"✅ Saved cleaned file to {output_path}")
        except Exception as e:
            print(f"❌ Failed to process {jsonl_file}: {e}")

# Main
if __name__ == "__main__":
    input_folder = "output/new"  # Input JSONL files
    output_folder = "cleaned_jsonl/new"  # Output folder for cleaned files
    clean_jsonl_folder(input_folder, output_folder)
