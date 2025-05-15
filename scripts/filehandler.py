import os
import json
import logging
import re
from dotenv import load_dotenv
import google.generativeai as genai
import PyPDF2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_extractor_batch")

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_TRANSLEGAL")
genai.configure(api_key=GEMINI_API_KEY)

def clean_text_content(text):
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"(\.){2,}", ".", text)
        text = re.sub(r"(,){2,}", ",", text)
        text = re.sub(r"\n\s*\n", "\n", text)
    return text

def upload_to_gemini(doc_path):
    try:
        doc_obj = genai.upload_file(doc_path)
        return ("Success", doc_obj)
    except Exception as e:
        logger.error(f"Gemini upload failed for {doc_path}: {e}")
        return ("Error", None)

def get_pdf_page_count(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return len(reader.pages)
    except Exception as e:
        logger.error(f"Failed to get page count for {pdf_path}: {e}")
        return None

def extract_pdf_with_ai(pdf_path, start_page, end_page):
    try:
        logger.info(f"Processing with Gemini AI: {os.path.basename(pdf_path)} [{start_page}-{end_page}]")
        status, document = upload_to_gemini(pdf_path)
        if status != "Success":
            raise Exception("Upload to Gemini failed.")

        prompt = (
            f"The input is an image or scanned PDF page containing text in english or tamil or sinhala.\n"
            f"Extract all text content **only from pages {start_page} to {end_page}**.\n"
            f"Identify and segment the text into distinct paragraphs.\n"
            f"Return the extracted content in JSON Lines (JSONL) format.\n"
            f"Each line MUST be a valid JSON object.\n"
            f"Do NOT include page numbers, headers, or footers.\n"
            f"Only output JSON lines, no explanation or extra content."
        )
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([prompt, document])
        return response.text
    except Exception as e:
        logger.error(f"AI extraction failed for {pdf_path}: {e}")
        return ""

def save_jsonl_string(jsonl_string, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(jsonl_string)
    logger.info(f"Saved extracted content to {output_path}")

def process_pdf_folder(input_folder, output_folder, start_page=1):
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    logger.info(f"Found {len(pdf_files)} PDF files in {input_folder}.")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        base_name = os.path.splitext(pdf_file)[0]
        output_path = os.path.join(output_folder, f"{base_name}.jsonl")

        total_pages = get_pdf_page_count(pdf_path)
        if total_pages is None:
            logger.warning(f"Skipping {pdf_file} due to page count error.")
            continue

        if start_page > total_pages:
            logger.warning(f"Skipping {pdf_file} because start_page ({start_page}) > total pages ({total_pages})")
            continue

        extracted_text = extract_pdf_with_ai(pdf_path, start_page, total_pages)
        if extracted_text:
            save_jsonl_string(extracted_text, output_path)
        else:
            logger.warning(f"No content extracted for {pdf_file}")

if __name__ == "__main__":
    input_folder = "documents/New"
    output_folder = "output/new"
    process_pdf_folder(input_folder, output_folder, start_page=2)
