import os
from google.api_core.client_options import ClientOptions
import google
import json
import requests
import math
from PyPDF2 import PdfReader, PdfWriter
import io
import re

from google.cloud import documentai_v1 as documentai

api_json = "paralegal-459016-b0fa7a68be47.json"

project_json_path = "project_credential.json"
with open(project_json_path) as file:
    project_data = json.load(file)


def get_google_documentai(
    pdf_chunk: io.BytesIO
) -> google.cloud.documentai_v1.types.document_processor_service.ProcessResponse:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = api_json

    global project_data

    PROJECT_ID = project_data["project_id"]
    LOCATION = project_data["location"]
    PROCESSOR_ID = project_data["processor_id"]
    MIME_TYPE = "application/pdf"

    docai_client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(
            api_endpoint=f"{LOCATION}-documentai.googleapis.com"
        )
    )

    RESOURCE_NAME = docai_client.processor_path(
        PROJECT_ID, LOCATION, PROCESSOR_ID)

    # Read PDF chunk content
    chunk_content = pdf_chunk.read()

    # with open(pdf_path, "rb") as image:
    #     image_content = image.read()

    raw_document = documentai.RawDocument(
        content=chunk_content, mime_type=MIME_TYPE)

    request = documentai.ProcessRequest(
        name=RESOURCE_NAME, raw_document=raw_document)

    result = docai_client.process_document(request=request)

    document_object = result.document.text

    return document_object

# Function to split the PDF into chunks of a specified size (default is 15 pages)


def get_pdf_into_chunks(pdf_path: str, chunk_size: int = 15) -> list:
    pdf_bucket = []

    # Open the PDF file
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        total_page_num = len(pdf_reader.pages)

        # Loop through the PDF in chunks of chunk_size
        for start_page in range(0, total_page_num, chunk_size):
            end_page = min(start_page + chunk_size, total_page_num)
            print(f"Processing pages {start_page + 1} to {end_page}...")

            pdf_writer = PdfWriter()

            # Write the pages in the range to a new PDF chunk
            for page_num in range(start_page, end_page):
                pdf_writer.add_page(pdf_reader.pages[page_num])

            # Save the chunk in a BytesIO stream
            chunk_stream = io.BytesIO()
            pdf_writer.write(chunk_stream)
            chunk_stream.seek(0)
            pdf_bucket.append(chunk_stream)

    return pdf_bucket

# Main function to process the entire PDF and return the concatenated text


def process_large_pdf(pdf_path: str, chunk_size: int = 15) -> str:
    # Split the PDF into chunks
    pdf_chunks = get_pdf_into_chunks(pdf_path, chunk_size)

    all_text = []

    # Process each chunk and collect the text
    for chunk in pdf_chunks:
        chunk_text = get_google_documentai(chunk)
        all_text.append(chunk_text)

    # Concatenate all the text from all chunks
    return "\n".join(all_text)

# Function to save text to a JSON file


def save_text_to_json(text: str, output_file: str):

    data = {
        "decision_text": text
    }
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

# Function to process all PDFs in a directory and save results


def process_pdfs_in_directory(directory_path: str):
    success_count = 0
    failed_pdfs = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                output_json_path = pdf_path.replace(".pdf", ".json")

                print(f"Processing PDF: {file}")

                try:
                    document_text = process_large_pdf(pdf_path)
                    save_text_to_json(document_text, output_json_path)
                    success_count += 1
                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")
                    failed_pdfs.append(file)

    print(f"\nTotal PDFs successfully processed: {success_count}")
    if failed_pdfs:
        print(f"PDFs that failed to process: {', '.join(failed_pdfs)}")
    else:
        print("All PDFs processed successfully!")

# pdf_path = "pdfs\\2003\\045-SLLR-SLLR-2003-V-3-PATHMANAYAKY-v.-MAHENTHIRAN.pdf"
# output_json_path = "pdfs\\2003\\045-SLLR-SLLR-2003-V-3-PATHMANAYAKY-v.-MAHENTHIRAN.json"

# document_text = process_large_pdf(pdf_path)
# save_text_to_json(document_text, output_json_path)

directory_path = "documents/Sinhala documents"
process_pdfs_in_directory(directory_path)
