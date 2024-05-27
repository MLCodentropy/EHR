import fitz  # PyMuPDF
import spacy
import json

# Load pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

def extract_entities(text):
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities[ent.label_] = entities.get(ent.label_, []) + [ent.text]
    return entities

def convert_to_json(data):
    return json.dumps(data, indent=4)

def process_pdf(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    entities = extract_entities(text)
    return convert_to_json(entities)

# Example usage
pdf_path = "C:\\Users\\sulla\\Downloads\\client\\ReportViewer.pdf"
json_data = process_pdf(pdf_path)
print(json_data)
