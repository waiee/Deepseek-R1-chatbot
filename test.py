import pdfplumber
import pytesseract
from pdf2image import convert_from_path

pdf_path = "Recommendationletter.pdf"  # Change to your actual file path

# Try pdfplumber first (normal text extraction)
with pdfplumber.open(pdf_path) as pdf:
    text = " ".join([page.extract_text() or '' for page in pdf.pages if page.extract_text()])

if not text.strip():  # If no text, use OCR
    print("⚠️ No text found. Using OCR...")
    images = convert_from_path(pdf_path)
    text = " ".join([pytesseract.image_to_string(img) for img in images])

print("Extracted Text Preview:\n", text[:1000])  # Print first 1000 characters
