import pdfplumber
import pytesseract
from pdf2image import convert_from_path

pdf_path = "Recommendationletter.pdf"

with pdfplumber.open(pdf_path) as pdf:
    text = " ".join([page.extract_text() or '' for page in pdf.pages if page.extract_text()])

if not text.strip(): #ocr if no text
    print("⚠️ No text found. Using OCR...")
    images = convert_from_path(pdf_path)
    text = " ".join([pytesseract.image_to_string(img) for img in images])

print("Extracted Text Preview:\n", text[:1000]) #print 1k characters
