import re
from PyPDF2 import PdfReader


def load_pdf(file) -> str:
    """Load text from a PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_txt(file_path: str) -> str:
    """Load text from a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def preprocess_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\s+', ' ', text)           # extra spaces remove
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # non-ascii remove
    text = re.sub(r'\s([?.!,:;])', r'\1', text) # punctuation fix
    text = text.strip().lower()
    return text


def load_and_preprocess(file, file_type: str = "pdf") -> str:
    """
    Main function — load + preprocess in one shot.
    file_type: 'pdf' or 'txt'
    """
    if file_type == "pdf":
        text = load_pdf(file)
    elif file_type == "txt":
        text = load_txt(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return preprocess_text(text)