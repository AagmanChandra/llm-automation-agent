import re
import pytesseract
from PIL import Image

def sanitize_task(task: str) -> str:
    """
    Sanitizes task description by converting to lowercase and removing extra spaces.
    """
    return task.strip().lower()

def extract_email(content: str) -> str:
    """
    Extracts an email address from text using regex.
    """
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", content)
    return match.group(0) if match else "No email found"

def extract_text_from_image(image_path: str) -> str:
    """
    Extracts text from an image using OCR (Tesseract).
    """
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)
