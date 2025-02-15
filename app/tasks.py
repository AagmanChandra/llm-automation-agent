import os
import json
import subprocess
import datetime
import shutil
import sqlite3
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("AIPROXY_TOKEN"))
model = SentenceTransformer("all-MiniLM-L6-v2")

# Security check: Prevent accessing files outside /data
def secure_path(file_path):
    abs_path = Path(file_path).resolve()
    if not str(abs_path).startswith(str(DATA_DIR.resolve())):
        raise PermissionError(f"Access denied: {file_path}")
    return abs_path

# Task Dispatcher
TASKS = {}

def register_task(name):
    """Decorator to register task functions."""
    def wrapper(func):
        TASKS[name] = func
        return func
    return wrapper

# --- TASK IMPLEMENTATIONS ---

@register_task("install_uv_and_run_datagen")
def install_uv_and_run_datagen(user_email):
    subprocess.run(["pip", "install", "--upgrade", "uv"], check=True)
    subprocess.run(["python", "-m", "uv", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", user_email], check=True)
    return {"status": 200, "message": "Data generation complete"}

@register_task("format_markdown")
def format_markdown():
    file_path = secure_path(DATA_DIR / "format.md")
    subprocess.run(["npx", "prettier@3.4.2", "--write", file_path], check=True)
    return {"status": 200, "message": "Markdown formatted"}

@register_task("count_wednesdays")
def count_wednesdays():
    input_path = secure_path(DATA_DIR / "dates.txt")
    output_path = secure_path(DATA_DIR / "dates-wednesdays.txt")
    
    with open(input_path, "r") as f:
        dates = [line.strip() for line in f]

    wednesday_count = sum(1 for date in dates if datetime.datetime.strptime(date, "%Y-%m-%d").weekday() == 2)

    with open(output_path, "w") as f:
        f.write(str(wednesday_count))
    
    return {"status": 200, "message": f"Wednesdays counted: {wednesday_count}"}

@register_task("sort_contacts")
def sort_contacts():
    input_path = secure_path(DATA_DIR / "contacts.json")
    output_path = secure_path(DATA_DIR / "contacts-sorted.json")

    with open(input_path, "r") as f:
        contacts = json.load(f)

    sorted_contacts = sorted(contacts, key=lambda x: (x["last_name"], x["first_name"]))

    with open(output_path, "w") as f:
        json.dump(sorted_contacts, f, indent=2)
    
    return {"status": 200, "message": "Contacts sorted"}

@register_task("extract_log_headers")
def extract_log_headers():
    log_dir = secure_path(DATA_DIR / "logs")
    output_path = secure_path(DATA_DIR / "logs-recent.txt")

    log_files = sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:10]

    with open(output_path, "w") as f:
        for log_file in log_files:
            with open(log_file, "r") as lf:
                first_line = lf.readline().strip()
                f.write(first_line + "\n")

    return {"status": 200, "message": "Extracted headers from logs"}

@register_task("extract_markdown_headers")
def extract_markdown_headers():
    docs_dir = secure_path(DATA_DIR / "docs")
    output_path = secure_path(DATA_DIR / "docs/index.json")

    index = {}

    for md_file in docs_dir.glob("*.md"):
        with open(md_file, "r") as f:
            for line in f:
                if line.startswith("# "):
                    index[md_file.name] = line.strip("# ").strip()
                    break

    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

    return {"status": 200, "message": "Markdown headers extracted"}

@register_task("extract_email_sender")
def extract_email_sender():
    input_path = secure_path(DATA_DIR / "email.txt")
    output_path = secure_path(DATA_DIR / "email-sender.txt")

    with open(input_path, "r") as f:
        email_content = f.read()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Extract the sender's email from this text:\n{email_content}"}]
    )

    sender_email = response.choices[0].message.content.strip()

    with open(output_path, "w") as f:
        f.write(sender_email)

    return {"status": 200, "message": "Sender email extracted"}

@register_task("extract_credit_card")
def extract_credit_card():
    input_path = secure_path(DATA_DIR / "credit-card.png")
    output_path = secure_path(DATA_DIR / "credit-card.txt")

    image = Image.open(input_path)
    card_number = pytesseract.image_to_string(image).replace(" ", "").strip()

    with open(output_path, "w") as f:
        f.write(card_number)

    return {"status": 200, "message": "Credit card extracted"}

@register_task("find_similar_comments")
def find_similar_comments():
    input_path = secure_path(DATA_DIR / "comments.txt")
    output_path = secure_path(DATA_DIR / "comments-similar.txt")

    with open(input_path, "r") as f:
        comments = [line.strip() for line in f]

    embeddings = model.encode(comments)
    similarity_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(similarity_matrix, -1)

    idx1, idx2 = divmod(np.argmax(similarity_matrix), similarity_matrix.shape[1])

    with open(output_path, "w") as f:
        f.write(comments[idx1] + "\n" + comments[idx2])

    return {"status": 200, "message": "Most similar comments extracted"}

@register_task("sum_gold_ticket_sales")
def sum_gold_ticket_sales():
    db_path = secure_path(DATA_DIR / "ticket-sales.db")
    output_path = secure_path(DATA_DIR / "ticket-sales-gold.txt")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
    total_sales = cursor.fetchone()[0] or 0

    conn.close()

    with open(output_path, "w") as f:
        f.write(str(total_sales))

    return {"status": 200, "message": "Gold ticket sales calculated"}

# --- EXECUTION HANDLER ---
def execute_task(task_name, *args):
    if task_name in TASKS:
        return TASKS[task_name](*args)
    return {"status": 400, "message": "Unknown task"}

