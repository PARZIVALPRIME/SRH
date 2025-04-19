import os
from dotenv import load_dotenv
load_dotenv()

import httpx
import fitz  # PyMuPDF
from flask import Flask, request, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import pytesseract
from PIL import Image
import torch
import re
from docx import Document

# Load Bloom’s Taxonomy model
model_path = "bert_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

label_map = {
    0: "BT1 - Remember",
    1: "BT2 - Understand",
    2: "BT3 - Apply",
    3: "BT4 - Analyze",
    4: "BT5 - Evaluate",
    5: "BT6 - Create"
}

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

def predict_bloom_level(question):
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_map[predicted_class]

def extract_text_from_pdf(path):
    text = ""
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(path):
    return pytesseract.image_to_string(Image.open(path))

def extract_questions_llm(text):
    api_key = os.getenv("GROQ_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts academic-style questions from exam papers. Ignore marks like [10], instructions, or headers. Return only questions, one per line."
            },
            {
                "role": "user",
                "content": f"Extract all meaningful exam or academic questions from this text:\n\n{text}"
            }
        ]
    }

    response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)

    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.status_code} - {response.text}")

    result = response.json()["choices"][0]["message"]["content"]
    questions = [q.strip() for q in result.split("\n") if q.strip()]
    return questions

def clean_question(q):
    return re.sub(r"^\s*\d+[\.\)]\s*", "", q).strip()

LLM_INTRO_PREFIXES = [
    "here are the",
    "below are",
    "the following are",
    "questions extracted",
    "exam-style questions",
    "list of questions",
    "extracted questions",
    "these are"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    question = ""
    error = None
    predictions = []
    source = ""
    bloom_counts = {}

    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        uploaded_file = request.files.get('file_input')

        try:
            if uploaded_file and uploaded_file.filename:
                ext = uploaded_file.filename.lower()
                path = os.path.join("uploads", uploaded_file.filename)
                os.makedirs("uploads", exist_ok=True)
                uploaded_file.save(path)

                if ext.endswith('.pdf'):
                    text = extract_text_from_pdf(path)
                    source = "PDF"
                elif ext.endswith('.docx'):
                    text = extract_text_from_docx(path)
                    source = "DOCX"
                elif ext.endswith(('.png', '.jpg', '.jpeg')):
                    text = extract_text_from_image(path)
                    source = "Image"
                else:
                    error = "Unsupported file type."
                    text = ""

                if text:
                    questions = extract_questions_llm(text)
                    predictions = [
                        (clean_question(q), predict_bloom_level(clean_question(q)))
                        for q in questions
                        if not any(q.lower().startswith(prefix) for prefix in LLM_INTRO_PREFIXES)
                    ]

                    bloom_counts = {
                        "BT1 - Remember": 0,
                        "BT2 - Understand": 0,
                        "BT3 - Apply": 0,
                        "BT4 - Analyze": 0,
                        "BT5 - Evaluate": 0,
                        "BT6 - Create": 0
                    }

                    for _, pred in predictions:
                        if pred in bloom_counts:
                            bloom_counts[pred] += 1

                    if not predictions:
                        bloom_counts = {}

            elif question:
                prediction = predict_bloom_level(question)

            else:
                error = "Please enter a question or upload a file."

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        question=question,
        predictions=predictions,
        source=source,
        error=error,
        bloom_counts=bloom_counts
    )

from flask import send_file
import csv
import io

@app.route('/download_csv')
def download_csv():
    questions = request.args.getlist('question')
    categories = request.args.getlist('category')

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Question', 'Bloom Category'])

    for q, c in zip(questions, categories):
        writer.writerow([q, c])

    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='bloom_predictions.csv'
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)
