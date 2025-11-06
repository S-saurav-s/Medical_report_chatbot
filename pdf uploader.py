from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import os
import google.generativeai as genai
from chromadb import Client
from chromadb.config import Settings
from flask.cli import load_dotenv

# Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

load_dotenv()

# Gemini setup
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Chroma setup (lightweight local vector store)
chroma_client = Client(Settings(anonymized_telemetry=False))
collection = chroma_client.create_collection("medical_reports")

uploaded_text = ""

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload_report', methods=['POST'])
def upload_report():
    global uploaded_text
    file = request.files.get('report')
    if not file:
        return jsonify({"error": "No file uploaded."}), 400

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    text = ""

    # PDF
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(filepath)
        for page in reader.pages:
            text += page.extract_text() or ""

    # Image (X-ray report, blood test screenshot, etc.)
    elif filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)

    if not text.strip():
        return jsonify({"error": "No readable text found."}), 200

    uploaded_text = text.strip()


    # store in chroma for RAG
    collection.upsert(documents=[uploaded_text], ids=["report_1"])

    return jsonify({"message": "✅ Report uploaded and analyzed successfully!",
                    "reset_chat":True})


@app.route('/ask_report', methods=['POST'])
def ask_report():
    user_input = request.form.get('user_input', '')

    if not uploaded_text:
        return jsonify({"reply": "Please upload a report first."})

    # Retrieve relevant text from collection
    results = collection.query(query_texts=[user_input], n_results=1)
    relevant_info = results['documents'][0][0] if results['documents'] else uploaded_text

    prompt = f"""
    You are an AI medical assistant. You help users understand health reports and answer health-related questions only.

    Instructions:
    - If the user asks to **analyze** or **interpret** the report (e.g. "analyze", "explain", "summarize", "give details", "what does my report say", "provide findings"):
      - Provide a detailed response in this structure:
        ### 🩺 Summary
        - Give a quick plain-language explanation of the overall report.

        ### ⚠️ Key Findings
        - List only the notable abnormal values and explain what they mean.

        ### 💡 Precautions & Recommendations
        - Suggest 3–5 health tips (diet, exercise, lifestyle, or follow-up).

    - If the user asks about **diet**, **nutrition**, **foods**, **lifestyle**, or **exercise**:
      - Give tailored diet and exercise tips based on the findings in the report.

    - If the user asks something **off-topic** (e.g. personal or irrelevant questions):
      - Respond: "Sorry, I am a medical assistant and can only answer health-related questions based on your report."

    Report Data:
    {relevant_info}

    User Question:
    {user_input}
    """

    # Gemini reasoning
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    return jsonify({"reply": response.text})


if __name__ == "__main__":
    app.run(debug=True)
