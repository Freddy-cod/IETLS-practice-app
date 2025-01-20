# Import necessary libraries
import streamlit as st
from vosk import Model, KaldiRecognizer
import pyaudio
import spacy
import json
from fpdf import FPDF
import os

# Suppress environment warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load Spacy English model
nlp = spacy.load("en_core_web_sm")

# Static IELTS-style questions
static_questions = [
    "Can you tell me about your hometown?",
    "Describe a memorable vacation you have had.",
    "Do you prefer working alone or in a team? Why?",
    "What are the advantages of public transportation in your country?",
    "How do you usually spend your weekends?"
]

# Speech-to-Text function using VOSK
def transcribe_audio():
    model_path = './vosk-model-small-en-us-0.15'  # Path to VOSK model directory
    if not os.path.exists(model_path):
        st.error("VOSK model not found. Please download and place it in the specified directory.")
        return ""

    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)

    # Set up microphone input
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    st.write("Start speaking... (press Ctrl+C to stop)")
    response_text = ""

    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = json.loads(result)["text"]
                response_text += " " + text
                st.write(f"Transcribed: {text}")
                return response_text.strip()
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        p.terminate()
        return response_text.strip()

# Analyze response for grammar, vocabulary, and pronunciation
def analyze_response(response):
    doc = nlp(response)
    grammar_errors = []
    vocabulary_suggestions = []
    filler_words = ["um", "uh", "like", "so", "you know"]  # Example filler words
    filler_count = sum(response.lower().count(fw) for fw in filler_words)

    # Grammar checks
    for token in doc:
        if token.dep_ == "ROOT" and token.tag_ not in ["VBD", "VBG", "VBN", "VBZ"]:  # Verb form check
            grammar_errors.append(f"Incorrect verb form: {token.text}")

    # Vocabulary analysis (overused words)
    token_counts = doc.count_by(spacy.attrs.LOWER)
    overused_words = [doc.vocab[lower].text for lower, count in token_counts.items() if count > 2]
    for word in overused_words:
        vocabulary_suggestions.append(f"Try using synonyms for '{word}'.")

    # Fluency scoring
    fluency_score = max(0, 10 - filler_count)
    grammar_score = max(0, 10 - len(grammar_errors))

    return {
        "grammar_errors": grammar_errors,
        "vocabulary_suggestions": vocabulary_suggestions,
        "filler_count": filler_count,
        "fluency_score": fluency_score,
        "grammar_score": grammar_score
    }

# Generate a feedback PDF
def generate_report(response, analysis, filename="IELTS_Feedback.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="IELTS Speaking Test Feedback", ln=True, align="C")

    pdf.cell(200, 10, txt="Your Response:", ln=True)
    pdf.multi_cell(0, 10, txt=response)

    pdf.cell(200, 10, txt="Analysis:", ln=True)
    pdf.cell(200, 10, txt=f"- Fluency Score: {analysis['fluency_score']}/10", ln=True)
    pdf.cell(200, 10, txt=f"- Grammar Score: {analysis['grammar_score']}/10", ln=True)

    if analysis["grammar_errors"]:
        pdf.cell(200, 10, txt="Grammar Errors:", ln=True)
        for error in analysis["grammar_errors"]:
            pdf.cell(200, 10, txt=f"  * {error}", ln=True)
    if analysis["vocabulary_suggestions"]:
        pdf.cell(200, 10, txt="Vocabulary Suggestions:", ln=True)
        for suggestion in analysis["vocabulary_suggestions"]:
            pdf.cell(200, 10, txt=f"  * {suggestion}", ln=True)

    pdf.output(filename)
    st.success(f"Feedback report saved as {filename}!")

# Streamlit application UI
st.title("IELTS Speaking Practice and Feedback Tool")

# Display instructions
st.write("""
Welcome to the IELTS Speaking Practice Tool! Follow these steps:
1. Start the test by clicking "Start Test".
2. Answer the question by speaking into your microphone.
3. Get detailed feedback and download a PDF report.
""")

# Main interaction loop
if st.button("Start Test"):
    st.write("Part 1: Introduction")
    question = static_questions[0]  # Example static question
    st.write(f"Question: {question}")

    response = transcribe_audio()
    if response:
        st.write("Your Response:", response)

        analysis = analyze_response(response)
        st.write("Feedback:")
        st.write(f"- Fluency Score: {analysis['fluency_score']}/10")
        st.write(f"- Grammar Score: {analysis['grammar_score']}/10")

        if analysis["grammar_errors"]:
            st.write("Grammar Errors:")
            for error in analysis["grammar_errors"]:
                st.write(f"  * {error}")
        if analysis["vocabulary_suggestions"]:
            st.write("Vocabulary Suggestions:")
            for suggestion in analysis["vocabulary_suggestions"]:
                st.write(f"  * {suggestion}")

        # Generate and download PDF report
        if st.button("Generate Report"):
            generate_report(response, analysis)
