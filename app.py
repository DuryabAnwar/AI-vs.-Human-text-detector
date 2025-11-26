import gradio as gr
import joblib
import numpy as np

# Load saved model + vectorizer
MODEL_PATH = "models/ai_detector_model.pkl"
VECT_PATH = "models/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

def predict(text):
    text = str(text)
    if not text.strip():
        return {"ai": 0.0, "human": 0.0}, "Please paste text."

    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    classes = model.classes_

    # probability dictionary
    prob_dict = {
        classes[i]: float(probs[i]) 
        for i in range(len(classes))
    }

    # best prediction
    top_idx = int(np.argmax(probs))
    top_label = classes[top_idx]
    top_score = float(probs[top_idx])

    result_text = f"{top_label.upper()} — Confidence: {round(top_score*100,1)}%"

    return prob_dict, result_text


# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        placeholder="Paste text here...",
        label="Text to analyze",
        lines=8
    ),
    outputs=[
        gr.Label(num_top_classes=2, label="Probabilities"),
        gr.Textbox(label="Prediction")
    ],
    title="AI vs Human Text Detector",
    description="TF-IDF + Logistic Regression",
    allow_flagging="never"
)

iface.launch()
