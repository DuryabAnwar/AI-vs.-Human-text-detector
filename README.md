# AI-vs.-Human-text-detector
# AI vs Human Text Detector

A machine-learning project that classifies text as **AI-written** or **Human-written** using:

- TF-IDF Vectorizer  
- Logistic Regression  
- Gradio Web Interface  

Dataset used: `cleaned_ai_human.csv`

---

## 🚀 Features
- End-to-end training pipeline  
- Text preprocessing & EDA  
- Model evaluation (accuracy, confusion matrix)  
- Saves trained model + vectorizer  
- GUI built using **Gradio**

---

## 📁 Project Structure
(see tree in repo)

---

## 📦 Installation
1. **Clone the repository**
 git clone https://github.com/your-username/ai-human-detector.git
cd ai-human-detector

2. **Install required dependencies**
pip install -r requirements.txt


3. (Optional) **Open the Colab/Jupyter notebook**
[notebooks/ai_detector.ipynb](https://colab.research.google.com/drive/1Wq1OuJ6tZSgwtBcYTk6F6sBdG88uIT9-?usp=sharing)

## 🧠 Training the Model

To train the model from scratch:

python src/train.py

This will:

✔ Clean and process the text  
✔ Train TF-IDF + Logistic Regression  
✔ Save these files:

models/ai_detector_model.pkl
models/tfidf_vectorizer.pkl

## 🌐 Running the Gradio App

Start the web interface:

python src/app.py

This launches a local server where you can paste text and see:

- AI probability  
- Human probability  
- Final predicted label  
- Confidence score  

## 🔍 Notebook Version (Full EDA)

Explore the project end-to-end:

[notebooks/ai_detector.ipynb](https://colab.research.google.com/drive/1Wq1OuJ6tZSgwtBcYTk6F6sBdG88uIT9-?usp=sharing)

It includes:

- Cleaning  
- Visualization  
- Word frequency  
- TF-IDF analysis  
- Model training  
- Feature importance  

## 📦 Requirements

See `requirements.txt` for all dependencies:

pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
gradio

## 📜 License

MIT

---

## 🤝 Contributing

Feel free to submit pull requests or open issues.

