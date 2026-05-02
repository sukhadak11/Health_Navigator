# 🩺 AI Health Navigator

## 📌 Project Overview

AI Health Navigator is an intelligent healthcare web application that predicts possible diseases based on user symptoms and provides personalized health recommendations. It integrates Machine Learning and Generative AI to deliver a complete health assistance system.

---

## 🚀 Features

* 🔍 Disease prediction using Machine Learning (SVC model)
* 📊 Top 3 disease predictions with probability
* 🧾 Detailed insights:

  * Description
  * Precautions
  * Medications
  * Diet plan
  * Workout suggestions
* 📄 PDF medical report generation
* 🤖 AI Chatbot with RAG (Retrieval-Augmented Generation)
* 🖼️ Image analysis (health-related)
* 🎥 Video frame analysis
* 🔗 URL-based health content summarization
* 📁 Patient history tracking (MongoDB / session)

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Machine Learning:** Scikit-learn (SVC Model)
* **Generative AI:** Groq API (LLM)
* **Database:** MongoDB (optional)
* **Libraries:** Pandas, NumPy, Matplotlib, OpenCV, BeautifulSoup, ReportLab

---

## 📂 Project Structure

```
Health_Navigator/
│
├── main.py
├── requirements.txt
├── README.md
├── svc.pkl
├── description.csv
├── precautions_df.csv
├── medications.csv
├── diets.csv
├── workout_df.csv
├── .streamlit/
│   └── secrets.toml (not uploaded)
```

---

## ▶️ Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/sukhadak11/Health_Navigator.git
cd Health_Navigator
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add secrets

Create file:

```
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_api_key"
MONGO_URI = "your_mongodb_uri"
```

### 4️⃣ Run the app

```bash
streamlit run main.py
```

---

## 🌐 Live Demo

👉 https://healthnavigator.streamlit.app
*(Replace with your actual deployed link if different)*

---

## 🔐 Security Note

* API keys are securely managed using Streamlit secrets
* Sensitive data is excluded using `.gitignore`

---

## ⚠️ Disclaimer

This application is for informational purposes only and does not replace professional medical advice. Always consult a qualified healthcare provider.

