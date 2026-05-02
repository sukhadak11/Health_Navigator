# 🩺 Health Navigator

**Health Navigator** is a web-based health recommendation system that assists users by analyzing symptoms, and suggesting appropriate workouts, diets, medications, and precautions using machine learning.

---

## 🚀 Features

- 🔍 **Symptom Analysis** – Accepts user-input symptoms and analyzes severity.
- 💊 **Medication Recommendation** – Provides basic info and suggested dosages.
- 🥗 **Diet Planning** – Suggests diet plans based on health conditions.
- 🏋️‍♂️ **Workout Plans** – Recommends workouts according to user fitness needs.
- ⚠️ **Precautions** – Offers health and safety advice for various conditions.
- 🧠 **Machine Learning Integration** – Uses a trained ML model (`svc.pkl`) for predictions.

---

## 🗂️ Project Structure

<pre>
Health_Navigator/
├── main.py # Main application file
├── svc.pkl # Trained ML model (Support Vector Classifier)
├── Medicine.ipynb # Notebook for medication recommendation logic
├── Notes.txt # Development notes and ideas
├── templates/ # HTML templates for UI
├── static/ # Static files (CSS, JS, images)
├── Symptom-severity.csv # Symptom severity dataset
├── symptoms_df.csv # Symptom dataframe
├── meds.csv # Medication data
├── diets.csv # Diet recommendation data
├── workout_df.csv # Workout data
├── precautions_df.csv # Health precautions
├── description.csv # Disease descriptions
</pre>

---
## 💡 How It Works
- User inputs symptoms via web UI.

- Backend uses the ML model (svc.pkl) to predict likely diseases.

- Corresponding data (medications, diets, workouts, precautions) is fetched from .csv files.

- Recommendations are displayed on the screen.
---
## 📈 Machine Learning Model
- Built using Support Vector Classification (SVC)

- Trained on Training.csv with preprocessed health data

- Serialized model saved as svc.pkl
---

---

## ⚙️ Installation & Setup

1. **Clone the Repository**

```bash
git clone https://github.com/sukhadak11/Health_Navigator.git
cd Health_Navigator

python main.py

 
