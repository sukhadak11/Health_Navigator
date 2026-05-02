import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet
from groq import Groq
import matplotlib.pyplot as plt
import base64
import requests
from bs4 import BeautifulSoup
import cv2
import tempfile
import os

# --------------------------------------------------
# MongoDB (optional - graceful fallback to session)
# --------------------------------------------------
try:
    from pymongo import MongoClient
    MONGO_URI = st.secrets.get("MONGO_URI", None)
    if MONGO_URI:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        db = mongo_client["health_navigator"]
        history_col = db["patient_history"]
        MONGO_ENABLED = True
    else:
        MONGO_ENABLED = False
except Exception:
    MONGO_ENABLED = False

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="AI Health Navigator", page_icon="🩺", layout="wide")

# --------------------------------------------------
# CUSTOM UI STYLE
# --------------------------------------------------
st.markdown("""
<style>
.main { background-color: #f4f6fb; }
h1 { color: #0f62fe; font-weight: 700; }
h2 { color: #0043ce; }
.stButton>button {
    background-color: #0f62fe; color: white;
    border-radius: 8px; padding: 10px 25px;
    font-weight: bold; border: none;
}
.stButton>button:hover { background-color: #0353e9; }
div[data-testid="stMetricValue"] { font-size: 28px; color: #0f62fe; }
.alert-critical {
    background-color: #fff1f1; border-left: 4px solid #e53e3e;
    padding: 12px 16px; border-radius: 4px; margin: 10px 0;
    color: #c53030; font-weight: bold;
}
.alert-warning {
    background-color: #fffbeb; border-left: 4px solid #d69e2e;
    padding: 12px 16px; border-radius: 4px; margin: 10px 0; color: #b7791f;
}
.rag-context-box {
    background: #eef4ff; border: 1px solid #c3dafe;
    border-radius: 8px; padding: 10px 14px;
    font-size: 13px; color: #2b4c8c; margin-bottom: 8px;
}
.upload-hint {
    background: #f0fff4; border: 1px dashed #68d391;
    border-radius: 8px; padding: 8px 12px;
    font-size: 12px; color: #276749; margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    description    = pd.read_csv("description.csv")
    precautions_df = pd.read_csv("precautions_df.csv")
    medications    = pd.read_csv("medications.csv")
    diets          = pd.read_csv("diets.csv")
    workout_df     = pd.read_csv("workout_df.csv")
    return description, precautions_df, medications, diets, workout_df

description, precautions_df, medications, diets, workout_df = load_data()

@st.cache_resource
def load_model():
    return pickle.load(open("svc.pkl", "rb"))

svc = load_model()

# --------------------------------------------------
# GROQ CLIENT  (initialised once — NOT called here)
# --------------------------------------------------
groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# --------------------------------------------------
# CRITICAL CONDITIONS
# --------------------------------------------------
CRITICAL_CONDITIONS = {
    "Heart attack":                 ("🚨 Possible Heart Attack! Call emergency services immediately.", "critical"),
    "Paralysis (brain hemorrhage)": ("🚨 Possible stroke! Call emergency services immediately.", "critical"),
    "Dengue":      ("⚠️ Dengue suspected. Seek hospital care urgently.", "warning"),
    "Typhoid":     ("⚠️ Typhoid suspected. Get blood tests done immediately.", "warning"),
    "Tuberculosis":("⚠️ TB suspected. Visit a pulmonologist immediately.", "warning"),
    "AIDS":        ("⚠️ Please consult a specialist. Confidential testing recommended.", "warning"),
    "Malaria":     ("⚠️ Malaria suspected. Start treatment immediately.", "warning"),
}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def clean_list(values):
    return [str(v) for v in values if pd.notna(v) and str(v).strip() not in ("", "nan")]

def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def extract_url_text(url: str) -> str:
    """Scrape readable text from a URL (capped at 3000 chars)."""
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not fetch URL: {e}"

def extract_video_frames(video_bytes: bytes, n_frames: int = 3) -> list:
    """Extract n evenly-spaced frames from a video; return base64 JPEG list."""
    frames_b64 = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 0:
            for idx in [int(total * i / n_frames) for i in range(n_frames)]:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    _, buf = cv2.imencode(".jpg", frame)
                    frames_b64.append(base64.b64encode(buf.tobytes()).decode("utf-8"))
        cap.release()
    finally:
        os.unlink(tmp_path)
    return frames_b64

# --------------------------------------------------
# SYMPTOMS + DISEASE DICT
# --------------------------------------------------
symptoms_dict = {
    'itching':0,'skin_rash':1,'nodal_skin_eruptions':2,'continuous_sneezing':3,
    'shivering':4,'chills':5,'joint_pain':6,'stomach_pain':7,'acidity':8,'ulcers_on_tongue':9,
    'muscle_wasting':10,'vomiting':11,'burning_micturition':12,'spotting_ urination':13,
    'fatigue':14,'weight_gain':15,'anxiety':16,'cold_hands_and_feets':17,'mood_swings':18,
    'weight_loss':19,'restlessness':20,'lethargy':21,'patches_in_throat':22,'irregular_sugar_level':23,
    'cough':24,'high_fever':25,'sunken_eyes':26,'breathlessness':27,'sweating':28,'dehydration':29,
    'indigestion':30,'headache':31,'yellowish_skin':32,'dark_urine':33,'nausea':34,
    'loss_of_appetite':35,'pain_behind_the_eyes':36,'back_pain':37,'constipation':38,
    'abdominal_pain':39,'diarrhoea':40,'mild_fever':41,'yellow_urine':42,'yellowing_of_eyes':43,
    'acute_liver_failure':44,'fluid_overload':45,'swelling_of_stomach':46,'swelled_lymph_nodes':47,
    'malaise':48,'blurred_and_distorted_vision':49,'phlegm':50,'throat_irritation':51,
    'redness_of_eyes':52,'sinus_pressure':53,'runny_nose':54,'congestion':55,'chest_pain':56,
    'weakness_in_limbs':57,'fast_heart_rate':58,'pain_during_bowel_movements':59,
    'pain_in_anal_region':60,'bloody_stool':61,'irritation_in_anus':62,'neck_pain':63,
    'dizziness':64,'cramps':65,'bruising':66,'obesity':67,'swollen_legs':68,'swollen_blood_vessels':69,
    'puffy_face_and_eyes':70,'enlarged_thyroid':71,'brittle_nails':72,'swollen_extremeties':73,
    'excessive_hunger':74,'extra_marital_contacts':75,'drying_and_tingling_lips':76,
    'slurred_speech':77,'knee_pain':78,'hip_joint_pain':79,'muscle_weakness':80,'stiff_neck':81,
    'swelling_joints':82,'movement_stiffness':83,'spinning_movements':84,'loss_of_balance':85,
    'unsteadiness':86,'weakness_of_one_body_side':87,'loss_of_smell':88,'bladder_discomfort':89,
    'foul_smell_of urine':90,'continuous_feel_of_urine':91,'passage_of_gases':92,'internal_itching':93,
    'toxic_look_(typhos)':94,'depression':95,'irritability':96,'muscle_pain':97,'altered_sensorium':98,
    'red_spots_over_body':99,'belly_pain':100,'abnormal_menstruation':101,'dischromic _patches':102,
    'watering_from_eyes':103,'increased_appetite':104,'polyuria':105,'family_history':106,
    'mucoid_sputum':107,'rusty_sputum':108,'lack_of_concentration':109,'visual_disturbances':110,
    'receiving_blood_transfusion':111,'receiving_unsterile_injections':112,'coma':113,
    'stomach_bleeding':114,'distention_of_abdomen':115,'history_of_alcohol_consumption':116,
    'fluid_overload.1':117,'blood_in_sputum':118,'prominent_veins_on_calf':119,'palpitations':120,
    'painful_walking':121,'pus_filled_pimples':122,'blackheads':123,'scurring':124,'skin_peeling':125,
    'silver_like_dusting':126,'small_dents_in_nails':127,'inflammatory_nails':128,'blister':129,
    'red_sore_around_nose':130,'yellow_crust_ooze':131
}

diseases_list = {
    15:'Fungal infection',4:'Allergy',16:'GERD',9:'Chronic cholestasis',14:'Drug Reaction',
    33:'Peptic ulcer diseae',1:'AIDS',12:'Diabetes',17:'Gastroenteritis',6:'Bronchial Asthma',
    23:'Hypertension',30:'Migraine',7:'Cervical spondylosis',32:'Paralysis (brain hemorrhage)',
    28:'Jaundice',29:'Malaria',8:'Chicken pox',11:'Dengue',37:'Typhoid',40:'Hepatitis A',
    19:'Hepatitis B',20:'Hepatitis C',21:'Hepatitis D',22:'Hepatitis E',3:'Alcoholic hepatitis',
    36:'Tuberculosis',10:'Common Cold',34:'Pneumonia',13:'Dimorphic hemmorhoids(piles)',
    18:'Heart attack',39:'Varicose veins',26:'Hypothyroidism',24:'Hyperthyroidism',
    25:'Hypoglycemia',31:'Osteoarthritis',5:'Arthritis',0:'Vertigo',2:'Acne',
    38:'Urinary tract infection',35:'Psoriasis',27:'Impetigo'
}

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
def predict(symptoms_input):
    vector = np.zeros(len(symptoms_dict))
    for s in symptoms_input:
        if s in symptoms_dict:
            vector[symptoms_dict[s]] = 1
    probs = svc.predict_proba([vector])[0]
    top3  = probs.argsort()[-3:][::-1]
    return [(diseases_list[i], probs[i] * 100) for i in top3]

def get_disease_details(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join(desc.values) if len(desc) > 0 else "No description available."
    pre  = precautions_df[precautions_df['Disease'] == dis][
        ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre_list = clean_list(pre.values[0]) if len(pre) > 0 else []
    med  = medications[medications['Disease'] == dis]['Medication']
    diet = diets[diets['Disease'] == dis]['Diet']
    wrk  = workout_df[workout_df['disease'] == dis]['workout']
    return desc, pre_list, clean_list(med.values), clean_list(diet.values), clean_list(wrk.values)

# --------------------------------------------------
# RAG + SYSTEM PROMPT
# --------------------------------------------------
def build_rag_context(disease, desc, pre, meds, diet_plan, workout_plan, symptoms_input):
    return "\n".join([
        f"PREDICTED DISEASE: {disease}",
        f"DESCRIPTION: {desc}",
        f"SYMPTOMS REPORTED: {', '.join(symptoms_input)}",
        f"PRECAUTIONS: {'; '.join(pre) if pre else 'N/A'}",
        f"MEDICATIONS: {'; '.join(meds) if meds else 'N/A'}",
        f"RECOMMENDED DIET: {'; '.join(diet_plan) if diet_plan else 'N/A'}",
        f"RECOMMENDED WORKOUT: {'; '.join(workout_plan) if workout_plan else 'N/A'}",
    ])

def build_system_prompt(rag_context=None):
    base = (
        "You are a knowledgeable AI health assistant. "
        "Give clear, compassionate, evidence-based health guidance. "
        "Always remind users to consult a licensed doctor for final diagnosis. "
        "Never suggest illegal treatments. Refuse unrelated questions. "
        "When given an image: analyze visible symptoms, skin conditions, injuries, or medical reports. "
        "When given video frames: analyze them for visible health-related content. "
        "When given a URL: summarize the health-relevant content. "
        "Format responses clearly using bullet points or sections when helpful."
    )
    if rag_context:
        base += f"\n\nPatient prediction context (prioritize this in your answers):\n---\n{rag_context}\n---"
    return base

# --------------------------------------------------
# CHART + PDF
# --------------------------------------------------
def render_prediction_chart(results):
    fig, ax = plt.subplots(figsize=(7, 2.5))
    bars = ax.barh([r[0] for r in results], [r[1] for r in results],
                   color=['#0f62fe','#6ea6ff','#d0e2ff'][:len(results)], height=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=11)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=10)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout()
    return fig

def generate_pdf(name, age, gender, disease, confidence,
                 desc, pre, meds, diet, workout, symptoms_input):
    file_path = "medical_report.pdf"
    doc, styles, els = SimpleDocTemplate(file_path), getSampleStyleSheet(), []
    els += [Paragraph("AI Health Navigator — Medical Report", styles['Heading1']),
            Spacer(1,10),
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']),
            Spacer(1,16),
            Paragraph("Patient Information", styles['Heading2']),
            Paragraph(f"Name: {name} | Age: {age} | Gender: {gender}", styles['Normal']),
            Spacer(1,14),
            Paragraph("Symptoms Reported", styles['Heading2']),
            Paragraph(", ".join(symptoms_input), styles['Normal']),
            Spacer(1,14),
            Paragraph("Prediction Result", styles['Heading2']),
            Paragraph(f"Disease: {disease} | Confidence: {confidence:.2f}%", styles['Normal']),
            Spacer(1,14),
            Paragraph("Description", styles['Heading2']),
            Paragraph(str(desc), styles['Normal'])]
    for title, items in [("Precautions",pre),("Medications",meds),("Diet Plan",diet),("Workout",workout)]:
        if items:
            els += [Spacer(1,12), Paragraph(title, styles['Heading2']),
                    ListFlowable([Paragraph(str(i), styles['Normal']) for i in items])]
    els += [Spacer(1,20),
            Paragraph("⚠ Disclaimer: AI-generated for informational purposes only. "
                       "Consult a qualified healthcare provider.", styles['Normal'])]
    doc.build(els)
    return file_path

def save_to_mongo(record):
    if MONGO_ENABLED:
        try:
            history_col.insert_one(record)
        except Exception as e:
            st.toast(f"MongoDB save failed: {e}", icon="⚠️")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
for k, v in {"history":[],"chat":[],"chat_media":[],"rag_context":None,
             "last_disease":None,"last_symptoms":[]}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("🩺 AI Health Navigator")
menu = st.sidebar.radio("Navigation", ["Predict Disease", "Patient History", "AI Chatbot"])

# ==================================================
# PAGE: PREDICT DISEASE
# ==================================================
if menu == "Predict Disease":

    st.title("🧑‍⚕️ AI Health Navigator")
    st.subheader("Patient Information")
    col1, col2, col3 = st.columns(3)
    with col1: name   = st.text_input("Patient Name")
    with col2: age    = st.number_input("Age", min_value=1, max_value=120)
    with col3: gender = st.selectbox("Gender", ["Male","Female","Other"])

    st.subheader("Select Symptoms")
    symptoms_input = st.multiselect("Choose symptoms", list(symptoms_dict.keys()))

    if st.button("🔍 Predict Disease"):
        errors = []
        if not name.strip():   errors.append("Patient name is required.")
        if not symptoms_input: errors.append("Please select at least one symptom.")
        for e in errors: st.warning(e)

        if not errors:
            with st.spinner("Running prediction model..."):
                results = predict(symptoms_input)
            disease, confidence = results[0]

            if disease in CRITICAL_CONDITIONS:
                msg, level = CRITICAL_CONDITIONS[disease]
                st.markdown(f'<div class="{"alert-critical" if level=="critical" else "alert-warning"}">{msg}</div>',
                            unsafe_allow_html=True)

            st.subheader("Top 3 Disease Predictions")
            c1, c2 = st.columns(2)
            with c1:
                df = pd.DataFrame(results, columns=["Disease","Probability (%)"])
                df["Probability (%)"] = df["Probability (%)"].round(2)
                st.dataframe(df, use_container_width=True, hide_index=True)
            with c2:
                fig = render_prediction_chart(results)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.success(f"Most Probable Disease: **{disease}**")
            st.metric("Model Confidence", f"{confidence:.2f}%")

            desc, pre, meds, diet_plan, workout_plan = get_disease_details(disease)
            with st.expander("📋 Description"):  st.write(desc)
            with st.expander("🛡️ Precautions"): [st.write("✔", p) for p in pre]
            with st.expander("💊 Medications"):  [st.write("💊", m) for m in meds]
            with st.expander("🥗 Diet Plan"):    [st.write("🥗", d) for d in diet_plan]
            with st.expander("🏃 Workout"):      [st.write("🏃", w) for w in workout_plan]

            rag_ctx = build_rag_context(disease, desc, pre, meds, diet_plan, workout_plan, symptoms_input)
            st.session_state.update({"rag_context": rag_ctx, "last_disease": disease,
                                      "last_symptoms": symptoms_input, "chat": [], "chat_media": []})
            st.info("💡 Go to **AI Chatbot** — now loaded with your prediction context.")

            pdf = generate_pdf(name, age, gender, disease, confidence,
                               desc, pre, meds, diet_plan, workout_plan, symptoms_input)
            with open(pdf,"rb") as f:
                st.download_button("📄 Download Medical Report (PDF)", f,
                    file_name=f"report_{name.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf")

            record = {"Time": datetime.now().strftime("%Y-%m-%d %H:%M"), "Patient": name,
                      "Age": age, "Gender": gender, "Symptoms": ", ".join(symptoms_input),
                      "Disease": disease, "Confidence": round(confidence, 2)}
            st.session_state["history"].append(record)
            save_to_mongo(record)

# ==================================================
# PAGE: PATIENT HISTORY
# ==================================================
elif menu == "Patient History":
    st.title("📁 Prediction History")
    all_records = st.session_state["history"].copy()

    if MONGO_ENABLED:
        try:
            mongo_recs = list(history_col.find({}, {"_id":0}).sort("Time",-1).limit(100))
            existing   = {(r["Time"], r["Patient"]) for r in all_records}
            for r in mongo_recs:
                if (r.get("Time"), r.get("Patient")) not in existing:
                    all_records.append(r)
            st.caption(f"Showing {len(all_records)} records (MongoDB + session)")
        except Exception as e:
            st.warning(f"Could not fetch MongoDB history: {e}")

    if all_records:
        df = pd.DataFrame(all_records)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download as CSV", df.to_csv(index=False),
                           file_name="patient_history.csv", mime="text/csv")
    else:
        st.info("No predictions yet.")

# ==================================================
# PAGE: AI CHATBOT  — multimodal
# ==================================================
elif menu == "AI Chatbot":

    st.title("🤖 AI Health Assistant")

    # RAG badge
    if st.session_state.get("rag_context"):
        st.markdown(f'<div class="rag-context-box">🔗 RAG context: '
                    f'<strong>{st.session_state["last_disease"]}</strong></div>',
                    unsafe_allow_html=True)
    else:
        st.info("Run a prediction first for context-aware answers.")

    # ── Media upload panel ──────────────────────────────────────────────
    with st.expander("📎 Attach media or link", expanded=False):
        st.markdown('<div class="upload-hint">📷 Image • 🎥 Video • 🔗 URL — the bot will analyze it</div>',
                    unsafe_allow_html=True)
        tab_img, tab_vid, tab_url = st.tabs(["🖼️ Image", "🎥 Video", "🔗 URL / Link"])

        with tab_img:
            uploaded_img = st.file_uploader("Upload image (JPG / PNG)",
                                            type=["jpg","jpeg","png"], key="img_upload")
            img_question = st.text_input("Question about the image (optional)",
                                         placeholder="e.g. What skin condition is this?", key="img_q")
            send_img = st.button("Send Image to Bot", key="send_img")

        with tab_vid:
            uploaded_vid = st.file_uploader("Upload video (MP4 / AVI / MOV)",
                                            type=["mp4","avi","mov"], key="vid_upload")
            vid_question = st.text_input("Question about the video (optional)",
                                         placeholder="e.g. Analyze these frames", key="vid_q")
            send_vid = st.button("Send Video to Bot", key="send_vid")

        with tab_url:
            pasted_url   = st.text_input("Paste a health article URL",
                                         placeholder="https://...", key="url_input")
            url_question = st.text_input("Question about the link (optional)",
                                         placeholder="e.g. Summarize for my condition", key="url_q")
            send_url = st.button("Send URL to Bot", key="send_url")

    # ── Render existing chat ────────────────────────────────────────────
    for i, msg in enumerate(st.session_state["chat"]):
        with st.chat_message(msg["role"]):
            media = st.session_state["chat_media"][i] if i < len(st.session_state["chat_media"]) else None
            if media:
                if media["type"] == "image":
                    st.image(media["data"], caption="Attached image", width=320)
                elif media["type"] == "video_frames":
                    cols = st.columns(len(media["data"]))
                    for col, fb64 in zip(cols, media["data"]):
                        col.image(base64.b64decode(fb64), use_container_width=True)
                elif media["type"] == "url":
                    st.markdown(f"🔗 **Link:** {media['data']}")
            st.write(msg["content"])

    # ── Core helper: call Groq with auto model selection ────────────────
    def call_groq(api_messages: list) -> str:
        has_image = any(
            isinstance(m.get("content"), list) and
            any(c.get("type") == "image_url" for c in m["content"])
            for m in api_messages
        )
        model = "meta-llama/llama-4-scout-17b-16e-instruct" if has_image else "llama-3.1-8b-instant"

        try:
            resp = groq_client.chat.completions.create(
                model=model, messages=api_messages, max_tokens=700
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error: {e}"

    # ── Core helper: append user turn + get reply ───────────────────────
    def append_and_reply(user_text: str, media_record, groq_content):
        """
        user_text    : plain text shown in chat history
        media_record : dict with type + data, or None
        groq_content : str or list (vision format) sent to the API
        """
        st.session_state["chat"].append({"role": "user", "content": user_text})
        st.session_state["chat_media"].append(media_record)

        system_prompt = build_system_prompt(st.session_state.get("rag_context"))
        history_msgs  = st.session_state["chat"][:-1]   # exclude the message we just added
        api_messages  = (
            [{"role": "system", "content": system_prompt}]
            + [{"role": m["role"], "content": m["content"]} for m in history_msgs]
            + [{"role": "user",   "content": groq_content}]
        )

        with st.chat_message("user"):
            if media_record:
                if media_record["type"] == "image":
                    st.image(media_record["data"], caption="Attached image", width=320)
                elif media_record["type"] == "video_frames":
                    cols = st.columns(len(media_record["data"]))
                    for col, fb64 in zip(cols, media_record["data"]):
                        col.image(base64.b64decode(fb64), use_container_width=True)
                elif media_record["type"] == "url":
                    st.markdown(f"🔗 **Link:** {media_record['data']}")
            st.write(user_text)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = call_groq(api_messages)
            st.write(reply)

        st.session_state["chat"].append({"role": "assistant", "content": reply})
        st.session_state["chat_media"].append(None)

    # ── IMAGE ───────────────────────────────────────────────────────────
    if send_img and uploaded_img:
        img_bytes  = uploaded_img.read()
        mime       = "image/jpeg" if uploaded_img.name.lower().endswith(("jpg","jpeg")) else "image/png"
        user_text  = img_question.strip() or "Please analyze this image and give health observations."
        groq_content = [
            {"type": "text",      "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_to_base64(img_bytes)}"}}
        ]
        append_and_reply(user_text, {"type":"image","data":img_bytes}, groq_content)

    # ── VIDEO ───────────────────────────────────────────────────────────
    elif send_vid and uploaded_vid:
        with st.spinner("Extracting frames from video..."):
            frames_b64 = extract_video_frames(uploaded_vid.read(), n_frames=3)
        if not frames_b64:
            st.warning("Could not extract frames. Try MP4 format.")
        else:
            user_text    = vid_question.strip() or "Please analyze these video frames for health observations."
            groq_content = [{"type":"text","text":user_text}] + [
                {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{fb}"}}
                for fb in frames_b64
            ]
            append_and_reply(user_text, {"type":"video_frames","data":frames_b64}, groq_content)

    # ── URL ─────────────────────────────────────────────────────────────
    elif send_url and pasted_url.strip():
        with st.spinner("Reading the URL..."):
            url_text = extract_url_text(pasted_url.strip())
        user_text    = url_question.strip() or "Please summarize the health information from this article."
        groq_content = f"{user_text}\n\n[Content from {pasted_url}]:\n{url_text}"
        append_and_reply(user_text, {"type":"url","data":pasted_url.strip()}, groq_content)

    # ── PLAIN TEXT ──────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a health question...")

    if user_input:
        system_prompt = build_system_prompt(st.session_state.get("rag_context"))
        st.session_state["chat"].append({"role":"user","content":user_input})
        st.session_state["chat_media"].append(None)

        api_messages = (
            [{"role":"system","content":system_prompt}]
            + [{"role":m["role"],"content":m["content"]} for m in st.session_state["chat"]]
        )

        st.chat_message("user").write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = call_groq(api_messages)
            st.write(reply)

        st.session_state["chat"].append({"role":"assistant","content":reply})
        st.session_state["chat_media"].append(None)

    # ── Utility buttons ─────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat"] = []
            st.session_state["chat_media"] = []
            st.rerun()
    with col_b:
        if st.button("🔄 Reset Context"):
            st.session_state.update({"rag_context":None,"last_disease":None,
                                      "last_symptoms":[],"chat":[],"chat_media":[]})
            st.rerun()