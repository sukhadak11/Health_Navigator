import streamlit as st
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from huggingface_hub import InferenceClient
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Health Navigator", layout="wide")

# --------------------------------------------------
# CUSTOM UI STYLE
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f4f6fb;
}

h1 {
    color: #0f62fe;
    font-weight: 700;
}

h2 {
    color: #0043ce;
}

.stButton>button {
    background-color: #0f62fe;
    color: white;
    border-radius: 8px;
    padding: 10px 25px;
    font-weight: bold;
}

.stButton>button:hover {
    background-color: #0353e9;
}

div[data-testid="stMetricValue"] {
    font-size: 28px;
    color: #0f62fe;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

description = pd.read_csv("description.csv")
precautions = pd.read_csv("precautions_df.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")
workout = pd.read_csv("workout_df.csv")

svc = pickle.load(open("svc.pkl", "rb"))

# --------------------------------------------------
# HUGGING FACE CLIENT
# --------------------------------------------------

HF_TOKEN = st.secrets["HF_TOKEN"]

client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN
)

# --------------------------------------------------
# SYMPTOMS + DISEASES
# --------------------------------------------------

symptoms_dict = {'itching':0,'skin_rash':1,'nodal_skin_eruptions':2,'continuous_sneezing':3,
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
'red_sore_around_nose':130,'yellow_crust_ooze':131}

diseases_list = {15:'Fungal infection',4:'Allergy',16:'GERD',9:'Chronic cholestasis',14:'Drug Reaction',
33:'Peptic ulcer diseae',1:'AIDS',12:'Diabetes',17:'Gastroenteritis',6:'Bronchial Asthma',
23:'Hypertension',30:'Migraine',7:'Cervical spondylosis',32:'Paralysis (brain hemorrhage)',
28:'Jaundice',29:'Malaria',8:'Chicken pox',11:'Dengue',37:'Typhoid',40:'Hepatitis A',
19:'Hepatitis B',20:'Hepatitis C',21:'Hepatitis D',22:'Hepatitis E',3:'Alcoholic hepatitis',
36:'Tuberculosis',10:'Common Cold',34:'Pneumonia',13:'Dimorphic hemmorhoids(piles)',
18:'Heart attack',39:'Varicose veins',26:'Hypothyroidism',24:'Hyperthyroidism',
25:'Hypoglycemia',31:'Osteoarthritis',5:'Arthritis',0:'Vertigo',2:'Acne',
38:'Urinary tract infection',35:'Psoriasis',27:'Impetigo'}

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict(symptoms):

    vector = np.zeros(len(symptoms_dict))

    for s in symptoms:
        if s in symptoms_dict:
            vector[symptoms_dict[s]] = 1

    probabilities = svc.predict_proba([vector])[0]

    top3 = probabilities.argsort()[-3:][::-1]

    results = []

    for i in top3:
        results.append((diseases_list[i], probabilities[i]*100))

    return results


# --------------------------------------------------
# DISEASE DETAILS
# --------------------------------------------------

def get_disease_details(dis):

    desc = description[description['Disease']==dis]['Description']
    desc = " ".join(desc.values) if len(desc)>0 else "No description"

    pre = precautions[precautions['Disease']==dis][
        ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
    ]

    precautions_list = pre.values[0] if len(pre)>0 else []

    med = medications[medications['Disease']==dis]['Medication']
    med_list = med.values if len(med)>0 else []

    die = diets[diets['Disease']==dis]['Diet']
    diet_list = die.values if len(die)>0 else []

    wrk = workout[workout['disease']==dis]['workout']
    workout_list = wrk.values if len(wrk)>0 else []

    return desc,precautions_list,med_list,diet_list,workout_list


# --------------------------------------------------
# PDF GENERATOR
# --------------------------------------------------

def generate_pdf(name,age,gender,disease,confidence,desc,precautions,meds,diet,workout):

    file_path="medical_report.pdf"

    doc=SimpleDocTemplate(file_path)
    styles=getSampleStyleSheet()

    elements=[]

    elements.append(Paragraph("AI Health Navigator Report",styles['Heading1']))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Patient Name: {name}",styles['Normal']))
    elements.append(Paragraph(f"Age: {age}",styles['Normal']))
    elements.append(Paragraph(f"Gender: {gender}",styles['Normal']))

    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Disease: {disease}",styles['Normal']))
    elements.append(Paragraph(f"Confidence: {confidence:.2f}%",styles['Normal']))

    elements.append(Spacer(1,20))

    elements.append(Paragraph("Description",styles['Heading2']))
    elements.append(Paragraph(desc,styles['Normal']))

    elements.append(Spacer(1,15))
    elements.append(Paragraph("Precautions",styles['Heading2']))
    elements.append(ListFlowable([Paragraph(p,styles['Normal']) for p in precautions]))

    elements.append(Spacer(1,15))
    elements.append(Paragraph("Medications",styles['Heading2']))
    elements.append(ListFlowable([Paragraph(m,styles['Normal']) for m in meds]))

    elements.append(Spacer(1,15))
    elements.append(Paragraph("Diet Plan",styles['Heading2']))
    elements.append(ListFlowable([Paragraph(d,styles['Normal']) for d in diet]))

    elements.append(Spacer(1,15))
    elements.append(Paragraph("Workout",styles['Heading2']))
    elements.append(ListFlowable([Paragraph(w,styles['Normal']) for w in workout]))

    doc.build(elements)

    return file_path


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history=[]

if "chat" not in st.session_state:
    st.session_state.chat=[]


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("AI Health Navigator")

menu=st.sidebar.radio(
"Navigation",
["Predict Disease","Patient History","AI Chatbot"]
)

# --------------------------------------------------
# PREDICT DISEASE PAGE
# --------------------------------------------------

if menu == "Predict Disease":

    st.title("🧑‍⚕ AI Health Navigator")

    # ---------------- Patient Information ----------------

    st.subheader("Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        name = st.text_input("Patient Name")

    with col2:
        age = st.number_input("Age", min_value=1, max_value=120, step=1)

    with col3:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female", "Other"]
        )

    # ---------------- Symptom Selection ----------------

    st.subheader("Select Symptoms")

    symptoms = st.multiselect(
        "Choose Symptoms",
        list(symptoms_dict.keys())
    )

    # ---------------- Predict Button ----------------

    predict_button = st.button("🔍 Predict Disease")

    if predict_button:

        # ----------- INPUT VALIDATION -----------

        if name.strip() == "":
            st.warning("⚠ Please enter patient name")

        elif len(symptoms) == 0:
            st.warning("⚠ Please select at least one symptom")

        else:

            with st.spinner("Analyzing symptoms using AI model..."):

                results = predict(symptoms)

            # ---------------- Top Predictions ----------------

            st.subheader("Top Disease Predictions")

            df = pd.DataFrame(
                results,
                columns=["Disease", "Probability (%)"]
            )

            st.table(df)

            # ---------------- Probability Chart ----------------

            fig = plt.figure()

            plt.barh(
                df["Disease"],
                df["Probability (%)"]
            )

            plt.xlabel("Probability (%)")

            st.pyplot(fig)

            # ---------------- Most Probable Disease ----------------

            disease = results[0][0]
            confidence = results[0][1]

            st.success(f"Most Probable Disease: {disease}")

            st.metric("Model Confidence", f"{confidence:.2f}%")

            # ---------------- Emergency Alert ----------------

            if disease.lower() == "heart attack":
                st.error("🚨 Emergency Alert: Possible Heart Attack! Please seek medical help immediately.")

            # ---------------- Disease Details ----------------

            desc, pre, meds, diet_plan, workout_plan = get_disease_details(disease)

            with st.expander("📖 Description"):
                st.write(desc)

            with st.expander("🛡 Precautions"):
                for p in pre:
                    st.write("✔", p)

            with st.expander("💊 Medications"):
                for m in meds:
                    st.write("💊", m)

            with st.expander("🥗 Diet Plan"):
                for d in diet_plan:
                    st.write("🥗", d)

            with st.expander("🏃 Workout"):
                for w in workout_plan:
                    st.write("🏃", w)

            # ---------------- Generate PDF ----------------

            pdf = generate_pdf(
                name,
                age,
                gender,
                disease,
                confidence,
                desc,
                pre,
                meds,
                diet_plan,
                workout_plan
            )

            with open(pdf, "rb") as f:

                st.download_button(
                    "Download Medical Report",
                    f,
                    file_name="medical_report.pdf"
                )

            # ---------------- Save History ----------------

            st.session_state.history.append({
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Patient": name,
                "Disease": disease,
                "Confidence": confidence
            })


# --------------------------------------------------
# HISTORY
# --------------------------------------------------

elif menu=="Patient History":

    st.title("Prediction History")

    if st.session_state.history:

        df=pd.DataFrame(st.session_state.history)

        st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "history.csv"
        )

    else:

        st.info("No predictions yet")


# --------------------------------------------------
# CHATBOT
# --------------------------------------------------

elif menu=="AI Chatbot":

    st.title("🤖 AI Health Assistant")

    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])

    user=st.chat_input("Ask health question")

    if user:

        st.session_state.chat.append({
            "role":"user",
            "content":user
        })

        st.chat_message("user").write(user)

        response=client.chat_completion(
            messages=st.session_state.chat,
            max_tokens=512
        )

        reply=response.choices[0].message["content"]

        st.session_state.chat.append({
            "role":"assistant",
            "content":reply
        })

        st.chat_message("assistant").write(reply)

    if st.button("Clear Chat"):
        st.session_state.chat=[]
        st.rerun()