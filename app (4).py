import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib
import pickle, warnings, requests, json
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Drug Safety Assistant", page_icon="💊", layout="wide")

st.markdown("""
<style>
/* Works in both light and dark mode */
.title{font-size:2.2rem;font-weight:800;color:#00a884;text-align:center}
.subtitle{font-size:1rem;color:#666;text-align:center;margin-bottom:1rem}
.metric-card{padding:1rem;border-radius:12px;text-align:center;margin-bottom:0.5rem;border:1px solid #ddd}
.metric-value{font-size:2rem;font-weight:700}
.metric-label{font-size:0.85rem;opacity:0.7}
.recommend-box{border-left:4px solid #00a884;padding:1rem 1.5rem;border-radius:8px;margin:0.5rem 0;border:1px solid #eee}
.side-box{padding:0.5rem 1rem;border-radius:8px;margin:0.3rem 0;border:1px solid #eee}
.risk-box{padding:1.5rem;border-radius:12px;text-align:center;margin:1rem 0}
.landing-hero{text-align:center;padding:3rem 1rem}
.feature-card{padding:1.5rem;border-radius:12px;border:1px solid #ddd;margin:0.5rem;text-align:center}
.chat-user{background:#00a884;color:white;padding:0.75rem 1rem;border-radius:12px 12px 2px 12px;margin:0.5rem 0;max-width:80%;margin-left:auto}
.chat-ai{background:#f0f0f0;color:#222;padding:0.75rem 1rem;border-radius:12px 12px 12px 2px;margin:0.5rem 0;max-width:80%}
.stButton>button{background:#00a884;color:white;border:none;border-radius:8px;padding:0.5rem 1rem;font-weight:600}
.stButton>button:hover{background:#007a60}
section[data-testid="stSidebar"]{background:#00a884}
section[data-testid="stSidebar"] *{color:white !important}
</style>
""", unsafe_allow_html=True)

# ── Drug Database ─────────────────────────────────────────────────────────────
DRUG_DATA = {
    "Paracetamol (Acetaminophen)":{"class":"Painkiller","common":["Nausea","Stomach pain","Loss of appetite"],"serious":["Liver damage","Kidney damage","Severe allergic reaction"],"risk_base":1,"usual_dose":"500mg - 1000mg"},
    "Ibuprofen":{"class":"Painkiller","common":["Stomach upset","Heartburn","Dizziness","Headache"],"serious":["Stomach bleeding","Kidney problems","Heart attack risk"],"risk_base":2,"usual_dose":"200mg - 400mg"},
    "Aspirin":{"class":"Painkiller","common":["Stomach irritation","Nausea","Heartburn"],"serious":["Internal bleeding","Reye's syndrome in children","Stroke risk"],"risk_base":2,"usual_dose":"300mg - 600mg"},
    "Diclofenac":{"class":"Painkiller","common":["Stomach pain","Nausea","Headache"],"serious":["Stomach ulcers","Heart attack","Kidney failure"],"risk_base":2,"usual_dose":"50mg - 75mg"},
    "Tramadol":{"class":"Painkiller","common":["Nausea","Dizziness","Constipation"],"serious":["Seizures","Addiction risk","Breathing problems"],"risk_base":3,"usual_dose":"50mg - 100mg"},
    "Naproxen":{"class":"Painkiller","common":["Stomach upset","Heartburn","Drowsiness"],"serious":["GI bleeding","Kidney damage","Heart problems"],"risk_base":2,"usual_dose":"250mg - 500mg"},
    "Codeine":{"class":"Painkiller","common":["Constipation","Drowsiness","Nausea"],"serious":["Addiction","Breathing problems","Liver damage"],"risk_base":3,"usual_dose":"15mg - 60mg"},
    "Mefenamic Acid":{"class":"Painkiller","common":["Stomach upset","Diarrhea","Dizziness"],"serious":["Kidney failure","Stomach bleeding","Seizures"],"risk_base":2,"usual_dose":"250mg - 500mg"},
    "Amoxicillin":{"class":"Antibiotic","common":["Diarrhea","Stomach upset","Skin rash"],"serious":["Severe allergic reaction","Liver problems","Colitis"],"risk_base":1,"usual_dose":"250mg - 500mg"},
    "Ciprofloxacin":{"class":"Antibiotic","common":["Nausea","Diarrhea","Headache"],"serious":["Tendon rupture","Nerve damage","Heart rhythm problems"],"risk_base":3,"usual_dose":"250mg - 500mg"},
    "Azithromycin":{"class":"Antibiotic","common":["Nausea","Diarrhea","Stomach pain"],"serious":["Heart rhythm problems","Liver damage","Severe allergic reaction"],"risk_base":2,"usual_dose":"250mg - 500mg"},
    "Doxycycline":{"class":"Antibiotic","common":["Nausea","Sun sensitivity","Stomach upset"],"serious":["Esophageal damage","Liver toxicity","Intracranial pressure"],"risk_base":2,"usual_dose":"100mg"},
    "Metronidazole":{"class":"Antibiotic","common":["Nausea","Metallic taste","Headache"],"serious":["Nerve damage","Seizures","Severe skin reactions"],"risk_base":2,"usual_dose":"200mg - 400mg"},
    "Clindamycin":{"class":"Antibiotic","common":["Diarrhea","Nausea","Stomach pain"],"serious":["Severe colitis","Allergic reactions","Liver problems"],"risk_base":2,"usual_dose":"150mg - 300mg"},
    "Cephalexin":{"class":"Antibiotic","common":["Diarrhea","Nausea","Stomach upset"],"serious":["Severe allergic reaction","Kidney problems","Colitis"],"risk_base":1,"usual_dose":"250mg - 500mg"},
    "Erythromycin":{"class":"Antibiotic","common":["Nausea","Stomach cramps","Diarrhea"],"serious":["Heart rhythm problems","Liver damage","Hearing loss"],"risk_base":2,"usual_dose":"250mg - 500mg"},
    "Trimethoprim":{"class":"Antibiotic","common":["Nausea","Rash","Headache"],"serious":["Kidney problems","Blood disorders","Severe skin reactions"],"risk_base":2,"usual_dose":"100mg - 200mg"},
    "Nitrofurantoin":{"class":"Antibiotic","common":["Nausea","Headache","Urine discoloration"],"serious":["Lung toxicity","Liver damage","Nerve damage"],"risk_base":2,"usual_dose":"50mg - 100mg"},
    "Ampicillin":{"class":"Antibiotic","common":["Diarrhea","Rash","Nausea"],"serious":["Severe allergic reaction","Colitis","Seizures"],"risk_base":1,"usual_dose":"250mg - 500mg"},
    "Levofloxacin":{"class":"Antibiotic","common":["Nausea","Diarrhea","Headache"],"serious":["Tendon rupture","Heart problems","Mental health effects"],"risk_base":3,"usual_dose":"250mg - 500mg"},
    "Cetirizine":{"class":"Antiallergy","common":["Drowsiness","Dry mouth","Headache"],"serious":["Severe allergic reaction","Fast heartbeat","Tremors"],"risk_base":1,"usual_dose":"10mg"},
    "Loratadine":{"class":"Antiallergy","common":["Headache","Dry mouth","Fatigue"],"serious":["Fast heartbeat","Liver problems","Severe allergic reaction"],"risk_base":1,"usual_dose":"10mg"},
    "Chlorpheniramine":{"class":"Antiallergy","common":["Drowsiness","Dry mouth","Dizziness"],"serious":["Urinary retention","Confusion in elderly","Vision problems"],"risk_base":1,"usual_dose":"4mg"},
    "Phenylephrine":{"class":"Antiallergy","common":["Headache","Nausea","Increased BP"],"serious":["Severe hypertension","Heart attack","Stroke"],"risk_base":2,"usual_dose":"10mg"},
    "Dextromethorphan":{"class":"Antiallergy","common":["Drowsiness","Dizziness","Nausea"],"serious":["Serotonin syndrome","Hallucinations","Dependency"],"risk_base":2,"usual_dose":"10mg - 20mg"},
    "Bromhexine":{"class":"Antiallergy","common":["Nausea","Diarrhea","Dizziness"],"serious":["Severe skin reactions","Liver problems"],"risk_base":1,"usual_dose":"8mg"},
    "Salbutamol (Albuterol)":{"class":"Antiallergy","common":["Tremors","Headache","Fast heartbeat"],"serious":["Severe chest pain","Irregular heartbeat","Low potassium"],"risk_base":2,"usual_dose":"2mg - 4mg"},
    "Omeprazole":{"class":"Gastrointestinal","common":["Headache","Nausea","Diarrhea"],"serious":["Kidney disease","Low magnesium","Bone fractures"],"risk_base":1,"usual_dose":"20mg - 40mg"},
    "Pantoprazole":{"class":"Gastrointestinal","common":["Headache","Diarrhea","Nausea"],"serious":["Kidney inflammation","Low magnesium","C. diff infection"],"risk_base":1,"usual_dose":"40mg"},
    "Ondansetron":{"class":"Gastrointestinal","common":["Headache","Constipation","Fatigue"],"serious":["Heart rhythm problems","Serotonin syndrome","Severe allergic reaction"],"risk_base":2,"usual_dose":"4mg - 8mg"},
    "Domperidone":{"class":"Gastrointestinal","common":["Dry mouth","Headache","Diarrhea"],"serious":["Heart rhythm problems","Sudden cardiac death","Hormonal effects"],"risk_base":2,"usual_dose":"10mg"},
    "Ranitidine":{"class":"Gastrointestinal","common":["Headache","Diarrhea","Nausea"],"serious":["Liver problems","Blood disorders","Kidney problems"],"risk_base":1,"usual_dose":"150mg"},
    "Loperamide":{"class":"Gastrointestinal","common":["Constipation","Dizziness","Nausea"],"serious":["Heart rhythm problems","Toxic megacolon","Ileus"],"risk_base":2,"usual_dose":"2mg"},
    "Lactulose":{"class":"Gastrointestinal","common":["Bloating","Diarrhea","Stomach cramps"],"serious":["Severe electrolyte imbalance","Dehydration"],"risk_base":1,"usual_dose":"15ml - 30ml"},
    "Amlodipine":{"class":"Antihypertensive","common":["Swollen ankles","Flushing","Headache"],"serious":["Severe low BP","Chest pain","Heart failure"],"risk_base":2,"usual_dose":"5mg - 10mg"},
    "Atorvastatin":{"class":"Antihypertensive","common":["Muscle pain","Joint pain","Diarrhea"],"serious":["Severe muscle breakdown","Liver damage","Memory problems"],"risk_base":2,"usual_dose":"10mg - 40mg"},
    "Lisinopril":{"class":"Antihypertensive","common":["Dry cough","Dizziness","Headache"],"serious":["Angioedema","Kidney failure","High potassium"],"risk_base":2,"usual_dose":"5mg - 20mg"},
    "Ramipril":{"class":"Antihypertensive","common":["Cough","Dizziness","Fatigue"],"serious":["Angioedema","Kidney problems","Low BP"],"risk_base":2,"usual_dose":"2.5mg - 10mg"},
    "Metoprolol":{"class":"Antihypertensive","common":["Fatigue","Dizziness","Cold hands"],"serious":["Severe low heart rate","Heart failure","Depression"],"risk_base":2,"usual_dose":"25mg - 100mg"},
    "Warfarin":{"class":"Antihypertensive","common":["Easy bruising","Bleeding gums","Fatigue"],"serious":["Severe internal bleeding","Brain hemorrhage","Stroke"],"risk_base":3,"usual_dose":"1mg - 10mg"},
    "Metformin":{"class":"Antidiabetic","common":["Nausea","Diarrhea","Stomach pain"],"serious":["Lactic acidosis","Vitamin B12 deficiency","Kidney stress"],"risk_base":2,"usual_dose":"500mg - 1000mg"},
    "Glibenclamide":{"class":"Antidiabetic","common":["Low blood sugar","Nausea","Weight gain"],"serious":["Severe hypoglycemia","Liver damage","Blood disorders"],"risk_base":3,"usual_dose":"2.5mg - 5mg"},
    "Insulin (Regular)":{"class":"Antidiabetic","common":["Low blood sugar","Injection site pain","Weight gain"],"serious":["Severe hypoglycemia","Hypokalemia","Lipodystrophy"],"risk_base":3,"usual_dose":"As prescribed by doctor"},
    "Sitagliptin":{"class":"Antidiabetic","common":["Runny nose","Headache","Stomach pain"],"serious":["Pancreatitis","Kidney problems","Severe joint pain"],"risk_base":2,"usual_dose":"100mg"},
    "Sertraline":{"class":"Antidepressant","common":["Nausea","Insomnia","Dizziness"],"serious":["Suicidal thoughts in youth","Serotonin syndrome","Bleeding risk"],"risk_base":2,"usual_dose":"50mg - 200mg"},
    "Diazepam":{"class":"Antidepressant","common":["Drowsiness","Dizziness","Fatigue"],"serious":["Addiction","Respiratory depression","Memory impairment"],"risk_base":3,"usual_dose":"2mg - 10mg"},
    "Alprazolam":{"class":"Antidepressant","common":["Drowsiness","Dizziness","Memory issues"],"serious":["Severe addiction","Withdrawal seizures","Respiratory depression"],"risk_base":3,"usual_dose":"0.25mg - 0.5mg"},
    "Melatonin":{"class":"Antidepressant","common":["Drowsiness","Headache","Dizziness"],"serious":["Hormonal effects","Depression worsening","Vivid dreams"],"risk_base":1,"usual_dose":"0.5mg - 5mg"},
    "Fluoxetine":{"class":"Antidepressant","common":["Nausea","Headache","Insomnia"],"serious":["Serotonin syndrome","Suicidal thoughts","Bleeding risk"],"risk_base":2,"usual_dose":"20mg - 60mg"},
    "Vitamin C (Ascorbic Acid)":{"class":"Supplement","common":["Stomach upset","Diarrhea","Nausea"],"serious":["Kidney stones at high dose","Iron overload","Digestive problems"],"risk_base":1,"usual_dose":"500mg - 1000mg"},
    "Vitamin D3":{"class":"Supplement","common":["Nausea","Constipation","Fatigue"],"serious":["Calcium toxicity","Kidney damage at high dose","Heart rhythm problems"],"risk_base":1,"usual_dose":"1000 IU - 4000 IU"},
    "Iron Supplement":{"class":"Supplement","common":["Constipation","Stomach pain","Dark stools"],"serious":["Iron toxicity","Liver damage","Stomach bleeding"],"risk_base":1,"usual_dose":"65mg - 200mg"},
}

DRUG_CLASS_MAP = {"Painkiller":0,"Antibiotic":1,"Antiallergy":2,"Gastrointestinal":3,"Antihypertensive":4,"Antidiabetic":5,"Antidepressant":6,"Supplement":7}

CATEGORIES = {
    "💊 Painkillers & Fever":["Paracetamol (Acetaminophen)","Ibuprofen","Aspirin","Diclofenac","Tramadol","Naproxen","Codeine","Mefenamic Acid"],
    "🦠 Antibiotics":["Amoxicillin","Ciprofloxacin","Azithromycin","Doxycycline","Metronidazole","Clindamycin","Cephalexin","Erythromycin","Trimethoprim","Nitrofurantoin","Ampicillin","Levofloxacin"],
    "🤧 Cold, Cough & Allergy":["Cetirizine","Loratadine","Chlorpheniramine","Phenylephrine","Dextromethorphan","Bromhexine","Salbutamol (Albuterol)"],
    "🫃 Stomach & Digestion":["Omeprazole","Pantoprazole","Ondansetron","Domperidone","Ranitidine","Loperamide","Lactulose"],
    "❤️ Heart & Blood Pressure":["Amlodipine","Atorvastatin","Lisinopril","Ramipril","Metoprolol","Warfarin"],
    "🩸 Diabetes":["Metformin","Glibenclamide","Insulin (Regular)","Sitagliptin"],
    "🧠 Mental Health & Sleep":["Sertraline","Diazepam","Alprazolam","Melatonin","Fluoxetine"],
    "🌿 Vitamins & Supplements":["Vitamin C (Ascorbic Acid)","Vitamin D3","Iron Supplement"],
}

@st.cache_data
def load_and_train():
    np.random.seed(42); n=800
    data={"drug_class":np.random.choice(list(range(8)),n),"drug_risk_base":np.random.choice([1,2,3],n),"num_ingredients":np.random.randint(1,6,n),"patient_age_group":np.random.choice([0,1,2],n),"has_kidney_issue":np.random.choice([0,1],n,p=[0.8,0.2]),"has_liver_issue":np.random.choice([0,1],n,p=[0.85,0.15]),"has_heart_issue":np.random.choice([0,1],n,p=[0.85,0.15]),"has_diabetes":np.random.choice([0,1],n,p=[0.8,0.2]),"is_pregnant":np.random.choice([0,1],n,p=[0.9,0.1]),"dosage_level":np.random.choice([1,2,3],n),"drug_interactions":np.random.randint(0,6,n),"allergy_history":np.random.choice([0,1],n,p=[0.85,0.15])}
    df=pd.DataFrame(data)
    risk=df["drug_risk_base"]+(df["dosage_level"]==3).astype(int)*2+df["has_kidney_issue"]*2+df["has_liver_issue"]*2+df["has_heart_issue"]+df["has_diabetes"]+df["is_pregnant"]*2+(df["drug_interactions"]>3).astype(int)*2+df["allergy_history"]*2+(df["patient_age_group"]==2).astype(int)
    df["risk_label"]=pd.cut(risk,bins=[-1,4,8,20],labels=[0,1,2]).astype(int)
    X=df.drop("risk_label",axis=1); y=df["risk_label"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    model=RandomForestClassifier(n_estimators=200,random_state=42)
    model.fit(X_train,y_train)
    acc=accuracy_score(y_test,model.predict(X_test))
    return model,X.columns.tolist(),acc

model,feature_cols,accuracy=load_and_train()

# ── AI Doctor function ────────────────────────────────────────────────────────
def ask_ai_doctor(question, lang="English"):
    drug_list = ", ".join(list(DRUG_DATA.keys())[:20]) + " and more..."
    lang_instruction = "Respond in Tamil language (தமிழில் பதில் சொல்லுங்கள்)" if lang=="தமிழ்" else "Respond in simple English"
    system_prompt = f"""You are a friendly AI Doctor assistant inside a Medicine Side Effect Risk Predictor app built for an international science project. 

You have knowledge about these 50 drugs: {drug_list}

Your role:
- Answer questions about medicine safety, side effects, and drug interactions
- Give simple, clear advice that a non-medical person can understand  
- Always recommend consulting a real doctor for serious issues
- Be warm, caring and supportive like a real doctor
- Keep answers concise (3-5 lines max)
- {lang_instruction}
- End every response with: "Remember: Always consult a real doctor before taking any medicine."

Never recommend specific doses. Never diagnose diseases."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "system": system_prompt,
                "messages": [{"role": "user", "content": question}]
            }
        )
        data = response.json()
        if "content" in data:
            return data["content"][0]["text"]
        else:
            return "I'm having trouble connecting right now. Please try again in a moment."
    except:
        return "Connection error. Please check your internet and try again."

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💊 Drug Safety AI")
    st.markdown("---")
    lang = st.radio("🌐 Language / மொழி", ["English", "தமிழ்"])
    st.markdown("---")
    page = st.radio("📍 Navigate", ["🏠 Home", "🔬 Risk Predictor", "🤖 AI Doctor"])
    st.markdown("---")
    st.markdown(f"""
**Model:** Random Forest  
**Accuracy:** {accuracy*100:.1f}%  
**Drugs:** 50  
**Features:** 12  
    """)
    st.markdown("---")
    st.caption("International Science Project 2025")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align:center;padding:2rem 0'>
        <div style='font-size:4rem'>💊</div>
        <h1 style='color:#00a884;font-size:2.5rem;font-weight:800'>AI-Powered Personalized<br>Drug Safety Assistant</h1>
        <p style='font-size:1.1rem;color:#666;max-width:600px;margin:auto'>
        An intelligent system that predicts medicine side effect risks using Machine Learning —
        helping patients and doctors make safer decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✨ What can this app do?")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size:2.5rem'>🔬</div>
            <h3>Risk Predictor</h3>
            <p>Enter your drug and patient details — get instant Low, Medium or High risk prediction with AI confidence score</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size:2.5rem'>🤖</div>
            <h3>AI Doctor</h3>
            <p>Chat with our AI Doctor — ask any question about medicines, side effects, or drug safety in English or Tamil</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size:2.5rem'>📊</div>
            <h3>Smart Analysis</h3>
            <p>Get visual risk charts, dosage trend graphs, smart recommendations and known side effects — all in one place</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💊 50 Drugs Covered Across 8 Categories")
    cols = st.columns(4)
    cats = list(CATEGORIES.keys())
    for i,cat in enumerate(cats):
        with cols[i%4]:
            st.markdown(f"""
            <div style='padding:0.75rem;border-radius:8px;border:1px solid #ddd;margin:0.3rem 0;text-align:center'>
                <b>{cat}</b><br><span style='color:#00a884;font-size:1.2rem'>{len(CATEGORIES[cat])} drugs</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Project Stats")
    s1,s2,s3,s4 = st.columns(4)
    for col,val,label in [(s1,"50","Drugs in Database"),(s2,"12","Patient Parameters"),(s3,f"{accuracy*100:.0f}%","Model Accuracy"),(s4,"2","Languages Supported")]:
        with col:
            st.markdown(f"""
            <div style='text-align:center;padding:1rem;border-radius:12px;border:2px solid #00a884'>
                <div style='font-size:2rem;font-weight:800;color:#00a884'>{val}</div>
                <div style='font-size:0.85rem;color:#666'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to **Risk Predictor** or **AI Doctor**")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Risk Predictor":
    st.markdown('<h2 style="color:#00a884">🔬 Medicine Side Effect Risk Predictor</h2>', unsafe_allow_html=True)
    st.caption("Fill in the drug and patient details below — our AI will predict the risk level instantly")
    st.markdown("---")

    col1,col2 = st.columns(2)

    with col1:
        st.markdown("### 💊 Step 1 — Select Your Drug")
        category  = st.selectbox("Drug Category (What type of medicine?)", list(CATEGORIES.keys()))
        drug_name = st.selectbox("Drug Name (Select the exact medicine)", CATEGORIES[category])
        drug_info = DRUG_DATA[drug_name]
        st.success(f"**Usual Dose:** {drug_info['usual_dose']}  |  **Type:** {drug_info['class']}  |  **Base Risk:** {'⭐'*drug_info['risk_base']}")

        dose_mg = st.text_input("Patient's Actual Dose (type like: 500mg or 1 tablet)", placeholder="e.g. 500mg")
        dosage_level = st.select_slider(
            "Dose Strength Compared to Normal",
            options=[1,2,3],
            format_func=lambda x:{1:"🟢 Low — Less than normal",2:"🟡 Medium — Normal prescribed dose",3:"🔴 High — More than normal"}[x]
        )
        st.caption("Low = taking less than prescribed | Medium = exactly as prescribed | High = more than prescribed (risky!)")
        num_ing = st.slider("Number of Active Ingredients in this drug", 1, 5, 2)
        interactions = st.slider(
            "How many OTHER medicines is the patient taking at the same time?",
            0, 5, 1,
            help="0 = only this medicine | 1-2 = a few others | 3-5 = many medicines together (more medicines = more risk of interaction)"
        )

    with col2:
        st.markdown("### 👤 Step 2 — Enter Patient Details")
        age_group = st.selectbox(
            "Patient Age Group",
            [0,1,2],
            format_func=lambda x:{0:"👦 Child / Teen (below 18 years)",1:"🧑 Adult (18 to 60 years)",2:"👴 Elderly (above 60 years)"}[x]
        )
        st.markdown("**Does the patient already have any of these conditions?**")
        st.caption("Tick all that apply — these affect how the body processes medicine")
        c1,c2 = st.columns(2)
        with c1:
            kidney = st.checkbox("🫘 Kidney Disease")
            liver  = st.checkbox("🫀 Liver Disease")
            heart  = st.checkbox("❤️ Heart Disease")
        with c2:
            diabetes = st.checkbox("🩸 Diabetes")
            pregnant = st.checkbox("🤰 Pregnant")
            allergy  = st.checkbox("⚠️ Past Drug Allergy")

        with st.expander(f"📋 View Known Side Effects of {drug_name}"):
            c1,c2 = st.columns(2)
            with c1:
                st.markdown("**Common (mild, happens often):**")
                for s in drug_info["common"]:
                    st.markdown(f"<div class='side-box'>🟡 {s}</div>",unsafe_allow_html=True)
            with c2:
                st.markdown("**Serious (rare but dangerous):**")
                for s in drug_info["serious"]:
                    st.markdown(f"<div class='side-box'>🔴 {s}</div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)

    if st.button("🔮 Predict My Risk Now", use_container_width=True):
        row={"drug_class":DRUG_CLASS_MAP[drug_info["class"]],"drug_risk_base":drug_info["risk_base"],"num_ingredients":num_ing,"patient_age_group":age_group,"has_kidney_issue":int(kidney),"has_liver_issue":int(liver),"has_heart_issue":int(heart),"has_diabetes":int(diabetes),"is_pregnant":int(pregnant),"dosage_level":dosage_level,"drug_interactions":interactions,"allergy_history":int(allergy)}
        inp=pd.DataFrame([row])
        for c in feature_cols:
            if c not in inp.columns: inp[c]=0
        inp=inp[feature_cols]
        pred=model.predict(inp)[0]
        proba=model.predict_proba(inp)[0]
        risk_pct=round(max(proba)*100,1)

        st.markdown("---")
        st.markdown("### 📋 Your Risk Assessment Result")

        risk_names={0:"LOW",1:"MEDIUM",2:"HIGH"}
        risk_colors={0:"#00c853",1:"#ffa000",2:"#d32f2f"}
        bg_colors={0:"#e8f5e9",1:"#fff8e1",2:"#ffebee"}

        m1,m2,m3=st.columns(3)
        for col_o,val,lbl,clr in [(m1,risk_names[pred],"Risk Level",risk_colors[pred]),(m2,f"{risk_pct}%","AI Confidence","#00a884"),(m3,f"{accuracy*100:.1f}%","Model Accuracy","#7c83fd")]:
            with col_o:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{clr}'>{val}</div><div class='metric-label'>{lbl}</div></div>",unsafe_allow_html=True)

        emoji={0:"🟢",1:"🟡",2:"🔴"}[pred]
        msg={0:f"{drug_name} appears relatively safe for this patient. Standard care applies.",1:f"Moderate risk detected for {drug_name}. Medical advice is recommended.",2:f"HIGH risk detected for {drug_name}. Immediate doctor consultation required!"}[pred]
        st.markdown(f"<div style='padding:1.5rem;border-radius:12px;text-align:center;border:2px solid {risk_colors[pred]};background:{bg_colors[pred]}'><h2 style='color:{risk_colors[pred]}'>{emoji} {risk_names[pred]} RISK</h2><p>{msg}</p>{f'<p><b>Patient dose entered:</b> {dose_mg}</p>' if dose_mg else ''}</div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("### ⚠️ Side Effects to Watch For")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Common side effects (mild, happens often):**")
            for s in drug_info["common"]:
                st.markdown(f"<div class='side-box'>🟡 {s}</div>",unsafe_allow_html=True)
        with c2:
            st.markdown("**Serious side effects (rare but dangerous — watch out!):**")
            icon={0:"🔵",1:"⚠️",2:"🚨"}[pred]
            for s in drug_info["serious"]:
                st.markdown(f"<div class='side-box' style='border-left:3px solid {risk_colors[pred]}'>{icon} {s}</div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("### 💡 What Should You Do?")
        recs={0:["✅ Continue medication exactly as prescribed by your doctor.","✅ Take medicine with food to avoid stomach upset.","✅ A routine checkup every 3 months is enough.","✅ Drink plenty of water throughout the day."],1:["⚠️ Visit your doctor before continuing this medicine.","⚠️ Get kidney and liver tests every 4 to 6 weeks.","⚠️ Do not start any new medicine without doctor approval.","⚠️ Report any new or unusual symptoms to your doctor immediately."],2:["🚨 See a specialist doctor immediately — do not delay.","🚨 Ask your doctor about switching to a safer alternative medicine.","🚨 Do NOT take any other medicines without direct supervision.","🚨 Get emergency blood tests and organ function tests done now.","🚨 Hospital admission may be needed if symptoms appear."]}
        for r in recs[pred]:
            st.markdown(f"<div class='recommend-box'>{r}</div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**📊 Risk Probability Chart**")
            st.caption("How likely is each risk level for this patient and drug?")
            fig,ax=plt.subplots(figsize=(5,3))
            bars=ax.barh([f"🟢 Low Risk",f"🟡 Medium Risk",f"🔴 High Risk"],proba,color=["#00c853","#ffa000","#d32f2f"],height=0.5)
            ax.set_xlim(0,1); ax.set_xlabel("← Less likely          More likely →",fontsize=9)
            ax.spines[['top','right','left']].set_visible(False)
            for bar,p in zip(bars,proba):
                ax.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,f"{p*100:.0f}% chance",va="center",fontsize=9)
            st.pyplot(fig)
        with c2:
            st.markdown("**🧠 What Caused This Risk?**")
            st.caption("Longer bar = this factor had more influence on the AI's decision")
            feat_imp=pd.Series(model.feature_importances_,index=feature_cols).sort_values(ascending=False).head(5)
            readable={"drug_risk_base":"Drug's own risk","dosage_level":"Dosage strength","has_kidney_issue":"Kidney condition","has_liver_issue":"Liver condition","drug_interactions":"Other drugs taken","allergy_history":"Allergy history","patient_age_group":"Patient age","has_heart_issue":"Heart condition","has_diabetes":"Diabetes","is_pregnant":"Pregnancy","num_ingredients":"No. of ingredients","drug_class":"Drug type"}
            feat_imp.index=[readable.get(i,i) for i in feat_imp.index]
            fig2,ax2=plt.subplots(figsize=(5,3))
            feat_imp.plot(kind="barh",ax=ax2,color="#00a884")
            ax2.set_xlabel("Impact on prediction →",fontsize=9)
            ax2.invert_yaxis(); ax2.spines[['top','right','bottom']].set_visible(False)
            st.pyplot(fig2)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("**📈 How Risk Changes With Dose**")
        st.caption("This shows what happens to risk if the dose is increased beyond normal")
        dosage_risks=[]
        for d in [1,2,3]:
            tr=row.copy(); tr["dosage_level"]=d
            ti=pd.DataFrame([tr])[feature_cols]
            dosage_risks.append(model.predict_proba(ti)[0][2]*100)
        fig3,ax3=plt.subplots(figsize=(8,3))
        xlabels=["🟢 Low Dose\n(Less than normal)","🟡 Medium Dose\n(As prescribed)","🔴 High Dose\n(More than normal)"]
        ax3.plot(xlabels,dosage_risks,color="#d32f2f",marker="o",linewidth=2.5,markersize=10)
        ax3.fill_between(xlabels,dosage_risks,alpha=0.1,color="#d32f2f")
        ax3.set_ylabel("High Risk Probability (%)",fontsize=9)
        ax3.set_title(f"{drug_name} — Risk increases as dose goes higher",fontsize=10)
        ax3.spines[['top','right']].set_visible(False)
        for x,y in zip(xlabels,dosage_risks):
            ax3.annotate(f"{y:.0f}% high risk",(x,y),textcoords="offset points",xytext=(0,12),ha="center",fontsize=9)
        st.pyplot(fig3)

        st.success("✅ For research and educational purposes only. Always consult a certified doctor before taking any medicine.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AI DOCTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Doctor":
    st.markdown('<h2 style="color:#00a884">🤖 AI Doctor — Ask Me Anything!</h2>', unsafe_allow_html=True)
    st.caption("Chat with our AI Doctor about medicines, side effects, and drug safety — in English or Tamil")

    col1,col2=st.columns([2,1])
    with col1:
        st.markdown("""
        <div style='background:#e8f5e9;padding:1rem;border-radius:12px;border-left:4px solid #00a884;margin-bottom:1rem'>
        <b>👨‍⚕️ Hi! I'm your AI Doctor.</b><br>
        I can help you with questions about:<br>
        💊 Medicine side effects &nbsp;|&nbsp; 🔄 Drug interactions &nbsp;|&nbsp; ⚠️ Safety warnings<br>
        🌿 When to see a real doctor &nbsp;|&nbsp; 📋 General medicine advice
        </div>""", unsafe_allow_html=True)
    with col2:
        chat_lang = st.radio("Chat Language", ["English", "தமிழ்"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("**💬 Conversation:**")
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown("""
            <div class='chat-ai'>
            👋 Hello! I'm your AI Doctor. Ask me anything about medicines and side effects!<br><br>
            Try asking:<br>
            • "Is Paracetamol safe for children?"<br>
            • "What are the side effects of Ibuprofen?"<br>
            • "Can I take Aspirin and Ibuprofen together?"<br>
            • "I have kidney disease, which painkillers are safer?"
            </div>""", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            if msg["role"]=="user":
                st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>",unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-ai'>👨‍⚕️ {msg['content']}</div>",unsafe_allow_html=True)

    st.markdown("**Ask your question:**")
    c1,c2=st.columns([4,1])
    with c1:
        user_input=st.text_input("Type here...", placeholder="e.g. Is it safe to take Ibuprofen with Paracetamol?", label_visibility="collapsed")
    with c2:
        send=st.button("Send 📤",use_container_width=True)

    st.markdown("**Quick questions — click to ask:**")
    q1,q2,q3,q4=st.columns(4)
    quick=""
    with q1:
        if st.button("💊 Paracetamol safe?"):
            quick="Is Paracetamol safe for everyone including children and elderly?"
    with q2:
        if st.button("🤔 Ibuprofen risks?"):
            quick="What are the main risks and side effects of taking Ibuprofen?"
    with q3:
        if st.button("⚠️ Drug interactions?"):
            quick="What happens when you take multiple medicines at the same time?"
    with q4:
        if st.button("🏥 When see doctor?"):
            quick="When should I stop taking medicine and see a doctor immediately?"

    question = quick if quick else (user_input if send and user_input else "")

    if question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.spinner("👨‍⚕️ AI Doctor is thinking..."):
            reply=ask_ai_doctor(question, chat_lang)
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages=[]
            st.rerun()

    st.markdown("---")
    st.warning("⚠️ The AI Doctor provides general information only. Always consult a real certified doctor for medical decisions.")
