import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI-Powered Personalized Drug Safety Assistant", page_icon="💊", layout="wide")

st.markdown("""
<style>
.title{font-size:2rem;font-weight:700;color:#00d4aa;text-align:center}
.subtitle{font-size:1rem;color:#aaaaaa;text-align:center;margin-bottom:1rem}
.risk-box{padding:1.5rem;border-radius:12px;text-align:center;margin:1rem 0}
.low{background-color:#1a3d2b;border:2px solid #00c853}
.med{background-color:#3d2e00;border:2px solid #ffa000}
.high{background-color:#3d1a1a;border:2px solid #d32f2f}
.metric-card{background-color:#1e2130;padding:1rem;border-radius:10px;text-align:center;margin-bottom:0.5rem}
.metric-value{font-size:2rem;font-weight:700}
.metric-label{font-size:0.85rem;color:#aaaaaa}
.recommend-box{background-color:#1a1f35;border-left:4px solid #00d4aa;padding:1rem 1.5rem;border-radius:8px;margin:0.5rem 0}
.side-effect-box{background-color:#1e2130;border:1px solid #333;padding:0.5rem 1rem;border-radius:8px;margin:0.3rem 0}
.section-header{font-size:1.1rem;font-weight:600;color:#00d4aa;margin:1rem 0 0.5rem 0}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">💊 AI-Powered Personalized Drug Safety Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Cloud-based ML system for real-time adverse drug reaction risk prediction</p>', unsafe_allow_html=True)
st.markdown("---")

DRUG_DATA = {
    # PAINKILLERS & FEVER
    "Paracetamol (Acetaminophen)": {"class":"Painkiller","common_side_effects":["Nausea","Stomach pain","Loss of appetite"],"serious_side_effects":["Liver damage","Kidney damage","Severe allergic reaction"],"risk_base":1},
    "Ibuprofen": {"class":"Painkiller","common_side_effects":["Stomach upset","Heartburn","Dizziness","Headache"],"serious_side_effects":["Stomach bleeding","Kidney problems","Heart attack risk"],"risk_base":2},
    "Aspirin": {"class":"Painkiller","common_side_effects":["Stomach irritation","Nausea","Heartburn"],"serious_side_effects":["Internal bleeding","Reye's syndrome in children","Stroke risk"],"risk_base":2},
    "Diclofenac": {"class":"Painkiller","common_side_effects":["Stomach pain","Nausea","Headache","Dizziness"],"serious_side_effects":["Stomach ulcers","Heart attack","Kidney failure"],"risk_base":2},
    "Tramadol": {"class":"Painkiller","common_side_effects":["Nausea","Dizziness","Constipation","Headache"],"serious_side_effects":["Seizures","Addiction risk","Respiratory depression"],"risk_base":3},
    "Naproxen": {"class":"Painkiller","common_side_effects":["Stomach upset","Heartburn","Drowsiness"],"serious_side_effects":["GI bleeding","Kidney damage","Heart problems"],"risk_base":2},
    "Codeine": {"class":"Painkiller","common_side_effects":["Constipation","Drowsiness","Nausea"],"serious_side_effects":["Addiction","Breathing problems","Liver damage with Paracetamol"],"risk_base":3},
    "Mefenamic Acid": {"class":"Painkiller","common_side_effects":["Stomach upset","Diarrhea","Dizziness"],"serious_side_effects":["Kidney failure","Stomach bleeding","Seizures"],"risk_base":2},

    # ANTIBIOTICS
    "Amoxicillin": {"class":"Antibiotic","common_side_effects":["Diarrhea","Stomach upset","Skin rash"],"serious_side_effects":["Severe allergic reaction","Liver problems","Colitis"],"risk_base":1},
    "Ciprofloxacin": {"class":"Antibiotic","common_side_effects":["Nausea","Diarrhea","Headache","Dizziness"],"serious_side_effects":["Tendon rupture","Nerve damage","Heart rhythm problems"],"risk_base":3},
    "Azithromycin": {"class":"Antibiotic","common_side_effects":["Nausea","Diarrhea","Stomach pain"],"serious_side_effects":["Heart rhythm problems","Liver damage","Severe allergic reaction"],"risk_base":2},
    "Doxycycline": {"class":"Antibiotic","common_side_effects":["Nausea","Sun sensitivity","Stomach upset"],"serious_side_effects":["Esophageal damage","Liver toxicity","Intracranial pressure"],"risk_base":2},
    "Metronidazole": {"class":"Antibiotic","common_side_effects":["Nausea","Metallic taste","Headache"],"serious_side_effects":["Nerve damage","Seizures","Severe skin reactions"],"risk_base":2},
    "Clindamycin": {"class":"Antibiotic","common_side_effects":["Diarrhea","Nausea","Stomach pain"],"serious_side_effects":["Severe colitis","Allergic reactions","Liver problems"],"risk_base":2},
    "Cephalexin": {"class":"Antibiotic","common_side_effects":["Diarrhea","Nausea","Stomach upset"],"serious_side_effects":["Severe allergic reaction","Kidney problems","Colitis"],"risk_base":1},
    "Erythromycin": {"class":"Antibiotic","common_side_effects":["Nausea","Stomach cramps","Diarrhea"],"serious_side_effects":["Heart rhythm problems","Liver damage","Hearing loss"],"risk_base":2},
    "Trimethoprim": {"class":"Antibiotic","common_side_effects":["Nausea","Rash","Headache"],"serious_side_effects":["Kidney problems","Blood disorders","Severe skin reactions"],"risk_base":2},
    "Nitrofurantoin": {"class":"Antibiotic","common_side_effects":["Nausea","Headache","Urine discoloration"],"serious_side_effects":["Lung toxicity","Liver damage","Nerve damage"],"risk_base":2},
    "Ampicillin": {"class":"Antibiotic","common_side_effects":["Diarrhea","Rash","Nausea"],"serious_side_effects":["Severe allergic reaction","Colitis","Seizures"],"risk_base":1},
    "Levofloxacin": {"class":"Antibiotic","common_side_effects":["Nausea","Diarrhea","Headache"],"serious_side_effects":["Tendon rupture","Heart problems","Mental health effects"],"risk_base":3},

    # COLD, COUGH & ALLERGY
    "Cetirizine": {"class":"Antiallergy","common_side_effects":["Drowsiness","Dry mouth","Headache"],"serious_side_effects":["Severe allergic reaction","Fast heartbeat","Tremors"],"risk_base":1},
    "Loratadine": {"class":"Antiallergy","common_side_effects":["Headache","Dry mouth","Fatigue"],"serious_side_effects":["Fast heartbeat","Liver problems","Severe allergic reaction"],"risk_base":1},
    "Chlorpheniramine": {"class":"Antiallergy","common_side_effects":["Drowsiness","Dry mouth","Dizziness"],"serious_side_effects":["Urinary retention","Confusion in elderly","Vision problems"],"risk_base":1},
    "Phenylephrine": {"class":"Antiallergy","common_side_effects":["Headache","Nausea","Increased BP"],"serious_side_effects":["Severe hypertension","Heart attack","Stroke"],"risk_base":2},
    "Dextromethorphan": {"class":"Antiallergy","common_side_effects":["Drowsiness","Dizziness","Nausea"],"serious_side_effects":["Serotonin syndrome","Hallucinations","Dependency"],"risk_base":2},
    "Bromhexine": {"class":"Antiallergy","common_side_effects":["Nausea","Diarrhea","Dizziness"],"serious_side_effects":["Severe skin reactions","Liver problems"],"risk_base":1},
    "Salbutamol (Albuterol)": {"class":"Antiallergy","common_side_effects":["Tremors","Headache","Fast heartbeat"],"serious_side_effects":["Severe chest pain","Irregular heartbeat","Low potassium"],"risk_base":2},

    # STOMACH & DIGESTION
    "Omeprazole": {"class":"Gastrointestinal","common_side_effects":["Headache","Nausea","Diarrhea"],"serious_side_effects":["Kidney disease","Low magnesium","Bone fractures"],"risk_base":1},
    "Pantoprazole": {"class":"Gastrointestinal","common_side_effects":["Headache","Diarrhea","Nausea"],"serious_side_effects":["Kidney inflammation","Low magnesium","C. diff infection"],"risk_base":1},
    "Ondansetron": {"class":"Gastrointestinal","common_side_effects":["Headache","Constipation","Fatigue"],"serious_side_effects":["Heart rhythm problems","Serotonin syndrome","Severe allergic reaction"],"risk_base":2},
    "Domperidone": {"class":"Gastrointestinal","common_side_effects":["Dry mouth","Headache","Diarrhea"],"serious_side_effects":["Heart rhythm problems","Sudden cardiac death","Hormonal effects"],"risk_base":2},
    "Ranitidine": {"class":"Gastrointestinal","common_side_effects":["Headache","Diarrhea","Nausea"],"serious_side_effects":["Liver problems","Blood disorders","Kidney problems"],"risk_base":1},
    "Loperamide": {"class":"Gastrointestinal","common_side_effects":["Constipation","Dizziness","Nausea"],"serious_side_effects":["Heart rhythm problems","Toxic megacolon","Ileus"],"risk_base":2},
    "Lactulose": {"class":"Gastrointestinal","common_side_effects":["Bloating","Diarrhea","Stomach cramps"],"serious_side_effects":["Severe electrolyte imbalance","Dehydration"],"risk_base":1},

    # HEART & BLOOD PRESSURE
    "Amlodipine": {"class":"Antihypertensive","common_side_effects":["Swollen ankles","Flushing","Headache"],"serious_side_effects":["Severe low BP","Chest pain","Heart failure worsening"],"risk_base":2},
    "Atorvastatin": {"class":"Antihypertensive","common_side_effects":["Muscle pain","Joint pain","Diarrhea"],"serious_side_effects":["Severe muscle breakdown","Liver damage","Memory problems"],"risk_base":2},
    "Lisinopril": {"class":"Antihypertensive","common_side_effects":["Dry cough","Dizziness","Headache"],"serious_side_effects":["Angioedema","Kidney failure","High potassium"],"risk_base":2},
    "Ramipril": {"class":"Antihypertensive","common_side_effects":["Cough","Dizziness","Fatigue"],"serious_side_effects":["Angioedema","Kidney problems","Low BP"],"risk_base":2},
    "Metoprolol": {"class":"Antihypertensive","common_side_effects":["Fatigue","Dizziness","Cold hands"],"serious_side_effects":["Severe low heart rate","Heart failure","Depression"],"risk_base":2},
    "Warfarin": {"class":"Antihypertensive","common_side_effects":["Easy bruising","Bleeding gums","Fatigue"],"serious_side_effects":["Severe internal bleeding","Brain hemorrhage","Stroke"],"risk_base":3},

    # DIABETES
    "Metformin": {"class":"Antidiabetic","common_side_effects":["Nausea","Diarrhea","Stomach pain"],"serious_side_effects":["Lactic acidosis","Vitamin B12 deficiency","Kidney stress"],"risk_base":2},
    "Glibenclamide": {"class":"Antidiabetic","common_side_effects":["Low blood sugar","Nausea","Weight gain"],"serious_side_effects":["Severe hypoglycemia","Liver damage","Blood disorders"],"risk_base":3},
    "Insulin (Regular)": {"class":"Antidiabetic","common_side_effects":["Low blood sugar","Injection site pain","Weight gain"],"serious_side_effects":["Severe hypoglycemia","Hypokalemia","Lipodystrophy"],"risk_base":3},
    "Sitagliptin": {"class":"Antidiabetic","common_side_effects":["Runny nose","Headache","Stomach pain"],"serious_side_effects":["Pancreatitis","Kidney problems","Severe joint pain"],"risk_base":2},

    # MENTAL HEALTH & SLEEP
    "Sertraline": {"class":"Antidepressant","common_side_effects":["Nausea","Insomnia","Dizziness"],"serious_side_effects":["Suicidal thoughts in youth","Serotonin syndrome","Bleeding risk"],"risk_base":2},
    "Diazepam": {"class":"Antidepressant","common_side_effects":["Drowsiness","Dizziness","Fatigue"],"serious_side_effects":["Addiction","Respiratory depression","Memory impairment"],"risk_base":3},
    "Alprazolam": {"class":"Antidepressant","common_side_effects":["Drowsiness","Dizziness","Memory issues"],"serious_side_effects":["Severe addiction","Withdrawal seizures","Respiratory depression"],"risk_base":3},
    "Melatonin": {"class":"Antidepressant","common_side_effects":["Drowsiness","Headache","Dizziness"],"serious_side_effects":["Hormonal effects","Depression worsening","Vivid dreams"],"risk_base":1},
    "Fluoxetine": {"class":"Antidepressant","common_side_effects":["Nausea","Headache","Insomnia"],"serious_side_effects":["Serotonin syndrome","Suicidal thoughts","Bleeding risk"],"risk_base":2},

    # VITAMINS & SUPPLEMENTS
    "Vitamin C (Ascorbic Acid)": {"class":"Supplement","common_side_effects":["Stomach upset","Diarrhea","Nausea"],"serious_side_effects":["Kidney stones (high dose)","Iron overload","Digestive problems"],"risk_base":1},
    "Vitamin D3": {"class":"Supplement","common_side_effects":["Nausea","Constipation","Fatigue"],"serious_side_effects":["Calcium toxicity","Kidney damage (high dose)","Heart rhythm problems"],"risk_base":1},
    "Iron Supplement": {"class":"Supplement","common_side_effects":["Constipation","Stomach pain","Dark stools"],"serious_side_effects":["Iron toxicity","Liver damage","Stomach bleeding"],"risk_base":1},
}

DRUG_CLASS_MAP = {
    "Painkiller":0,"Antibiotic":1,"Antiallergy":2,
    "Gastrointestinal":3,"Antihypertensive":4,
    "Antidiabetic":5,"Antidepressant":6,"Supplement":7
}

CATEGORIES = {
    "💊 Painkillers & Fever": ["Paracetamol (Acetaminophen)","Ibuprofen","Aspirin","Diclofenac","Tramadol","Naproxen","Codeine","Mefenamic Acid"],
    "🦠 Antibiotics": ["Amoxicillin","Ciprofloxacin","Azithromycin","Doxycycline","Metronidazole","Clindamycin","Cephalexin","Erythromycin","Trimethoprim","Nitrofurantoin","Ampicillin","Levofloxacin"],
    "🤧 Cold, Cough & Allergy": ["Cetirizine","Loratadine","Chlorpheniramine","Phenylephrine","Dextromethorphan","Bromhexine","Salbutamol (Albuterol)"],
    "🫃 Stomach & Digestion": ["Omeprazole","Pantoprazole","Ondansetron","Domperidone","Ranitidine","Loperamide","Lactulose"],
    "❤️ Heart & Blood Pressure": ["Amlodipine","Atorvastatin","Lisinopril","Ramipril","Metoprolol","Warfarin"],
    "🩸 Diabetes": ["Metformin","Glibenclamide","Insulin (Regular)","Sitagliptin"],
    "🧠 Mental Health & Sleep": ["Sertraline","Diazepam","Alprazolam","Melatonin","Fluoxetine"],
    "🌿 Vitamins & Supplements": ["Vitamin C (Ascorbic Acid)","Vitamin D3","Iron Supplement"],
}

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 800
    data = {
        "drug_class":        np.random.choice(list(range(8)), n),
        "drug_risk_base":    np.random.choice([1,2,3], n),
        "num_ingredients":   np.random.randint(1, 6, n),
        "patient_age_group": np.random.choice([0,1,2], n),
        "has_kidney_issue":  np.random.choice([0,1], n, p=[0.8,0.2]),
        "has_liver_issue":   np.random.choice([0,1], n, p=[0.85,0.15]),
        "has_heart_issue":   np.random.choice([0,1], n, p=[0.85,0.15]),
        "has_diabetes":      np.random.choice([0,1], n, p=[0.8,0.2]),
        "is_pregnant":       np.random.choice([0,1], n, p=[0.9,0.1]),
        "dosage_level":      np.random.choice([1,2,3], n),
        "drug_interactions": np.random.randint(0, 6, n),
        "allergy_history":   np.random.choice([0,1], n, p=[0.85,0.15]),
    }
    df = pd.DataFrame(data)
    risk = (
        df["drug_risk_base"]
        + (df["dosage_level"]==3).astype(int)*2
        + df["has_kidney_issue"]*2
        + df["has_liver_issue"]*2
        + df["has_heart_issue"]
        + df["has_diabetes"]
        + df["is_pregnant"]*2
        + (df["drug_interactions"]>3).astype(int)*2
        + df["allergy_history"]*2
        + (df["patient_age_group"]==2).astype(int)
    )
    df["risk_label"] = pd.cut(risk, bins=[-1,4,8,20], labels=[0,1,2]).astype(int)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("risk_label", axis=1)
    y = df["risk_label"]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    pickle.dump(model, open("model.pkl","wb"))
    return model, X.columns.tolist(), acc

df = load_data()
model, feature_cols, accuracy = train_model(df)

with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-value' style='color:#00d4aa'>{accuracy*100:.1f}%</div>
    <div class='metric-label'>Model Accuracy</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
**Algorithm:** Random Forest Classifier  
**Training Samples:** 640  
**Features:** 12 clinical parameters  
**Total Drugs:** 50  
**Environment:** Cloud-based (Google Colab)  
**Deployment:** Streamlit Web App  
    """)
    st.markdown("---")
    st.markdown("### 📋 Drug Categories")
    for cat in CATEGORIES.keys():
        st.markdown(f"{cat} — {len(CATEGORIES[cat])} drugs")
    st.markdown("---")
    st.caption("AI-Powered Personalized Drug Safety Assistant | International Science Project 2025")

st.subheader("🔬 Patient & Drug Assessment Form")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-header">💊 Drug Information</p>', unsafe_allow_html=True)
    category  = st.selectbox("Drug Category", list(CATEGORIES.keys()))
    drug_name = st.selectbox("Select Drug Name", CATEGORIES[category])
    drug_info = DRUG_DATA[drug_name]
    dosage_level = st.selectbox("Dosage Level", [1,2,3],
        format_func=lambda x:{1:"🟢 Low (Recommended)",2:"🟡 Medium (Standard)",3:"🔴 High (Above normal)"}[x])
    num_ing      = st.slider("Number of Active Ingredients", 1, 5, 2)
    interactions = st.slider("Other Drugs Taken Together", 0, 5, 1)
    st.info(f"**Drug Class:** {drug_info['class']}  |  **Base Risk:** {'⭐' * drug_info['risk_base']}")

with col2:
    st.markdown('<p class="section-header">👤 Patient Medical History</p>', unsafe_allow_html=True)
    age_group = st.selectbox("Patient Age Group", [0,1,2],
        format_func=lambda x:{0:"👦 Pediatric (below 18)",1:"🧑 Adult (18–60)",2:"👴 Geriatric (above 60)"}[x])
    st.markdown("**Pre-existing Medical Conditions:**")
    c1,c2 = st.columns(2)
    with c1:
        kidney  = st.checkbox("🫘 Kidney Disease")
        liver   = st.checkbox("🫀 Liver Disease")
        heart   = st.checkbox("❤️ Heart Disease")
    with c2:
        diabetes = st.checkbox("🩸 Diabetes")
        pregnant = st.checkbox("🤰 Pregnant")
        allergy  = st.checkbox("⚠️ Drug Allergy History")

with st.expander(f"📋 View Known Side Effects of {drug_name}"):
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Common Side Effects:**")
        for s in drug_info["common_side_effects"]:
            st.markdown(f"<div class='side-effect-box'>🟡 {s}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Serious Side Effects:**")
        for s in drug_info["serious_side_effects"]:
            st.markdown(f"<div class='side-effect-box'>🔴 {s}</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Analyze & Predict Side Effect Risk", use_container_width=True):
    row = {
        "drug_class":        DRUG_CLASS_MAP[drug_info["class"]],
        "drug_risk_base":    drug_info["risk_base"],
        "num_ingredients":   num_ing,
        "patient_age_group": age_group,
        "has_kidney_issue":  int(kidney),
        "has_liver_issue":   int(liver),
        "has_heart_issue":   int(heart),
        "has_diabetes":      int(diabetes),
        "is_pregnant":       int(pregnant),
        "dosage_level":      dosage_level,
        "drug_interactions": interactions,
        "allergy_history":   int(allergy),
    }
    inp = pd.DataFrame([row])
    for c in feature_cols:
        if c not in inp.columns: inp[c] = 0
    inp = inp[feature_cols]

    pred  = model.predict(inp)[0]
    proba = model.predict_proba(inp)[0]
    risk_pct = round(max(proba)*100, 1)

    st.markdown("---")
    st.subheader("📋 AI Risk Assessment Results")

    m1,m2,m3 = st.columns(3)
    risk_names  = {0:"LOW",1:"MEDIUM",2:"HIGH"}
    risk_colors = {0:"#00c853",1:"#ffa000",2:"#d32f2f"}

    with m1:
        st.markdown(f"""<div class='metric-card'>
        <div class='metric-value' style='color:{risk_colors[pred]}'>{risk_names[pred]}</div>
        <div class='metric-label'>Risk Level</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class='metric-card'>
        <div class='metric-value' style='color:#00d4aa'>{risk_pct}%</div>
        <div class='metric-label'>Confidence Score</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class='metric-card'>
        <div class='metric-value' style='color:#7c83fd'>{accuracy*100:.1f}%</div>
        <div class='metric-label'>Model Accuracy</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    css_class = {0:"low",1:"med",2:"high"}[pred]
    emoji     = {0:"🟢",1:"🟡",2:"🔴"}[pred]
    message   = {
        0:f"{drug_name} appears relatively safe for this patient profile.",
        1:f"Moderate risk detected for {drug_name}. Careful monitoring advised.",
        2:f"High risk of serious side effects with {drug_name}. Immediate medical review required."
    }[pred]
    st.markdown(f"""<div class='risk-box {css_class}'>
    <h2>{emoji} {risk_names[pred]} RISK</h2>
    <p style='color:#cccccc'>{message}</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader(f"⚠️ Side Effects to Watch for with {drug_name}")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Likely Common Effects:**")
        for s in drug_info["common_side_effects"]:
            st.markdown(f"<div class='side-effect-box'>🟡 {s}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("**Serious Effects to Monitor:**")
        color = {0:"#00c853",1:"#ffa000",2:"#d32f2f"}[pred]
        icon  = {0:"🔵",1:"⚠️",2:"🚨"}[pred]
        for s in drug_info["serious_side_effects"]:
            st.markdown(f"<div class='side-effect-box' style='border-left:3px solid {color}'>{icon} {s}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💡 AI-Generated Smart Recommendations")
    recs = {
        0:["✅ Standard dosage can be continued as prescribed.","✅ Routine follow-up every 3 months is sufficient.","✅ No immediate lifestyle changes required.","✅ Take medication with food to minimize mild effects."],
        1:["⚠️ Consult your physician before continuing this medication.","⚠️ Monitor kidney and liver function every 4–6 weeks.","⚠️ Avoid adding new medications without medical advice.","⚠️ Report any unusual symptoms immediately to your doctor."],
        2:["🚨 Immediate specialist consultation is strongly advised.","🚨 Consider dosage reduction or switching to a safer alternative.","🚨 Do NOT combine with additional drugs without supervision.","🚨 Emergency blood tests and organ function tests recommended.","🚨 Hospitalization may be required if symptoms worsen."]
    }
    for r in recs[pred]:
        st.markdown(f"<div class='recommend-box'>{r}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("📊 Risk Probability")
        fig, ax = plt.subplots(figsize=(5,3))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        bars = ax.barh(["Low","Medium","High"], proba, color=["#00c853","#ffa000","#d32f2f"], height=0.5)
        ax.set_xlim(0,1); ax.set_xlabel("Probability", color="white"); ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_visible(False)
        for bar,p in zip(bars,proba):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f"{p*100:.1f}%", va="center", color="white", fontsize=10)
        st.pyplot(fig)

    with c2:
        st.subheader("🧠 Key Risk Factors")
        feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(6)
        fig2, ax2 = plt.subplots(figsize=(5,3))
        fig2.patch.set_facecolor('#1e2130'); ax2.set_facecolor('#1e2130')
        feat_imp.plot(kind="barh", ax=ax2, color="#00d4aa")
        ax2.set_xlabel("Importance", color="white"); ax2.tick_params(colors="white"); ax2.invert_yaxis()
        for spine in ax2.spines.values(): spine.set_visible(False)
        st.pyplot(fig2)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Risk Trend Across Dosage Levels")
    dosage_risks = []
    for d in [1,2,3]:
        test_row = row.copy(); test_row["dosage_level"] = d
        test_inp = pd.DataFrame([test_row])[feature_cols]
        p = model.predict_proba(test_inp)[0]
        dosage_risks.append(p[2]*100)
    fig3, ax3 = plt.subplots(figsize=(8,3))
    fig3.patch.set_facecolor('#1e2130'); ax3.set_facecolor('#1e2130')
    ax3.plot(["Low Dose","Medium Dose","High Dose"], dosage_risks, color="#d32f2f", marker="o", linewidth=2, markersize=8)
    ax3.fill_between(["Low Dose","Medium Dose","High Dose"], dosage_risks, alpha=0.2, color="#d32f2f")
    ax3.set_ylabel("High Risk %", color="white"); ax3.set_title(f"Risk Trend — {drug_name}", color="white")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values(): spine.set_visible(False)
    for x,y in zip(["Low Dose","Medium Dose","High Dose"], dosage_risks):
        ax3.annotate(f"{y:.1f}%",(x,y), textcoords="offset points", xytext=(0,10), ha="center", color="white", fontsize=10)
    st.pyplot(fig3)

    st.success("✅ Assessment complete. This AI tool is for research and educational purposes only. Always consult a certified medical professional.")
