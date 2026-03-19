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
.subtitle{font-size:1rem;color:#aaaaaa;text-align:center;margin-bottom:2rem}
.risk-box{padding:1.5rem;border-radius:12px;text-align:center;margin:1rem 0}
.low{background-color:#1a3d2b;border:2px solid #00c853}
.med{background-color:#3d2e00;border:2px solid #ffa000}
.high{background-color:#3d1a1a;border:2px solid #d32f2f}
.metric-card{background-color:#1e2130;padding:1rem;border-radius:10px;text-align:center}
.metric-value{font-size:2rem;font-weight:700}
.metric-label{font-size:0.85rem;color:#aaaaaa}
.recommend-box{background-color:#1a1f35;border-left:4px solid #00d4aa;padding:1rem 1.5rem;border-radius:8px;margin:0.5rem 0}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">💊 AI-Powered Personalized Drug Safety Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Cloud-based Machine Learning system for real-time adverse drug reaction risk prediction</p>', unsafe_allow_html=True)
st.markdown("---")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500
    drug_classes = ["Antibiotic","Painkiller","Antidepressant","Antihypertensive","Antidiabetic"]
    data = {
        "drug_class":        np.random.choice(drug_classes, n),
        "num_ingredients":   np.random.randint(1, 6, n),
        "patient_age_group": np.random.choice([0,1,2], n),
        "has_kidney_issue":  np.random.choice([0,1], n, p=[0.8,0.2]),
        "has_liver_issue":   np.random.choice([0,1], n, p=[0.85,0.15]),
        "dosage_level":      np.random.choice([1,2,3], n),
        "drug_interactions": np.random.randint(0, 5, n),
    }
    df = pd.DataFrame(data)
    risk = (
        (df["dosage_level"]==3).astype(int)*2
        + df["has_kidney_issue"]
        + df["has_liver_issue"]
        + (df["drug_interactions"]>2).astype(int)
        + (df["patient_age_group"]==2).astype(int)
    )
    df["risk_label"] = pd.cut(risk, bins=[-1,1,3,10], labels=[0,1,2]).astype(int)
    df = pd.get_dummies(df, columns=["drug_class"])
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("risk_label", axis=1)
    y = df["risk_label"]
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    pickle.dump(model, open("model.pkl","wb"))
    return model, X.columns.tolist(), acc

df = load_data()
model, feature_cols, accuracy = train_model(df)

with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value' style='color:#00d4aa'>{accuracy*100:.1f}%</div>
        <div class='metric-label'>Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
**Algorithm:** Random Forest Classifier  
**Training Samples:** 400  
**Features:** 11 parameters  
**Environment:** Cloud-based (Google Colab)  
**Deployment:** Streamlit Web App  
    """)
    st.markdown("---")
    st.caption("AI-Powered Personalized Drug Safety Assistant | International Science Project 2025")

st.subheader("🔬 Enter Patient & Drug Details")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Drug Information**")
    drug_class   = st.selectbox("Drug Category", ["Antibiotic","Painkiller","Antidepressant","Antihypertensive","Antidiabetic"])
    dosage_level = st.selectbox("Dosage Level", [1,2,3], format_func=lambda x:{1:"🟢 Low",2:"🟡 Medium",3:"🔴 High"}[x])
    num_ing      = st.slider("Number of Active Ingredients", 1, 5, 2)
    interactions = st.slider("Other Drugs Taken Simultaneously", 0, 4, 1)

with col2:
    st.markdown("**Patient Information**")
    age_group = st.selectbox("Age Group", [0,1,2], format_func=lambda x:{0:"👦 Young (below 18)",1:"🧑 Adult (18-60)",2:"👴 Elderly (above 60)"}[x])
    kidney = st.checkbox("🫘 Has Kidney Issues")
    liver  = st.checkbox("🫀 Has Liver Issues")
    st.info("💡 Tip: Elderly patients with organ issues and high dosage are highest risk.")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Side Effect Risk", use_container_width=True):
    row = {
        "num_ingredients":   num_ing,
        "patient_age_group": age_group,
        "has_kidney_issue":  int(kidney),
        "has_liver_issue":   int(liver),
        "dosage_level":      dosage_level,
        "drug_interactions": interactions,
    }
    for dc in ["Antibiotic","Painkiller","Antidepressant","Antihypertensive","Antidiabetic"]:
        row[f"drug_class_{dc}"] = 1 if drug_class==dc else 0
    inp = pd.DataFrame([row])
    for c in feature_cols:
        if c not in inp.columns: inp[c] = 0
    inp = inp[feature_cols]

    pred  = model.predict(inp)[0]
    proba = model.predict_proba(inp)[0]
    risk_pct = round(max(proba)*100, 1)

    st.markdown("---")
    st.subheader("📋 Prediction Results")

    m1,m2,m3 = st.columns(3)
    risk_names  = {0:"LOW", 1:"MEDIUM", 2:"HIGH"}
    risk_colors = {0:"#00c853", 1:"#ffa000", 2:"#d32f2f"}

    with m1:
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{risk_colors[pred]}'>{risk_names[pred]}</div><div class='metric-label'>Risk Level</div></div>", unsafe_allow_html=True)
    with m2:
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#00d4aa'>{risk_pct}%</div><div class='metric-label'>Confidence Score</div></div>", unsafe_allow_html=True)
    with m3:
        st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:#7c83fd'>{accuracy*100:.1f}%</div><div class='metric-label'>Model Accuracy</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    css_class = {0:"low",1:"med",2:"high"}[pred]
    emoji     = {0:"🟢",1:"🟡",2:"🔴"}[pred]
    message   = {0:"This drug combination appears relatively safe for the patient.",
                 1:"Moderate side effect risk detected. Careful monitoring advised.",
                 2:"High risk of serious side effects detected. Immediate review required."}[pred]
    st.markdown(f"<div class='risk-box {css_class}'><h2>{emoji} {risk_names[pred]} RISK</h2><p style='color:#cccccc'>{message}</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("💡 AI-Generated Smart Recommendations")
    recs = {
        0:["✅ Standard dosage can be continued as prescribed.",
           "✅ Routine monitoring every 3 months is sufficient.",
           "✅ No immediate lifestyle changes required."],
        1:["⚠️ Consult your physician before continuing this dosage.",
           "⚠️ Monitor kidney and liver function every 4–6 weeks.",
           "⚠️ Avoid adding new medications without medical advice.",
           "⚠️ Stay well hydrated and reduce alcohol consumption."],
        2:["🚨 Immediate consultation with a specialist is strongly advised.",
           "🚨 Consider dosage reduction or alternative medication.",
           "🚨 Do NOT combine with additional drugs without supervision.",
           "🚨 Emergency blood and organ function tests recommended."]
    }
    for r in recs[pred]:
        st.markdown(f"<div class='recommend-box'>{r}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Risk Probability Distribution")
    fig, ax = plt.subplots(figsize=(7,3))
    fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
    bars = ax.barh(["Low Risk","Medium Risk","High Risk"], proba, color=["#00c853","#ffa000","#d32f2f"], height=0.5)
    ax.set_xlim(0,1); ax.set_xlabel("Probability", color="white"); ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_visible(False)
    for bar,p in zip(bars,proba):
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2, f"{p*100:.1f}%", va="center", color="white", fontsize=11)
    st.pyplot(fig)

    st.subheader("🧠 Key Factors Influencing This Prediction")
    feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(5)
    fig2, ax2 = plt.subplots(figsize=(7,3))
    fig2.patch.set_facecolor('#1e2130'); ax2.set_facecolor('#1e2130')
    feat_imp.plot(kind="barh", ax=ax2, color="#00d4aa")
    ax2.set_xlabel("Importance Score", color="white"); ax2.tick_params(colors="white"); ax2.invert_yaxis()
    for spine in ax2.spines.values(): spine.set_visible(False)
    st.pyplot(fig2)

    st.success("✅ Prediction complete. This tool is for research and educational purposes. Always consult a certified medical professional.")

