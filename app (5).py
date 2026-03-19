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
    q = question.lower()

    responses_en = {
        "paracetamol": "Paracetamol is generally safe for most people including children and elderly when taken at the correct dose. For children, the dose depends on their weight — always use a children's formulation. Avoid taking more than 4 doses in 24 hours. Never combine with alcohol as it can cause serious liver damage. Remember: Always consult a real doctor before taking any medicine.",
        "ibuprofen": "Ibuprofen is effective for pain and fever but should be taken with food to avoid stomach upset. It is NOT recommended for people with kidney problems, stomach ulcers, or heart disease. Avoid in pregnancy especially after 30 weeks. Children under 6 months should not take it. Remember: Always consult a real doctor before taking any medicine.",
        "aspirin": "Aspirin is used for pain relief and also to prevent blood clots. However it should NEVER be given to children under 16 years as it can cause a dangerous condition called Reye's syndrome. People with stomach ulcers or bleeding disorders should avoid it. Remember: Always consult a real doctor before taking any medicine.",
        "antibiotic": "Antibiotics only work against bacterial infections — not viral ones like common cold or flu. Always complete the full course even if you feel better. Never share antibiotics or use leftover ones. Overuse leads to antibiotic resistance which is a global health crisis. Remember: Always consult a real doctor before taking any medicine.",
        "side effect": "Side effects vary by medicine. Common ones include nausea, headache, and dizziness. Serious side effects like allergic reactions (rash, swelling, breathing difficulty) need immediate medical attention. If you experience unusual symptoms after starting a new medicine, contact your doctor immediately. Remember: Always consult a real doctor before taking any medicine.",
        "interaction": "Drug interactions happen when two or more medicines affect each other. This can make one medicine less effective or cause dangerous side effects. Always tell your doctor ALL medicines you are taking including vitamins and herbal supplements. Never combine medicines without medical advice. Remember: Always consult a real doctor before taking any medicine.",
        "kidney": "People with kidney disease need to be very careful with medicines. Many drugs like Ibuprofen, Naproxen, and some antibiotics can worsen kidney function. Paracetamol at recommended doses is generally safer for kidney patients. Always inform your doctor about your kidney condition before taking any medicine. Remember: Always consult a real doctor before taking any medicine.",
        "liver": "The liver processes most medicines in your body. People with liver disease must be extra careful. Paracetamol in high doses can cause serious liver damage even in healthy people. Avoid alcohol with any medicine. Always tell your doctor about any liver condition you have. Remember: Always consult a real doctor before taking any medicine.",
        "pregnant": "Pregnancy requires extra caution with medicines. Many common drugs like Ibuprofen, Aspirin (high dose), and certain antibiotics are NOT safe during pregnancy. Paracetamol is considered relatively safer but should still be used minimally. Always consult your doctor or gynecologist before taking any medicine during pregnancy. Remember: Always consult a real doctor before taking any medicine.",
        "child": "Children need different doses than adults — usually based on their body weight. Never give adult medicines to children without medical advice. Some medicines like Aspirin are completely banned for children. Always use a children's formulation and check the label carefully. Remember: Always consult a real doctor before taking any medicine.",
        "elderly": "Elderly patients are at higher risk of side effects because their kidneys and liver work slower, so medicines stay in the body longer. They are also more likely to take multiple medicines which increases interaction risk. Start with lower doses and monitor carefully. Regular doctor checkups are very important. Remember: Always consult a real doctor before taking any medicine.",
        "overdose": "A medicine overdose is a medical emergency. If you suspect someone has taken too much medicine, call emergency services (108 in India) immediately. Do NOT try to make the person vomit unless told to by a medical professional. Keep the medicine bottle to show doctors. Time is critical — act fast. Remember: Always consult a real doctor before taking any medicine.",
        "allergy": "Drug allergies can range from mild rash to life-threatening anaphylaxis. Symptoms include hives, rash, swelling of face or throat, difficulty breathing, or rapid heartbeat. If you experience these after taking medicine, stop the medicine and seek emergency care immediately. Always inform your doctor about any known drug allergies. Remember: Always consult a real doctor before taking any medicine.",
        "dose": "Taking the correct dose is very important. Too little may not treat your condition. Too much can cause serious side effects or overdose. Always follow your doctor's prescription. Do not adjust your dose on your own. If you miss a dose, take it as soon as you remember unless it is almost time for the next dose. Remember: Always consult a real doctor before taking any medicine.",
        "diabetes": "People with diabetes need to be careful as some medicines can affect blood sugar levels. Certain antibiotics and steroids can raise blood sugar. Always monitor your blood sugar when starting a new medicine. Inform your doctor about your diabetes before any treatment. Remember: Always consult a real doctor before taking any medicine.",
        "heart": "Heart patients must be very careful with medicines. NSAIDs like Ibuprofen and Diclofenac can increase heart attack risk. Always inform your cardiologist about ALL medicines you take. Never stop heart medicines suddenly without medical advice. Remember: Always consult a real doctor before taking any medicine.",
        "fever": "For fever, Paracetamol is the safest first choice for most people. Keep hydrated and rest. If fever is above 103°F (39.4°C), lasts more than 3 days, or is accompanied by severe headache, rash or stiff neck — seek medical attention immediately. Do not use Aspirin for fever in children. Remember: Always consult a real doctor before taking any medicine.",
        "pain": "For mild to moderate pain, Paracetamol is usually the safest choice. For inflammation-related pain, NSAIDs like Ibuprofen may help but must be taken with food. For severe or persistent pain, always consult a doctor as it may indicate a serious condition. Remember: Always consult a real doctor before taking any medicine.",
        "safe": "Medicine safety depends on many factors — your age, weight, other medical conditions, and other medicines you take. What is safe for one person may not be safe for another. Always read the medicine label and consult your doctor or pharmacist before starting any new medicine. Remember: Always consult a real doctor before taking any medicine.",
        "stop": "Never stop a prescribed medicine suddenly without consulting your doctor — especially for heart medicines, blood pressure drugs, antidepressants, and steroids. Stopping suddenly can cause serious withdrawal effects or worsen your condition. If you want to stop a medicine, always discuss with your doctor first. Remember: Always consult a real doctor before taking any medicine.",
    }

    responses_ta = {
        "paracetamol": "பாராசிட்டமால் பெரும்பாலான மக்களுக்கு பாதுகாப்பானது — குழந்தைகள் மற்றும் முதியவர்களுக்கும் சரியான அளவில் எடுத்தால் பாதுகாப்பானது. குழந்தைகளுக்கு அளவு அவர்களின் எடையை பொறுத்து இருக்கும். 24 மணி நேரத்தில் 4 டோஸுக்கு மேல் எடுக்காதீர்கள். மது அருந்துவோர் கல்லீரல் பாதிப்புக்கு ஆளாகலாம். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்.",
        "ibuprofen": "இப்யூப்ரோஃபென் வலி மற்றும் காய்ச்சலுக்கு பயனுள்ளது, ஆனால் வயிற்று எரிச்சலை தவிர்க்க உணவுடன் எடுக்கவும். சிறுநீரக பிரச்சனை, வயிற்று புண் அல்லது இதய நோய் உள்ளவர்கள் தவிர்க்கவும். கர்ப்பகாலத்தில் குறிப்பாக 30 வாரங்களுக்கு பிறகு தவிர்க்கவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்.",
        "side effect": "பக்க விளைவுகள் மருந்துக்கு மருந்து வேறுபடும். குமட்டல், தலைவலி, தலைச்சுற்றல் பொதுவானவை. ஒவ்வாமை அறிகுறிகள் — தோல் தடிப்பு, வீக்கம், சுவாசிப்பு கஷ்டம் — உடனடியாக மருத்துவ உதவி தேவை. புதிய மருந்து எடுத்த பிறகு எந்த அசாதாரண அறிகுறியும் தோன்றினால் மருத்துவரை அணுகவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்.",
        "காய்ச்சல்": "காய்ச்சலுக்கு பாராசிட்டமால் முதல் தேர்வாக பயன்படுத்தலாம். நிறைய தண்ணீர் குடியுங்கள். காய்ச்சல் 39.4°C (103°F) க்கு மேல் இருந்தால் அல்லது 3 நாட்களுக்கு மேல் நீடித்தால் உடனே மருத்துவரை அணுகவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்.",
        "வலி": "லேசான வலிக்கு பாராசிட்டமால் பாதுகாப்பான தேர்வு. அழற்சி சம்பந்தமான வலிக்கு இப்யூப்ரோஃபென் உதவலாம், ஆனால் உணவுடன் எடுக்கவும். கடுமையான அல்லது நீடிக்கும் வலி இருந்தால் மருத்துவரை அணுகவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்.",
        "குழந்தை": "குழந்தைகளுக்கு அளவு அவர்களின் எடையை பொறுத்து இருக்கும். பெரியவர்களின் மருந்தை குழந்தைகளுக்கு கொடுக்காதீர்கள். ஆஸ்பிரின் 16 வயதுக்கு கீழ் குழந்தைகளுக்கு கொடுக்கக் கூடாது. குழந்தைகளுக்கான சிறப்பு மருந்து பயன்படுத்துங்கள். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்.",
        "interaction": "இரண்டு அல்லது அதிக மருந்துகள் ஒன்றாக எடுக்கும்போது ஒன்று மற்றதை பாதிக்கலாம். இது மருந்தை குறைவாக செயல்பட வைக்கலாம் அல்லது ஆபத்தான பக்க விளைவுகளை ஏற்படுத்தலாம். நீங்கள் எடுக்கும் அனைத்து மருந்துகளையும் மருத்துவரிடம் தெரிவிக்கவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்.",
    }

    responses = responses_ta if lang == "தமிழ்" else responses_en

    # Match keywords
    for keyword, reply in responses.items():
        if keyword in q:
            return reply

    # Check if asking about a specific drug
    for drug_name, info in DRUG_DATA.items():
        if drug_name.lower().split("(")[0].strip() in q or drug_name.lower() in q:
            common = ", ".join(info["common"])
            serious = ", ".join(info["serious"])
            if lang == "தமிழ்":
                return f"{drug_name} ({info['class']}) பற்றிய தகவல்:\n\n**பொதுவான பக்க விளைவுகள்:** {common}\n\n**தீவிர பக்க விளைவுகள்:** {serious}\n\n**வழக்கமான அளவு:** {info['usual_dose']}\n\nநினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
            else:
                return f"Here's what I know about **{drug_name}** ({info['class']}):\n\n**Common side effects:** {common}\n\n**Serious side effects to watch:** {serious}\n\n**Usual dose:** {info['usual_dose']}\n\nRemember: Always consult a real doctor before taking any medicine."

    # Default response
    if lang == "தமிழ்":
        return "உங்கள் கேள்விக்கு நன்றி! மருந்து பாதுகாப்பு குறித்த பொதுவான ஆலோசனை: எப்போதும் மருத்துவரின் பரிந்துரைப்படி மருந்து எடுக்கவும், சரியான அளவை கடைப்பிடிக்கவும், பக்க விளைவுகள் தோன்றினால் உடனே மருத்துவரை அணுகவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
    else:
        return "Thank you for your question! For general medicine safety: always follow your doctor's prescription, take the correct dose at the right time, never share medicines, and contact your doctor immediately if you notice any unusual symptoms after taking medicine. Remember: Always consult a real doctor before taking any medicine."

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
