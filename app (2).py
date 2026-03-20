import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle, warnings, re
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Smart Medicine Safety Dashboard", page_icon="💊", layout="wide")

st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
.main-title { font-size:2.4rem; font-weight:800; color:#00a884; text-align:center; margin-bottom:0 }
.subtitle { font-size:1rem; color:#666; text-align:center; margin-bottom:1.5rem }
.card { background:white; border-radius:16px; padding:1.5rem; border:1px solid #eee; margin-bottom:1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06) }
.risk-low { background:#e8f5e9; border:2px solid #00c853; border-radius:16px; padding:1.5rem; text-align:center }
.risk-med { background:#fff8e1; border:2px solid #ffa000; border-radius:16px; padding:1.5rem; text-align:center }
.risk-high { background:#ffebee; border:2px solid #d32f2f; border-radius:16px; padding:1.5rem; text-align:center }
.metric-card { border-radius:12px; padding:1rem; text-align:center; border:1px solid #eee }
.metric-val { font-size:2rem; font-weight:800 }
.metric-lbl { font-size:0.8rem; color:#888 }
.why-box { background:#f3f0ff; border-left:4px solid #7c4dff; padding:1rem 1.5rem; border-radius:8px; margin:0.4rem 0 }
.tip-box { background:#e3f2fd; border-left:4px solid #1976d2; padding:1rem 1.5rem; border-radius:8px; margin:0.4rem 0 }
.suggest-box { background:#fff3e0; border-left:4px solid #ff6f00; padding:1.2rem 1.5rem; border-radius:12px; margin:0.5rem 0; font-size:1.05rem }
.safe-tag { background:#e8f5e9; color:#2e7d32; padding:0.3rem 0.8rem; border-radius:20px; font-weight:600; font-size:0.9rem }
.warn-tag { background:#fff8e1; color:#e65100; padding:0.3rem 0.8rem; border-radius:20px; font-weight:600; font-size:0.9rem }
.danger-tag { background:#ffebee; color:#c62828; padding:0.3rem 0.8rem; border-radius:20px; font-weight:600; font-size:0.9rem }
.chat-user { background:#00a884; color:white; padding:0.75rem 1rem; border-radius:12px 12px 2px 12px; margin:0.5rem 0; max-width:80%; margin-left:auto; text-align:right }
.chat-ai { background:#f0f0f0; color:#222; padding:0.75rem 1rem; border-radius:12px 12px 12px 2px; margin:0.5rem 0; max-width:85% }
.feature-card { padding:1.5rem; border-radius:12px; border:1px solid #ddd; text-align:center }
section[data-testid="stSidebar"] { background:#00a884 }
section[data-testid="stSidebar"] * { color:white !important }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DRUG DATABASE — context aware
# ══════════════════════════════════════════════════════════════════════════════
DRUG_DATA = {
    "Paracetamol":{"class":"Painkiller","side_effects":["Nausea","Stomach pain","Loss of appetite"],"serious":["Liver damage at high dose","Kidney damage","Severe allergic reaction"],"risk_base":"low","drowsy":False,"best_time":"any","food":"after food","usual_dose":"500mg-1000mg","tip":"Safe for most people. Do not exceed 4g per day."},
    "Ibuprofen":{"class":"Painkiller","side_effects":["Stomach upset","Heartburn","Dizziness"],"serious":["Stomach bleeding","Kidney problems","Heart attack risk"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"always with food","usual_dose":"200mg-400mg","tip":"Always take with food. Avoid if you have stomach ulcers."},
    "Aspirin":{"class":"Painkiller","side_effects":["Stomach irritation","Nausea","Heartburn"],"serious":["Internal bleeding","Stroke risk"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"300mg-600mg","tip":"Never give to children under 16. Take with food."},
    "Diclofenac":{"class":"Painkiller","side_effects":["Stomach pain","Nausea","Headache"],"serious":["Stomach ulcers","Heart attack","Kidney failure"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"50mg-75mg","tip":"Take with food. Avoid in heart disease patients."},
    "Tramadol":{"class":"Painkiller","side_effects":["Nausea","Dizziness","Constipation","Drowsiness"],"serious":["Seizures","Addiction","Breathing problems"],"risk_base":"high","drowsy":True,"best_time":"night","food":"with or without food","usual_dose":"50mg-100mg","tip":"Very drowsy medicine. Never drive after taking. Take at night only."},
    "Mefenamic Acid":{"class":"Painkiller","side_effects":["Stomach upset","Diarrhea","Dizziness"],"serious":["Kidney failure","Stomach bleeding","Seizures"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"250mg-500mg","tip":"Best for period pain. Always take with food."},
    "Codeine":{"class":"Painkiller","side_effects":["Constipation","Drowsiness","Nausea"],"serious":["Addiction","Breathing problems"],"risk_base":"high","drowsy":True,"best_time":"night","food":"with food","usual_dose":"15mg-60mg","tip":"Highly drowsy. Never drive. Take only at night."},
    "Naproxen":{"class":"Painkiller","side_effects":["Stomach upset","Heartburn","Drowsiness"],"serious":["GI bleeding","Kidney damage","Heart problems"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"250mg-500mg","tip":"Take with food. Avoid long-term use."},
    "Amoxicillin":{"class":"Antibiotic","side_effects":["Diarrhea","Stomach upset","Skin rash"],"serious":["Severe allergic reaction","Liver problems"],"risk_base":"low","drowsy":False,"best_time":"any","food":"with or without food","usual_dose":"250mg-500mg","tip":"Complete the full course even if you feel better."},
    "Ciprofloxacin":{"class":"Antibiotic","side_effects":["Nausea","Diarrhea","Headache","Dizziness"],"serious":["Tendon rupture","Nerve damage","Heart rhythm problems"],"risk_base":"high","drowsy":False,"best_time":"morning","food":"with or without food","usual_dose":"250mg-500mg","tip":"Avoid sunlight. Do not take with dairy or antacids."},
    "Azithromycin":{"class":"Antibiotic","side_effects":["Nausea","Diarrhea","Stomach pain"],"serious":["Heart rhythm problems","Liver damage"],"risk_base":"medium","drowsy":False,"best_time":"any","food":"with or without food","usual_dose":"250mg-500mg","tip":"Complete full 3-5 day course. Take at same time daily."},
    "Doxycycline":{"class":"Antibiotic","side_effects":["Nausea","Sun sensitivity","Stomach upset"],"serious":["Esophageal damage","Liver toxicity"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"100mg","tip":"Avoid direct sunlight. Take with plenty of water. Do not lie down for 30 mins after."},
    "Metronidazole":{"class":"Antibiotic","side_effects":["Nausea","Metallic taste","Headache"],"serious":["Nerve damage","Seizures"],"risk_base":"medium","drowsy":False,"best_time":"any","food":"with food","usual_dose":"200mg-400mg","tip":"Never mix with alcohol — causes violent reaction."},
    "Cetirizine":{"class":"Antiallergy","side_effects":["Drowsiness","Dry mouth","Headache"],"serious":["Severe allergic reaction","Fast heartbeat"],"risk_base":"low","drowsy":True,"best_time":"night","food":"with or without food","usual_dose":"10mg","tip":"Very drowsy. Best taken at bedtime. Avoid driving."},
    "Loratadine":{"class":"Antiallergy","side_effects":["Headache","Dry mouth","Fatigue"],"serious":["Fast heartbeat","Liver problems"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"with or without food","usual_dose":"10mg","tip":"Less drowsy than other antihistamines. Safe for daytime use."},
    "Chlorpheniramine":{"class":"Antiallergy","side_effects":["Drowsiness","Dry mouth","Dizziness"],"serious":["Urinary retention","Vision problems"],"risk_base":"low","drowsy":True,"best_time":"night","food":"with or without food","usual_dose":"4mg","tip":"Very drowsy. Take only at night. Avoid driving."},
    "Salbutamol":{"class":"Antiallergy","side_effects":["Tremors","Headache","Fast heartbeat"],"serious":["Severe chest pain","Irregular heartbeat"],"risk_base":"medium","drowsy":False,"best_time":"any","food":"not applicable — inhaler","usual_dose":"2mg-4mg or 1-2 puffs","tip":"Shake inhaler before use. Rinse mouth after steroid inhalers."},
    "Fexofenadine":{"class":"Antiallergy","side_effects":["Headache","Nausea","Dizziness"],"serious":["Severe allergic reaction"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"avoid with fruit juice","usual_dose":"120mg-180mg","tip":"Non-drowsy antihistamine. Safe for daytime. Avoid fruit juice."},
    "Omeprazole":{"class":"Gastrointestinal","side_effects":["Headache","Nausea","Diarrhea"],"serious":["Kidney disease","Low magnesium","Bone fractures"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"30 mins before food","usual_dose":"20mg-40mg","tip":"Take 30 minutes before breakfast for best effect."},
    "Pantoprazole":{"class":"Gastrointestinal","side_effects":["Headache","Diarrhea","Nausea"],"serious":["Kidney inflammation","Low magnesium"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"30 mins before food","usual_dose":"40mg","tip":"Take before breakfast. Do not crush or chew tablet."},
    "Ondansetron":{"class":"Gastrointestinal","side_effects":["Headache","Constipation","Fatigue"],"serious":["Heart rhythm problems","Serotonin syndrome"],"risk_base":"medium","drowsy":False,"best_time":"any","food":"with or without food","usual_dose":"4mg-8mg","tip":"Dissolves under tongue for fast relief. Best for vomiting."},
    "Domperidone":{"class":"Gastrointestinal","side_effects":["Dry mouth","Headache","Diarrhea"],"serious":["Heart rhythm problems","Sudden cardiac death"],"risk_base":"medium","drowsy":False,"best_time":"any","food":"30 mins before food","usual_dose":"10mg","tip":"Take 30 minutes before meals. Do not exceed recommended dose."},
    "Omeprazole":{"class":"Gastrointestinal","side_effects":["Headache","Nausea","Diarrhea"],"serious":["Kidney disease","Low magnesium"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"before food","usual_dose":"20mg-40mg","tip":"Take before breakfast for best effect."},
    "Loperamide":{"class":"Gastrointestinal","side_effects":["Constipation","Dizziness","Nausea"],"serious":["Heart rhythm problems","Toxic megacolon"],"risk_base":"medium","drowsy":False,"best_time":"any","food":"with or without food","usual_dose":"2mg","tip":"Use for maximum 2 days only. Stay hydrated."},
    "Amlodipine":{"class":"Antihypertensive","side_effects":["Swollen ankles","Flushing","Headache"],"serious":["Severe low BP","Chest pain","Heart failure"],"risk_base":"medium","drowsy":False,"best_time":"evening","food":"with or without food","usual_dose":"5mg-10mg","tip":"Take at same time daily. Do not stop suddenly."},
    "Atorvastatin":{"class":"Antihypertensive","side_effects":["Muscle pain","Joint pain","Diarrhea"],"serious":["Severe muscle breakdown","Liver damage"],"risk_base":"medium","drowsy":False,"best_time":"night","food":"with or without food","usual_dose":"10mg-40mg","tip":"Best taken at night — liver makes more cholesterol at night."},
    "Lisinopril":{"class":"Antihypertensive","side_effects":["Dry cough","Dizziness","Headache"],"serious":["Angioedema","Kidney failure"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with or without food","usual_dose":"5mg-20mg","tip":"May cause dry cough — normal side effect. Do not stop without doctor."},
    "Metoprolol":{"class":"Antihypertensive","side_effects":["Fatigue","Dizziness","Cold hands"],"serious":["Severe low heart rate","Heart failure"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"25mg-100mg","tip":"Never stop suddenly — can trigger heart attack. Always taper with doctor guidance."},
    "Warfarin":{"class":"Antihypertensive","side_effects":["Easy bruising","Bleeding gums","Fatigue"],"serious":["Severe internal bleeding","Brain hemorrhage","Stroke"],"risk_base":"high","drowsy":False,"best_time":"evening","food":"consistent diet","usual_dose":"1mg-10mg","tip":"Avoid vitamin K rich foods. Regular INR blood tests needed. No alcohol."},
    "Clopidogrel":{"class":"Antihypertensive","side_effects":["Bleeding easily","Bruising","Stomach pain"],"serious":["Severe bleeding","Stomach ulcers"],"risk_base":"high","drowsy":False,"best_time":"morning","food":"with or without food","usual_dose":"75mg","tip":"Never stop without doctor advice. Avoid aspirin unless prescribed together."},
    "Metformin":{"class":"Antidiabetic","side_effects":["Nausea","Diarrhea","Stomach pain"],"serious":["Lactic acidosis","Vitamin B12 deficiency"],"risk_base":"medium","drowsy":False,"best_time":"any","food":"always with food","usual_dose":"500mg-1000mg","tip":"Always take with meals to reduce stomach upset."},
    "Insulin (Regular)":{"class":"Antidiabetic","side_effects":["Low blood sugar","Injection site pain","Weight gain"],"serious":["Severe hypoglycemia","Hypokalemia"],"risk_base":"high","drowsy":False,"best_time":"before meals","food":"take before meals","usual_dose":"As prescribed only","tip":"Always carry glucose tablets. Never skip meals after taking insulin."},
    "Glibenclamide":{"class":"Antidiabetic","side_effects":["Low blood sugar","Nausea","Weight gain"],"serious":["Severe hypoglycemia","Liver damage"],"risk_base":"high","drowsy":False,"best_time":"morning","food":"with breakfast","usual_dose":"2.5mg-5mg","tip":"Take with breakfast. Carry sugar/glucose in case of hypoglycemia."},
    "Sitagliptin":{"class":"Antidiabetic","side_effects":["Runny nose","Headache","Stomach pain"],"serious":["Pancreatitis","Kidney problems"],"risk_base":"medium","drowsy":False,"best_time":"any","food":"with or without food","usual_dose":"100mg","tip":"Can be taken any time of day. Report severe stomach pain immediately."},
    "Sertraline":{"class":"Antidepressant","side_effects":["Nausea","Insomnia","Dizziness"],"serious":["Suicidal thoughts in youth","Serotonin syndrome"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"50mg-200mg","tip":"Takes 2-4 weeks to work fully. Never stop suddenly."},
    "Diazepam":{"class":"Antidepressant","side_effects":["Drowsiness","Dizziness","Fatigue"],"serious":["Addiction","Respiratory depression"],"risk_base":"high","drowsy":True,"best_time":"night","food":"with or without food","usual_dose":"2mg-10mg","tip":"Extremely drowsy. Never drive. High addiction risk. Short term use only."},
    "Alprazolam":{"class":"Antidepressant","side_effects":["Drowsiness","Dizziness","Memory issues"],"serious":["Severe addiction","Withdrawal seizures"],"risk_base":"high","drowsy":True,"best_time":"night","food":"with or without food","usual_dose":"0.25mg-0.5mg","tip":"Highly addictive. Never take without strict doctor supervision. Never drive."},
    "Fluoxetine":{"class":"Antidepressant","side_effects":["Nausea","Headache","Insomnia"],"serious":["Serotonin syndrome","Suicidal thoughts"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"20mg-60mg","tip":"Take in morning — can cause insomnia if taken at night."},
    "Melatonin":{"class":"Antidepressant","side_effects":["Drowsiness","Headache","Dizziness"],"serious":["Hormonal effects","Depression worsening"],"risk_base":"low","drowsy":True,"best_time":"night","food":"30 mins before bed","usual_dose":"0.5mg-5mg","tip":"Take 30 minutes before bedtime. Start with lowest dose."},
    "Nicotine Patch":{"class":"Smoking Cessation","side_effects":["Skin irritation","Vivid dreams","Headache"],"serious":["Heart problems if smoking continues","Nicotine overdose"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"not applicable","usual_dose":"7mg-21mg patch","tip":"Apply to clean dry skin. Remove at night if vivid dreams occur. Never smoke while using patch."},
    "Varenicline (Champix)":{"class":"Smoking Cessation","side_effects":["Nausea","Headache","Vivid dreams","Insomnia"],"serious":["Suicidal thoughts","Severe mood changes"],"risk_base":"high","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"0.5mg-1mg","tip":"Take with food. Report any mood changes immediately to your doctor."},
    "Nicotine Gum":{"class":"Smoking Cessation","side_effects":["Jaw pain","Hiccups","Mouth irritation"],"serious":["Nicotine overdose","Heart palpitations"],"risk_base":"low","drowsy":False,"best_time":"any","food":"avoid eating 15 mins before","usual_dose":"2mg-4mg","tip":"Chew slowly. Park between cheek and gum. Do not swallow."},
    "Bupropion (Zyban)":{"class":"Smoking Cessation","side_effects":["Dry mouth","Insomnia","Headache"],"serious":["Seizures","Suicidal thoughts"],"risk_base":"high","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"150mg","tip":"Start 1 week before quit date. Avoid if history of seizures."},
    "Disulfiram":{"class":"Alcohol Treatment","side_effects":["Drowsiness","Headache","Metallic taste"],"serious":["Severe reaction with alcohol","Liver damage"],"risk_base":"high","drowsy":True,"best_time":"morning","food":"with food","usual_dose":"250mg-500mg","tip":"NEVER drink alcohol while on this medicine — reaction can be life threatening."},
    "Naltrexone":{"class":"Alcohol Treatment","side_effects":["Nausea","Headache","Fatigue"],"serious":["Liver damage","Severe withdrawal"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"50mg","tip":"Do not take if using opioid pain medicines. Get regular liver tests."},
    "Alteplase":{"class":"Stroke Medicine","side_effects":["Bleeding","Bruising","Fever"],"serious":["Brain bleeding","Internal bleeding"],"risk_base":"high","drowsy":False,"best_time":"hospital only","food":"hospital only","usual_dose":"Hospital IV only","tip":"Emergency hospital medicine only. Given within 4.5 hours of stroke."},
    "Rivaroxaban":{"class":"Stroke Medicine","side_effects":["Bleeding","Nausea","Anemia"],"serious":["Severe uncontrolled bleeding","Spinal bleeding"],"risk_base":"high","drowsy":False,"best_time":"evening","food":"with evening meal","usual_dose":"10mg-20mg","tip":"Take with evening meal. Never stop without doctor advice."},
    "Vitamin C":{"class":"Supplement","side_effects":["Stomach upset","Diarrhea","Nausea"],"serious":["Kidney stones at high dose"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"500mg-1000mg","tip":"Take with food to avoid stomach upset. Excess is excreted in urine."},
    "Vitamin D3":{"class":"Supplement","side_effects":["Nausea","Constipation","Fatigue"],"serious":["Calcium toxicity at high dose"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"with fatty food","usual_dose":"1000-4000 IU","tip":"Take with a fatty meal for best absorption. Get blood levels checked every 6 months."},
    "Iron Supplement":{"class":"Supplement","side_effects":["Constipation","Stomach pain","Dark stools"],"serious":["Iron toxicity","Stomach bleeding"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"empty stomach or with vitamin C","usual_dose":"65mg-200mg","tip":"Take with Vitamin C for better absorption. Avoid with tea or coffee."},
    "Folic Acid":{"class":"Supplement","side_effects":["Nausea","Bloating","Loss of appetite"],"serious":["Masks B12 deficiency"],"risk_base":"low","drowsy":False,"best_time":"morning","food":"with or without food","usual_dose":"400mcg-5mg","tip":"Essential during pregnancy. Take daily at same time."},
    "Calcium Supplement":{"class":"Supplement","side_effects":["Constipation","Bloating","Gas"],"serious":["Kidney stones","High blood calcium"],"risk_base":"low","drowsy":False,"best_time":"evening","food":"with food","usual_dose":"500mg-1000mg","tip":"Take with food. Split into 2 doses for better absorption."},
    "Budesonide Inhaler":{"class":"Respiratory","side_effects":["Mouth thrush","Hoarse voice","Cough"],"serious":["Adrenal suppression","Bone loss"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"not applicable","usual_dose":"200mcg-400mcg","tip":"Rinse mouth after every use to prevent thrush. Never stop suddenly."},
    "Theophylline":{"class":"Respiratory","side_effects":["Nausea","Headache","Insomnia","Tremors"],"serious":["Seizures","Heart rhythm problems"],"risk_base":"high","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"100mg-300mg","tip":"Very narrow safety window. Regular blood level monitoring needed."},
    "Montelukast":{"class":"Respiratory","side_effects":["Headache","Stomach pain","Fatigue"],"serious":["Mental health changes","Suicidal thoughts"],"risk_base":"medium","drowsy":False,"best_time":"night","food":"with or without food","usual_dose":"10mg","tip":"Take at night. Report any mood changes or unusual behaviour immediately."},
    "Colchicine":{"class":"Bone Medicine","side_effects":["Diarrhea","Nausea","Stomach pain"],"serious":["Muscle damage","Nerve damage","Blood disorders"],"risk_base":"medium","drowsy":False,"best_time":"any","food":"with food","usual_dose":"0.5mg-1mg","tip":"Take at first sign of gout attack. Stop if diarrhea develops."},
    "Allopurinol":{"class":"Bone Medicine","side_effects":["Rash","Nausea","Drowsiness"],"serious":["Severe skin reactions","Liver damage"],"risk_base":"medium","drowsy":False,"best_time":"morning","food":"with food","usual_dose":"100mg-300mg","tip":"Drink plenty of water daily. Takes weeks to work. Do not take during acute gout attack."},
    "Latanoprost Eye Drops":{"class":"Eye Medicine","side_effects":["Eye redness","Iris color change","Eye irritation"],"serious":["Macular edema","Severe eye inflammation"],"risk_base":"low","drowsy":False,"best_time":"night","food":"not applicable","usual_dose":"1 drop daily","tip":"Apply at bedtime. Remove contact lenses before use. Wash hands thoroughly."},
    "Chloramphenicol Eye Drops":{"class":"Eye Medicine","side_effects":["Eye stinging","Temporary blurred vision"],"serious":["Aplastic anemia (very rare)"],"risk_base":"low","drowsy":False,"best_time":"any","food":"not applicable","usual_dose":"1 drop every 2 hours","tip":"Wash hands before applying. Do not touch dropper tip to eye."},
}

DRUG_CLASS_MAP = {"Painkiller":0,"Antibiotic":1,"Antiallergy":2,"Gastrointestinal":3,"Antihypertensive":4,"Antidiabetic":5,"Antidepressant":6,"Supplement":7,"Smoking Cessation":8,"Alcohol Treatment":9,"Stroke Medicine":10,"Respiratory":11,"Bone Medicine":12,"Eye Medicine":13}

CATEGORIES = {
    "💊 Painkillers & Fever":["Paracetamol","Ibuprofen","Aspirin","Diclofenac","Tramadol","Mefenamic Acid","Codeine","Naproxen"],
    "🦠 Antibiotics":["Amoxicillin","Ciprofloxacin","Azithromycin","Doxycycline","Metronidazole"],
    "🤧 Cold, Cough & Allergy":["Cetirizine","Loratadine","Chlorpheniramine","Salbutamol","Fexofenadine"],
    "🫃 Stomach & Digestion":["Omeprazole","Pantoprazole","Ondansetron","Domperidone","Loperamide"],
    "❤️ Heart & Blood Pressure":["Amlodipine","Atorvastatin","Lisinopril","Metoprolol","Warfarin","Clopidogrel"],
    "🩸 Diabetes":["Metformin","Insulin (Regular)","Glibenclamide","Sitagliptin"],
    "🧠 Mental Health & Sleep":["Sertraline","Diazepam","Alprazolam","Fluoxetine","Melatonin"],
    "🚬 Smoking Cessation":["Nicotine Patch","Varenicline (Champix)","Nicotine Gum","Bupropion (Zyban)"],
    "🍺 Alcohol Treatment":["Disulfiram","Naltrexone"],
    "🧠 Stroke Medicines":["Alteplase","Rivaroxaban"],
    "🌿 Vitamins & Supplements":["Vitamin C","Vitamin D3","Iron Supplement","Folic Acid","Calcium Supplement"],
    "🫁 Respiratory":["Budesonide Inhaler","Theophylline","Montelukast"],
    "🦴 Bone & Joint":["Colchicine","Allopurinol"],
    "👁️ Eye Medicines":["Latanoprost Eye Drops","Chloramphenicol Eye Drops"],
}

# ── Context-aware risk logic ──────────────────────────────────────────────────
def calculate_context_risk(drug_info, age, condition, activity, time_of_day):
    base = drug_info["risk_base"]
    risk_score = {"low":1,"medium":2,"high":3}[base]

    reasons = []
    suggestions = []
    tips = []

    # Age factor
    if age > 60:
        risk_score += 1
        reasons.append("👴 Elderly patients process medicine slower — higher risk")
    elif age < 18:
        risk_score += 1
        reasons.append("👦 Young patients need weight-based dosing — consult doctor")

    # Condition factor
    if condition == "Kidney Disease":
        risk_score += 1
        reasons.append("🫘 Kidney disease affects how medicine is removed from body")
        if drug_info["class"] == "Painkiller":
            risk_score += 1
            suggestions.append("💡 Consider Paracetamol instead — safer for kidney patients")
    elif condition == "Liver Disease":
        risk_score += 1
        reasons.append("🫀 Liver disease affects how medicine is processed")
    elif condition == "Heart Disease":
        risk_score += 1
        reasons.append("❤️ Heart disease increases sensitivity to many medicines")
        if drug_info["class"] == "Painkiller" and drug_info.get("drowsy") == False:
            suggestions.append("💡 NSAIDs increase heart attack risk — ask doctor for safer alternative")
    elif condition == "Diabetes":
        risk_score += 1
        reasons.append("🩸 Diabetes can affect how medicine interacts with blood sugar")
    elif condition == "Pregnancy":
        risk_score += 2
        reasons.append("🤰 Pregnancy requires extreme caution — many medicines are unsafe")
        suggestions.append("💡 Always consult gynecologist before any medicine during pregnancy")

    # Drowsy medicine + activity
    if drug_info["drowsy"]:
        if activity == "🚗 Driving":
            risk_score += 2
            reasons.append("🚗 This medicine causes DROWSINESS — extremely dangerous while driving")
            suggestions.append("🚨 DO NOT drive after taking this medicine — serious accident risk!")
        elif activity == "📚 Studying":
            risk_score += 1
            reasons.append("📚 Drowsy medicine will reduce your concentration and focus")
            suggestions.append("💡 Take this medicine AFTER studying, not before")
        elif activity == "💻 Working":
            risk_score += 1
            reasons.append("💻 Drowsiness from medicine may affect work performance")
            suggestions.append("💡 Schedule this medicine for after work hours or at night")
        elif activity == "🏃 Exercising":
            risk_score += 1
            reasons.append("🏃 Drowsy medicine + exercise = dizziness and fall risk")
            suggestions.append("💡 Avoid exercise for 4-6 hours after taking this medicine")

    # Time of day factor
    best = drug_info["best_time"]
    if best == "night" and time_of_day == "🌅 Morning":
        suggestions.append(f"⏰ This medicine works BEST at night — consider switching to night dose")
    elif best == "morning" and time_of_day == "🌙 Night":
        suggestions.append(f"⏰ This medicine is best taken in the morning for optimal effect")
    elif best == "morning" and time_of_day == "🌅 Morning":
        tips.append(f"✅ Great timing! Morning is the best time for this medicine")

    # Food tip
    tips.append(f"🍽️ Take {drug_info['food']}")
    tips.append(f"💊 {drug_info['tip']}")

    # Final risk level
    if risk_score <= 2:
        final_risk = "LOW"
    elif risk_score <= 4:
        final_risk = "MEDIUM"
    else:
        final_risk = "HIGH"

    # Base reason always added
    reasons.insert(0, f"💊 {drug_info['class']} medicine with {drug_info['risk_base']} base risk profile")

    return final_risk, reasons, suggestions, tips

# ── Activity impact table ─────────────────────────────────────────────────────
def get_activity_impact(drug_info):
    activities = {
        "🚗 Driving": None,
        "📚 Studying": None,
        "💻 Working": None,
        "🏃 Exercising": None,
        "😴 Resting/Sleeping": None,
    }
    drowsy = drug_info["drowsy"]
    risk = drug_info["risk_base"]

    activities["🚗 Driving"] = ("❌ AVOID","danger") if drowsy else ("⚠️ Be Careful","warn") if risk=="high" else ("✅ Generally Safe","safe")
    activities["📚 Studying"] = ("⚠️ Low Focus","warn") if drowsy else ("✅ Safe","safe")
    activities["💻 Working"] = ("⚠️ Take Care","warn") if drowsy else ("✅ Safe","safe")
    activities["🏃 Exercising"] = ("⚠️ Avoid Heavy Exercise","warn") if drowsy or risk=="high" else ("✅ Safe","safe")
    activities["😴 Resting/Sleeping"] = ("✅ Best Time","safe") if drowsy else ("✅ Safe","safe")

    return activities

# ── Model training ────────────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    np.random.seed(42); n=800
    data={"drug_class":np.random.choice(list(range(14)),n),"drug_risk_base":np.random.choice([1,2,3],n),"num_ingredients":np.random.randint(1,6,n),"patient_age_group":np.random.choice([0,1,2],n),"has_kidney_issue":np.random.choice([0,1],n,p=[0.8,0.2]),"has_liver_issue":np.random.choice([0,1],n,p=[0.85,0.15]),"has_heart_issue":np.random.choice([0,1],n,p=[0.85,0.15]),"has_diabetes":np.random.choice([0,1],n,p=[0.8,0.2]),"is_pregnant":np.random.choice([0,1],n,p=[0.9,0.1]),"dosage_level":np.random.choice([1,2,3],n),"drug_interactions":np.random.randint(0,6,n),"allergy_history":np.random.choice([0,1],n,p=[0.85,0.15])}
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

# ── AI Doctor ─────────────────────────────────────────────────────────────────
def ask_ai_doctor(question):
    q = question.lower().strip()
    q = re.sub(r'[^\w\s]', ' ', q)

    greetings = ["hi","hello","hey","hii","helo","hai","vanakkam","good morning","good evening","sup","wassup","yo"]
    if any(g in q.split() for g in greetings) and len(q.split()) < 5:
        return "Hello! I'm your AI Doctor 👨‍⚕️ Ask me anything about medicines, side effects, or drug safety — I'm here to help! 😊"

    if any(w in q for w in ["thank","thanks","thank you","thx","ty"]):
        return "You're welcome! Your health is my priority. Feel free to ask anything else! 😊"

    food_words = ["biriyani","biryani","food","eat","rice","chapati","dosa","idli","coffee","tea","milk","juice","drink"]
    if any(f in q for f in food_words) and any(m in q for m in ["medicine","tablet","paracetamol","ibuprofen","drug","pill"]):
        return "Good question! Most medicines are actually BETTER taken with food — it protects your stomach. However some medicines need to be taken on an empty stomach. Always check the label or ask your pharmacist! Taking medicine with a full meal is totally fine for most tablets 😄 Remember: Always consult a real doctor before taking any medicine."

    overdose_hints = ["10 tablet","too many","too much","lot of tablet","many tablet","20 tablet","whole bottle","full strip","overdose","ate too many","swallowed"]
    if any(o in q for o in overdose_hints):
        return "⚠️ THIS IS SERIOUS! If someone has taken too many tablets, call emergency services IMMEDIATELY — 108 in India. Do NOT try to make them vomit. Take the medicine bottle with you to the hospital. Every second matters — please act fast! Remember: Always consult a real doctor before taking any medicine."

    smoke_words = ["smoke","smoking","cigarette","tobacco","nicotine","nicotin","nicoten","nico","bidi","hookah","vape","quit smoking","stop smoking","chewing tobacco","gutka","beedi","cigar","champix","varenicline","bupropion","zyban"]
    if any(s in q for s in smoke_words):
        return "Quitting smoking is one of the BEST decisions you can make! 🚭 Medicines that help: Nicotine Patch, Nicotine Gum, Varenicline (Champix), Bupropion (Zyban). These work by reducing cravings and withdrawal symptoms. IMPORTANT — never smoke while using nicotine replacement therapy as it causes nicotine overdose. Always get a doctor's guidance before starting. Remember: Always consult a real doctor before taking any medicine."

    alcohol_words = ["alcohol","drink","drinking","beer","wine","whiskey","rum","liquor","drunk","arrack","toddy","குடி","மது"]
    if any(a in q for a in alcohol_words):
        return "Mixing alcohol with medicines is VERY DANGEROUS! 🍺⚠️ Critical combinations to NEVER do: Paracetamol + alcohol = serious liver damage, Metronidazole + alcohol = violent vomiting and heart problems, Diazepam/Alprazolam + alcohol = can stop breathing (potentially fatal), Warfarin + alcohol = dangerous bleeding risk. For quitting alcohol, medicines like Disulfiram, Naltrexone can help — but ONLY under doctor supervision. Remember: Always consult a real doctor before taking any medicine."

    stroke_words = ["stroke","paralysis","brain attack","clot","blood clot","brain bleed","sudden weakness","face drooping","speech problem"]
    if any(s in q for s in stroke_words):
        return "Stroke is a MEDICAL EMERGENCY — every minute counts! 🧠⚠️ Remember FAST: F = Face drooping, A = Arm weakness, S = Speech difficulty, T = Time to call 108 immediately! Treatment with Alteplase must happen within 4.5 hours of stroke onset. After stroke, blood thinners and statins are often prescribed to prevent another stroke. Remember: Always consult a real doctor before taking any medicine."

    period_words = ["period","periods","menstrual","menstruation","cramps","period pain","monthly","menses","pms","irregular period","heavy bleeding","period cramp","dysmenorrhea"]
    if any(p in q for p in period_words):
        return "For period pain! 🩸 Best medicines: Mefenamic Acid (250mg-500mg) — most effective for period cramps, take at start of pain. Ibuprofen (200mg-400mg) — reduces pain and inflammation, take with food. Paracetamol — for mild pain. Non-medicine tips: Hot water bottle on abdomen, stay hydrated, gentle exercise helps. See a doctor if pain is severely unbearable or periods are irregular — could indicate endometriosis or PCOS. Remember: Always consult a real doctor before taking any medicine."

    fever_words = ["fever","temperature","hot","pyrexia","high temperature","body heat"]
    if any(f in q for f in fever_words):
        return "For fever! 🌡️ Best medicine: Paracetamol (500mg-1000mg every 6-8 hours, max 4g per day). Stay well hydrated, rest, use cool wet cloth on forehead. See a doctor IMMEDIATELY if: Fever above 39.4°C (103°F), fever lasts more than 3 days, fever with severe headache + stiff neck + rash. Don't use Aspirin for fever in children EVER. Remember: Always consult a real doctor before taking any medicine."

    preg_words = ["pregnant","pregnancy","baby","unborn","fetus","trimester","expecting"]
    if any(p in q for p in preg_words):
        return "Pregnancy requires EXTRA caution with medicines! 🤰 Generally safer (with doctor advice): Paracetamol, Folic Acid, Iron supplements. AVOID during pregnancy: Ibuprofen (especially after 30 weeks), Aspirin (high dose), Tramadol, Codeine, Warfarin, most antibiotics. NEVER take any medicine during pregnancy without consulting your gynecologist first. Remember: Always consult a real doctor before taking any medicine."

    child_words = ["child","baby","infant","kid","children","toddler","years old","month old","year old","newborn"]
    if any(c in q for c in child_words):
        return "Children need SPECIAL care with medicines! 👶 Key rules: Doses are always based on weight or age — never give adult doses. NEVER give Aspirin to children under 16 — it can cause dangerous Reye's syndrome. Ibuprofen not recommended under 6 months. For fever, use children's Paracetamol syrup at correct weight-based dose. NEVER break adult tablets for children without medical advice. Remember: Always consult a real doctor before taking any medicine."

    kidney_words = ["kidney","renal","dialysis","kidney disease","kidney problem","kidney failure"]
    if any(k in q for k in kidney_words):
        return "Kidney disease patients need EXTRA caution! 🫘 AVOID: Ibuprofen, Naproxen, Diclofenac (NSAIDs damage kidneys), Metformin (banned in kidney failure). SAFER options: Paracetamol at recommended doses. Always tell EVERY doctor about your kidney condition. Get regular kidney function tests. Remember: Always consult a real doctor before taking any medicine."

    sleep_words = ["sleep","insomnia","cant sleep","sleepless","sleeping problem","not sleeping","sleep medicine"]
    if any(s in q for s in sleep_words):
        return "For sleep problems! 😴 Safe medicines: Melatonin (0.5mg-5mg) — natural sleep hormone, take 30 mins before bed. AVOID without doctor: Diazepam, Alprazolam — highly addictive. Non-medicine tips: Stop phone/screen 1 hour before bed, sleep and wake at same time daily, avoid caffeine after 3pm, keep room cool and dark. Remember: Always consult a real doctor before taking any medicine."

    muscle_words = ["muscle","body ache","bodyache","muscle pain","body pain","sore","ache","sprain","stiff"]
    if any(m in q for m in muscle_words):
        return "For muscle pain and body aches! 💪 Best medicines: Paracetamol — safest first choice. Ibuprofen or Diclofenac — better for inflammation-related pain, take with food. Diclofenac gel — apply directly on sore muscle. Non-medicine tips: Heat pack for muscle spasms, ice pack for sprains (first 24 hours), rest and gentle stretching. Remember: Always consult a real doctor before taking any medicine."

    cold_words = ["cold","flu","running nose","runny nose","blocked nose","sneezing","sore throat","cough","congestion","throat pain"]
    if any(c in q for c in cold_words):
        return "For cold and flu! 🤧 Medicines that help: Paracetamol — for fever and headache. Cetirizine or Loratadine — for runny nose and sneezing. Bromhexine — to loosen mucus. Throat lozenges — for sore throat. Home remedies: Steam inhalation, honey + ginger tea, gargle with warm salt water, rest and lots of fluids. Antibiotics do NOT work for cold/flu — it's viral! Remember: Always consult a real doctor before taking any medicine."

    tooth_words = ["tooth","teeth","toothache","dental","tooth pain","gum","cavity"]
    if any(t in q for t in tooth_words):
        return "For tooth pain! 🦷 Medicines for relief: Paracetamol (500mg-1000mg) — safest first choice. Ibuprofen (400mg) — better if there is swelling. Clove oil — soak cotton and apply on tooth — natural numbing effect! IMPORTANT — tooth pain is just a symptom! Always see a dentist — medicine only masks the pain. Remember: Always consult a real doctor before taking any medicine."

    nausea_words = ["nausea","vomit","vomiting","nauseous","feel sick","throwing up","motion sickness"]
    if any(n in q for n in nausea_words):
        return "For nausea and vomiting! 🤢 Best medicines: Ondansetron (4mg) — very effective. Domperidone (10mg) — take 30 minutes before meals. Home remedies: Ginger tea, lemon smell, cold water sips, eat small amounts frequently. See a doctor if vomiting lasts more than 24 hours or there is blood in vomit. Remember: Always consult a real doctor before taking any medicine."

    emergency_words = ["emergency","urgent","serious","danger","cant breathe","not breathing","unconscious","collapsed","108","ambulance"]
    if any(e in q for e in emergency_words):
        return "🚨 EMERGENCY! Call 108 immediately if you see: Chest pain or pressure, difficulty breathing, sudden weakness of face/arm/leg, slurred speech, loss of consciousness, severe uncontrolled bleeding, suspected poisoning or overdose. Do NOT drive yourself to the hospital — call 108. Time is life — don't delay! Remember: Always consult a real doctor before taking any medicine."

    driving_words = ["drive","driving","can i drive","safe to drive","after medicine"]
    if any(d in q for d in driving_words):
        return "Driving safety after medicine is very important! 🚗 NEVER drive after: Diazepam, Alprazolam, Codeine, Tramadol, Cetirizine, Chlorpheniramine, Disulfiram — these cause heavy drowsiness. GENERALLY SAFE to drive: Paracetamol, Amoxicillin, Metformin, Loratadine, Fexofenadine. When in doubt — DON'T drive. Your safety and others' safety matters more. Remember: Always consult a real doctor before taking any medicine."

    for drug_name, info in DRUG_DATA.items():
        drug_lower = drug_name.lower().replace("(","").replace(")","").replace("inhaler","").strip()
        short_name = drug_lower.split()[0]
        if short_name in q or drug_lower in q or drug_name.lower() in q:
            common = ", ".join(info["side_effects"])
            serious = ", ".join(info["serious"])
            return f"Here's what you need to know about **{drug_name}** ({info['class']}):\n\n📋 **Common side effects:** {common}\n\n⚠️ **Serious side effects:** {serious}\n\n💊 **Usual dose:** {info['usual_dose']}\n\n⏰ **Best time to take:** {info['best_time']}\n\n💡 **Tip:** {info['tip']}\n\nRemember: Always consult a real doctor before taking any medicine."

    return "Thank you for your question! For general medicine safety: always follow your doctor's prescription exactly, take the correct dose, never share medicines, and contact your doctor immediately if you notice any unusual symptoms. Could you be more specific? Ask about a drug name, a condition, or a symptom and I'll give you a detailed answer! Remember: Always consult a real doctor before taking any medicine."

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💊 Drug Safety AI")
    st.markdown("---")
    page=st.radio("📍 Navigate",["🏠 Home","🔬 Safety Dashboard","🤖 AI Doctor"])
    st.markdown("---")
    st.markdown(f"**Model Accuracy:** {accuracy*100:.1f}%")
    st.markdown(f"**Drugs:** {len(DRUG_DATA)}")
    st.markdown(f"**Categories:** {len(CATEGORIES)}")
    st.markdown("---")
    st.caption("Context-Aware Smart Medicine Safety Dashboard | International Science Project 2025")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page=="🏠 Home":
    st.markdown("<div style='text-align:center;padding:2rem 0'><div style='font-size:4rem'>💊</div><h1 style='color:#00a884;font-size:2.5rem;font-weight:800'>Context-Aware Smart<br>Medicine Safety Dashboard</h1><p style='font-size:1.1rem;color:#666;max-width:650px;margin:auto;margin-top:1rem'>Unlike traditional systems, this project combines side effect prediction with real-life context like user activity and time — helping users make safer everyday decisions.</p></div>",unsafe_allow_html=True)
    st.markdown("---")
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("<div class='feature-card'><div style='font-size:2.5rem'>🧠</div><h3>Context-Aware Risk</h3><p>Considers your activity (driving, studying, working) and time of day to give personalized risk assessment</p></div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='feature-card'><div style='font-size:2.5rem'>💥</div><h3>Smart Life Impact</h3><p>Dynamic table showing how each medicine affects your daily activities — unique feature no other tool provides</p></div>",unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='feature-card'><div style='font-size:2.5rem'>💡</div><h3>Better Choice Suggestion</h3><p>AI tells you the better time to take your medicine and activities to avoid — like a real doctor's advice</p></div>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"### 💊 {len(DRUG_DATA)} Drugs Across {len(CATEGORIES)} Categories")
    cols=st.columns(4)
    for i,cat in enumerate(CATEGORIES.keys()):
        with cols[i%4]:
            st.markdown(f"<div style='padding:0.75rem;border-radius:8px;border:1px solid #ddd;margin:0.3rem 0;text-align:center'><b>{cat}</b><br><span style='color:#00a884;font-size:1.2rem'>{len(CATEGORIES[cat])} drugs</span></div>",unsafe_allow_html=True)
    st.markdown("---")
    s1,s2,s3,s4=st.columns(4)
    for col,val,label in [(s1,str(len(DRUG_DATA)),"Drugs Database"),(s2,"5","Activity Contexts"),(s3,f"{accuracy*100:.0f}%","Model Accuracy"),(s4,"4","Time Contexts")]:
        with col:
            st.markdown(f"<div style='text-align:center;padding:1rem;border-radius:12px;border:2px solid #00a884'><div style='font-size:2rem;font-weight:800;color:#00a884'>{val}</div><div style='font-size:0.85rem;color:#666'>{label}</div></div>",unsafe_allow_html=True)
    st.markdown("---")
    st.info("👈 Go to **Safety Dashboard** to analyze your medicine!")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SAFETY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🔬 Safety Dashboard":
    st.markdown('<h2 style="color:#00a884">🔬 Context-Aware Medicine Safety Dashboard</h2>',unsafe_allow_html=True)
    st.caption("Enter your medicine and real-life context — get personalized safety analysis instantly")
    st.markdown("---")

    with st.container():
        st.markdown("### 📋 Enter Your Details")
        c1,c2,c3=st.columns(3)
        with c1:
            st.markdown("**💊 Medicine**")
            category=st.selectbox("Category",list(CATEGORIES.keys()),label_visibility="collapsed")
            drug_name=st.selectbox("Drug",CATEGORIES[category],label_visibility="collapsed")
            drug_info=DRUG_DATA[drug_name]
            st.success(f"**{drug_info['class']}** | Dose: {drug_info['usual_dose']}")
        with c2:
            st.markdown("**👤 Patient Info**")
            age=st.number_input("Age",min_value=1,max_value=100,value=25)
            condition=st.selectbox("Medical Condition",["None","Kidney Disease","Liver Disease","Heart Disease","Diabetes","Pregnancy","High BP"])
        with c3:
            st.markdown("**⚡ Your Context Right Now**")
            activity=st.selectbox("Current Activity",["🚗 Driving","📚 Studying","💻 Working","🏃 Exercising","😴 Resting/Sleeping"])
            time_of_day=st.selectbox("Time of Day",["🌅 Morning","☀️ Afternoon","🌆 Evening","🌙 Night"])

    st.markdown("<br>",unsafe_allow_html=True)

    if st.button("🔍 Analyze Safety Now",use_container_width=True):
        final_risk,reasons,suggestions,tips=calculate_context_risk(drug_info,age,condition,activity,time_of_day)

        st.markdown("---")
        st.markdown("## 📊 Your Safety Dashboard")

        # Risk meter
        risk_colors={"LOW":"#00c853","MEDIUM":"#ffa000","HIGH":"#d32f2f"}
        risk_bg={"LOW":"#e8f5e9","MEDIUM":"#fff8e1","HIGH":"#ffebee"}
        risk_emoji={"LOW":"🟢","MEDIUM":"🟡","HIGH":"🔴"}
        rc=risk_colors[final_risk]
        rb=risk_bg[final_risk]

        m1,m2,m3=st.columns(3)
        with m1:
            st.markdown(f"<div style='background:{rb};border:2px solid {rc};border-radius:16px;padding:1.5rem;text-align:center'><div style='font-size:3rem'>{risk_emoji[final_risk]}</div><div style='font-size:2.5rem;font-weight:800;color:{rc}'>{final_risk}</div><div style='color:#666'>Risk Level</div></div>",unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:#00a884'>{accuracy*100:.0f}%</div><div class='metric-lbl'>Model Accuracy</div></div>",unsafe_allow_html=True)
        with m3:
            drowsy_txt="⚠️ YES — Causes Drowsiness" if drug_info["drowsy"] else "✅ No Drowsiness"
            drowsy_color="#d32f2f" if drug_info["drowsy"] else "#00c853"
            st.markdown(f"<div class='metric-card'><div class='metric-val' style='color:{drowsy_color};font-size:1.2rem'>{drowsy_txt}</div><div class='metric-lbl'>Drowsiness Alert</div></div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)

        # Side effects + Why this risk
        c1,c2=st.columns(2)
        with c1:
            st.markdown("### 💊 Side Effects")
            st.markdown("**Common (mild):**")
            for s in drug_info["side_effects"]:
                st.markdown(f"<div class='tip-box'>🟡 {s}</div>",unsafe_allow_html=True)
            st.markdown("**Serious (watch out):**")
            for s in drug_info["serious"]:
                st.markdown(f"<div style='background:#ffebee;border-left:4px solid #d32f2f;padding:0.8rem 1.2rem;border-radius:8px;margin:0.3rem 0'>🔴 {s}</div>",unsafe_allow_html=True)

        with c2:
            st.markdown("### 🧠 Why This Risk?")
            st.caption("AI explanation of factors affecting your risk")
            for r in reasons:
                st.markdown(f"<div class='why-box'>{r}</div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)

        # Smart Life Impact Panel — THE KILLER FEATURE
        st.markdown("### 💥 Smart Life Impact Panel")
        st.caption("How this medicine affects your daily activities — personalized to your profile")

        impact=get_activity_impact(drug_info)
        cols=st.columns(5)
        activity_list=list(impact.items())
        for i,(act,(status,level)) in enumerate(activity_list):
            with cols[i]:
                color={"safe":"#e8f5e9","warn":"#fff8e1","danger":"#ffebee"}[level]
                border={"safe":"#00c853","warn":"#ffa000","danger":"#d32f2f"}[level]
                highlight=" border: 3px solid "+border+";" if act==activity else f" border: 1px solid {border};"
                tag="👈 YOU ARE HERE" if act==activity else ""
                st.markdown(f"<div style='background:{color};{highlight}border-radius:12px;padding:1rem;text-align:center'><div style='font-size:1.5rem'>{act.split()[0]}</div><div style='font-weight:600;font-size:0.85rem'>{act[2:]}</div><div style='margin-top:0.5rem;font-weight:700;font-size:0.85rem;color:{border}'>{status}</div><div style='font-size:0.7rem;color:#888'>{tag}</div></div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)

        # Better choice suggestions
        if suggestions:
            st.markdown("### 💡 Better Choice Suggestions")
            st.caption("AI recommendations to make this medicine safer for you")
            for s in suggestions:
                st.markdown(f"<div class='suggest-box'>{s}</div>",unsafe_allow_html=True)

        # Smart tips
        st.markdown("### 🌟 Smart Tips for You")
        for t in tips:
            st.markdown(f"<div class='tip-box'>{t}</div>",unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.success("✅ For research and educational purposes only. Always consult a certified doctor before taking any medicine.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AI DOCTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🤖 AI Doctor":
    st.markdown('<h2 style="color:#00a884">🤖 AI Doctor — Ask Me Anything!</h2>',unsafe_allow_html=True)
    st.caption("Chat with our AI Doctor about medicines, side effects, alcohol, smoking, stroke and more!")
    chat_lang="English"

    st.markdown("<div style='background:#e8f5e9;padding:1rem;border-radius:12px;border-left:4px solid #00a884;margin-bottom:1rem'><b>👨‍⚕️ Hi! I'm your AI Doctor.</b> Ask me about any medicine, symptom, or health concern — I'll give you clear, simple advice!</div>",unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    st.markdown("**⚡ Quick Questions:**")
    q1,q2,q3,q4,q5,q6=st.columns(6)
    quick=""
    with q1:
        if st.button("🤒 Fever"): quick="I have fever what medicine should I take"
    with q2:
        if st.button("🚬 Smoking"): quick="How to quit smoking what medicines help"
    with q3:
        if st.button("🍺 Alcohol"): quick="Is it safe to drink alcohol with medicine"
    with q4:
        if st.button("🧠 Stroke"): quick="What are stroke symptoms and medicines"
    with q5:
        if st.button("👶 Child"): quick="My child has fever what medicine is safe"
    with q6:
        if st.button("🩸 Periods"): quick="Medicine for period pain and cramps"

    st.markdown("**💬 Conversation:**")
    if not st.session_state.messages:
        st.markdown("<div class='chat-ai'>👨‍⚕️ Hello! I'm your AI Doctor 😊 Ask me anything about:<br>💊 Medicine side effects &nbsp;|&nbsp; 🚬 Smoking &nbsp;|&nbsp; 🍺 Alcohol &nbsp;|&nbsp; 🧠 Stroke<br>👶 Children &nbsp;|&nbsp; 🤰 Pregnancy &nbsp;|&nbsp; 🩸 Periods &nbsp;|&nbsp; 🚨 Emergencies</div>",unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"]=="user":
            st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>",unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>👨‍⚕️ {msg['content']}</div>",unsafe_allow_html=True)

    c1,c2=st.columns([4,1])
    with c1:
        user_input=st.text_input("Type your question...",placeholder="e.g. Can I take Paracetamol and drive?",label_visibility="collapsed")
    with c2:
        send=st.button("Send 📤",use_container_width=True)

    question=quick if quick else (user_input if send and user_input else "")
    if question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.spinner("👨‍⚕️ Thinking..."):
            reply=ask_ai_doctor(question)
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages=[]
            st.rerun()

    st.markdown("---")
    st.warning("⚠️ AI Doctor provides general information only. Always consult a real certified doctor for medical decisions.")
