import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle, warnings, re
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Drug Safety Assistant", page_icon="💊", layout="wide")

st.markdown("""
<style>
.title{font-size:2.2rem;font-weight:800;color:#00a884;text-align:center}
.subtitle{font-size:1rem;color:#666;text-align:center;margin-bottom:1rem}
.metric-card{padding:1rem;border-radius:12px;text-align:center;margin-bottom:0.5rem;border:1px solid #ddd}
.metric-value{font-size:2rem;font-weight:700}
.metric-label{font-size:0.85rem;opacity:0.7}
.recommend-box{border-left:4px solid #00a884;padding:1rem 1.5rem;border-radius:8px;margin:0.5rem 0;border:1px solid #eee}
.side-box{padding:0.5rem 1rem;border-radius:8px;margin:0.3rem 0;border:1px solid #eee}
.risk-box{padding:1.5rem;border-radius:12px;text-align:center;margin:1rem 0}
.feature-card{padding:1.5rem;border-radius:12px;border:1px solid #ddd;margin:0.5rem;text-align:center}
.chat-user{background:#00a884;color:white;padding:0.75rem 1rem;border-radius:12px 12px 2px 12px;margin:0.5rem 0;max-width:80%;margin-left:auto;text-align:right}
.chat-ai{background:#f0f0f0;color:#222;padding:0.75rem 1rem;border-radius:12px 12px 12px 2px;margin:0.5rem 0;max-width:85%}
section[data-testid="stSidebar"]{background:#00a884}
section[data-testid="stSidebar"] *{color:white !important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DRUG DATABASE — 100 DRUGS
# ══════════════════════════════════════════════════════════════════════════════
DRUG_DATA = {
    # PAINKILLERS
    "Paracetamol (Acetaminophen)":{"class":"Painkiller","common":["Nausea","Stomach pain","Loss of appetite"],"serious":["Liver damage","Kidney damage","Severe allergic reaction"],"risk_base":1,"usual_dose":"500mg - 1000mg"},
    "Ibuprofen":{"class":"Painkiller","common":["Stomach upset","Heartburn","Dizziness"],"serious":["Stomach bleeding","Kidney problems","Heart attack risk"],"risk_base":2,"usual_dose":"200mg - 400mg"},
    "Aspirin":{"class":"Painkiller","common":["Stomach irritation","Nausea","Heartburn"],"serious":["Internal bleeding","Reye's syndrome in children","Stroke risk"],"risk_base":2,"usual_dose":"300mg - 600mg"},
    "Diclofenac":{"class":"Painkiller","common":["Stomach pain","Nausea","Headache"],"serious":["Stomach ulcers","Heart attack","Kidney failure"],"risk_base":2,"usual_dose":"50mg - 75mg"},
    "Tramadol":{"class":"Painkiller","common":["Nausea","Dizziness","Constipation"],"serious":["Seizures","Addiction risk","Breathing problems"],"risk_base":3,"usual_dose":"50mg - 100mg"},
    "Naproxen":{"class":"Painkiller","common":["Stomach upset","Heartburn","Drowsiness"],"serious":["GI bleeding","Kidney damage","Heart problems"],"risk_base":2,"usual_dose":"250mg - 500mg"},
    "Codeine":{"class":"Painkiller","common":["Constipation","Drowsiness","Nausea"],"serious":["Addiction","Breathing problems","Liver damage"],"risk_base":3,"usual_dose":"15mg - 60mg"},
    "Mefenamic Acid":{"class":"Painkiller","common":["Stomach upset","Diarrhea","Dizziness"],"serious":["Kidney failure","Stomach bleeding","Seizures"],"risk_base":2,"usual_dose":"250mg - 500mg"},
    "Ketorolac":{"class":"Painkiller","common":["Stomach pain","Nausea","Dizziness"],"serious":["Stomach bleeding","Kidney failure","Heart attack"],"risk_base":3,"usual_dose":"10mg"},
    "Morphine":{"class":"Painkiller","common":["Drowsiness","Constipation","Nausea"],"serious":["Addiction","Breathing depression","Overdose risk"],"risk_base":3,"usual_dose":"As prescribed only"},
    # ANTIBIOTICS
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
    # COLD ALLERGY
    "Cetirizine":{"class":"Antiallergy","common":["Drowsiness","Dry mouth","Headache"],"serious":["Severe allergic reaction","Fast heartbeat","Tremors"],"risk_base":1,"usual_dose":"10mg"},
    "Loratadine":{"class":"Antiallergy","common":["Headache","Dry mouth","Fatigue"],"serious":["Fast heartbeat","Liver problems","Severe allergic reaction"],"risk_base":1,"usual_dose":"10mg"},
    "Chlorpheniramine":{"class":"Antiallergy","common":["Drowsiness","Dry mouth","Dizziness"],"serious":["Urinary retention","Confusion in elderly","Vision problems"],"risk_base":1,"usual_dose":"4mg"},
    "Phenylephrine":{"class":"Antiallergy","common":["Headache","Nausea","Increased BP"],"serious":["Severe hypertension","Heart attack","Stroke"],"risk_base":2,"usual_dose":"10mg"},
    "Dextromethorphan":{"class":"Antiallergy","common":["Drowsiness","Dizziness","Nausea"],"serious":["Serotonin syndrome","Hallucinations","Dependency"],"risk_base":2,"usual_dose":"10mg - 20mg"},
    "Bromhexine":{"class":"Antiallergy","common":["Nausea","Diarrhea","Dizziness"],"serious":["Severe skin reactions","Liver problems"],"risk_base":1,"usual_dose":"8mg"},
    "Salbutamol":{"class":"Antiallergy","common":["Tremors","Headache","Fast heartbeat"],"serious":["Severe chest pain","Irregular heartbeat","Low potassium"],"risk_base":2,"usual_dose":"2mg - 4mg"},
    "Montelukast":{"class":"Antiallergy","common":["Headache","Stomach pain","Fatigue"],"serious":["Mental health changes","Suicidal thoughts","Severe allergic reaction"],"risk_base":2,"usual_dose":"10mg"},
    "Fexofenadine":{"class":"Antiallergy","common":["Headache","Nausea","Dizziness"],"serious":["Severe allergic reaction","Heart rhythm problems"],"risk_base":1,"usual_dose":"120mg - 180mg"},
    # STOMACH
    "Omeprazole":{"class":"Gastrointestinal","common":["Headache","Nausea","Diarrhea"],"serious":["Kidney disease","Low magnesium","Bone fractures"],"risk_base":1,"usual_dose":"20mg - 40mg"},
    "Pantoprazole":{"class":"Gastrointestinal","common":["Headache","Diarrhea","Nausea"],"serious":["Kidney inflammation","Low magnesium","C. diff infection"],"risk_base":1,"usual_dose":"40mg"},
    "Ondansetron":{"class":"Gastrointestinal","common":["Headache","Constipation","Fatigue"],"serious":["Heart rhythm problems","Serotonin syndrome","Severe allergic reaction"],"risk_base":2,"usual_dose":"4mg - 8mg"},
    "Domperidone":{"class":"Gastrointestinal","common":["Dry mouth","Headache","Diarrhea"],"serious":["Heart rhythm problems","Sudden cardiac death","Hormonal effects"],"risk_base":2,"usual_dose":"10mg"},
    "Ranitidine":{"class":"Gastrointestinal","common":["Headache","Diarrhea","Nausea"],"serious":["Liver problems","Blood disorders","Kidney problems"],"risk_base":1,"usual_dose":"150mg"},
    "Loperamide":{"class":"Gastrointestinal","common":["Constipation","Dizziness","Nausea"],"serious":["Heart rhythm problems","Toxic megacolon","Ileus"],"risk_base":2,"usual_dose":"2mg"},
    "Lactulose":{"class":"Gastrointestinal","common":["Bloating","Diarrhea","Stomach cramps"],"serious":["Severe electrolyte imbalance","Dehydration"],"risk_base":1,"usual_dose":"15ml - 30ml"},
    "Antacid (Aluminium Hydroxide)":{"class":"Gastrointestinal","common":["Constipation","Chalky taste","Nausea"],"serious":["Phosphate deficiency","Aluminium toxicity in kidney patients"],"risk_base":1,"usual_dose":"400mg - 800mg"},
    "Metoclopramide":{"class":"Gastrointestinal","common":["Drowsiness","Fatigue","Diarrhea"],"serious":["Involuntary movements","Parkinson-like symptoms","Depression"],"risk_base":2,"usual_dose":"10mg"},
    "Bisacodyl":{"class":"Gastrointestinal","common":["Stomach cramps","Diarrhea","Nausea"],"serious":["Severe dehydration","Electrolyte imbalance","Rectal bleeding"],"risk_base":1,"usual_dose":"5mg - 10mg"},
    # HEART BP
    "Amlodipine":{"class":"Antihypertensive","common":["Swollen ankles","Flushing","Headache"],"serious":["Severe low BP","Chest pain","Heart failure"],"risk_base":2,"usual_dose":"5mg - 10mg"},
    "Atorvastatin":{"class":"Antihypertensive","common":["Muscle pain","Joint pain","Diarrhea"],"serious":["Severe muscle breakdown","Liver damage","Memory problems"],"risk_base":2,"usual_dose":"10mg - 40mg"},
    "Lisinopril":{"class":"Antihypertensive","common":["Dry cough","Dizziness","Headache"],"serious":["Angioedema","Kidney failure","High potassium"],"risk_base":2,"usual_dose":"5mg - 20mg"},
    "Ramipril":{"class":"Antihypertensive","common":["Cough","Dizziness","Fatigue"],"serious":["Angioedema","Kidney problems","Low BP"],"risk_base":2,"usual_dose":"2.5mg - 10mg"},
    "Metoprolol":{"class":"Antihypertensive","common":["Fatigue","Dizziness","Cold hands"],"serious":["Severe low heart rate","Heart failure","Depression"],"risk_base":2,"usual_dose":"25mg - 100mg"},
    "Warfarin":{"class":"Antihypertensive","common":["Easy bruising","Bleeding gums","Fatigue"],"serious":["Severe internal bleeding","Brain hemorrhage","Stroke"],"risk_base":3,"usual_dose":"1mg - 10mg"},
    "Clopidogrel":{"class":"Antihypertensive","common":["Bleeding easily","Bruising","Stomach pain"],"serious":["Severe bleeding","Thrombotic thrombocytopenic purpura","Stomach ulcers"],"risk_base":3,"usual_dose":"75mg"},
    "Digoxin":{"class":"Antihypertensive","common":["Nausea","Loss of appetite","Visual disturbances"],"serious":["Heart rhythm problems","Digoxin toxicity","Death at high levels"],"risk_base":3,"usual_dose":"0.125mg - 0.25mg"},
    "Furosemide":{"class":"Antihypertensive","common":["Increased urination","Dizziness","Low potassium"],"serious":["Severe dehydration","Kidney failure","Hearing loss"],"risk_base":2,"usual_dose":"20mg - 80mg"},
    "Spironolactone":{"class":"Antihypertensive","common":["Increased potassium","Dizziness","Breast tenderness"],"serious":["Severe high potassium","Kidney problems","Hormonal effects"],"risk_base":2,"usual_dose":"25mg - 100mg"},
    # DIABETES
    "Metformin":{"class":"Antidiabetic","common":["Nausea","Diarrhea","Stomach pain"],"serious":["Lactic acidosis","Vitamin B12 deficiency","Kidney stress"],"risk_base":2,"usual_dose":"500mg - 1000mg"},
    "Glibenclamide":{"class":"Antidiabetic","common":["Low blood sugar","Nausea","Weight gain"],"serious":["Severe hypoglycemia","Liver damage","Blood disorders"],"risk_base":3,"usual_dose":"2.5mg - 5mg"},
    "Insulin (Regular)":{"class":"Antidiabetic","common":["Low blood sugar","Injection site pain","Weight gain"],"serious":["Severe hypoglycemia","Hypokalemia","Lipodystrophy"],"risk_base":3,"usual_dose":"As prescribed only"},
    "Sitagliptin":{"class":"Antidiabetic","common":["Runny nose","Headache","Stomach pain"],"serious":["Pancreatitis","Kidney problems","Severe joint pain"],"risk_base":2,"usual_dose":"100mg"},
    "Empagliflozin":{"class":"Antidiabetic","common":["Urinary tract infection","Increased urination","Genital infection"],"serious":["Diabetic ketoacidosis","Kidney problems","Low BP"],"risk_base":2,"usual_dose":"10mg - 25mg"},
    "Glipizide":{"class":"Antidiabetic","common":["Low blood sugar","Nausea","Dizziness"],"serious":["Severe hypoglycemia","Liver damage","Blood disorders"],"risk_base":3,"usual_dose":"5mg - 10mg"},
    # MENTAL HEALTH
    "Sertraline":{"class":"Antidepressant","common":["Nausea","Insomnia","Dizziness"],"serious":["Suicidal thoughts in youth","Serotonin syndrome","Bleeding risk"],"risk_base":2,"usual_dose":"50mg - 200mg"},
    "Diazepam":{"class":"Antidepressant","common":["Drowsiness","Dizziness","Fatigue"],"serious":["Addiction","Respiratory depression","Memory impairment"],"risk_base":3,"usual_dose":"2mg - 10mg"},
    "Alprazolam":{"class":"Antidepressant","common":["Drowsiness","Dizziness","Memory issues"],"serious":["Severe addiction","Withdrawal seizures","Respiratory depression"],"risk_base":3,"usual_dose":"0.25mg - 0.5mg"},
    "Melatonin":{"class":"Antidepressant","common":["Drowsiness","Headache","Dizziness"],"serious":["Hormonal effects","Depression worsening","Vivid dreams"],"risk_base":1,"usual_dose":"0.5mg - 5mg"},
    "Fluoxetine":{"class":"Antidepressant","common":["Nausea","Headache","Insomnia"],"serious":["Serotonin syndrome","Suicidal thoughts","Bleeding risk"],"risk_base":2,"usual_dose":"20mg - 60mg"},
    "Haloperidol":{"class":"Antidepressant","common":["Drowsiness","Stiff muscles","Restlessness"],"serious":["Severe movement disorders","Heart rhythm problems","Neuroleptic malignant syndrome"],"risk_base":3,"usual_dose":"0.5mg - 5mg"},
    "Lithium":{"class":"Antidepressant","common":["Tremor","Increased thirst","Frequent urination"],"serious":["Lithium toxicity","Kidney damage","Thyroid problems"],"risk_base":3,"usual_dose":"300mg - 600mg"},
    # SMOKING TOBACCO
    "Nicotine Patch":{"class":"Smoking Cessation","common":["Skin irritation","Vivid dreams","Headache"],"serious":["Heart problems if smoking continues","Severe skin reactions","Nicotine overdose"],"risk_base":1,"usual_dose":"7mg - 21mg patch"},
    "Varenicline (Champix)":{"class":"Smoking Cessation","common":["Nausea","Headache","Vivid dreams","Insomnia"],"serious":["Suicidal thoughts","Severe mood changes","Heart problems"],"risk_base":3,"usual_dose":"0.5mg - 1mg"},
    "Nicotine Gum":{"class":"Smoking Cessation","common":["Jaw pain","Hiccups","Mouth irritation"],"serious":["Nicotine overdose","Heart palpitations","Dependency"],"risk_base":1,"usual_dose":"2mg - 4mg"},
    "Bupropion (Zyban)":{"class":"Smoking Cessation","common":["Dry mouth","Insomnia","Headache","Nausea"],"serious":["Seizures","Suicidal thoughts","Severe allergic reaction"],"risk_base":3,"usual_dose":"150mg"},
    "Nicotine Lozenge":{"class":"Smoking Cessation","common":["Mouth irritation","Hiccups","Nausea"],"serious":["Nicotine overdose","Heart palpitations"],"risk_base":1,"usual_dose":"2mg - 4mg"},
    # ALCOHOL
    "Disulfiram (Antabuse)":{"class":"Alcohol Treatment","common":["Drowsiness","Headache","Metallic taste"],"serious":["Severe reaction with alcohol — vomiting heart palpitations collapse","Liver damage","Nerve damage"],"risk_base":3,"usual_dose":"250mg - 500mg"},
    "Naltrexone":{"class":"Alcohol Treatment","common":["Nausea","Headache","Fatigue","Stomach pain"],"serious":["Liver damage","Severe withdrawal if opioid dependent","Allergic reaction"],"risk_base":2,"usual_dose":"50mg"},
    "Acamprosate":{"class":"Alcohol Treatment","common":["Diarrhea","Nausea","Stomach pain"],"serious":["Kidney problems","Suicidal thoughts","Severe allergic reaction"],"risk_base":2,"usual_dose":"666mg"},
    "Thiamine (Vitamin B1 for alcohol)":{"class":"Alcohol Treatment","common":["Mild nausea","Restlessness"],"serious":["Severe allergic reaction (injection form)"],"risk_base":1,"usual_dose":"100mg - 300mg"},
    # STROKE
    "Alteplase (tPA)":{"class":"Stroke Medicine","common":["Bleeding at injection site","Bruising","Fever"],"serious":["Brain bleeding","Internal bleeding","Severe allergic reaction"],"risk_base":3,"usual_dose":"Hospital use only — IV"},
    "Clopidogrel":{"class":"Stroke Medicine","common":["Bleeding easily","Bruising","Stomach pain"],"serious":["Severe bleeding","Blood clotting disorder","Stomach ulcers"],"risk_base":3,"usual_dose":"75mg"},
    "Rivaroxaban":{"class":"Stroke Medicine","common":["Bleeding","Nausea","Anemia"],"serious":["Severe uncontrolled bleeding","Spinal bleeding","Liver problems"],"risk_base":3,"usual_dose":"10mg - 20mg"},
    "Dabigatran":{"class":"Stroke Medicine","common":["Stomach upset","Bleeding","Nausea"],"serious":["Severe bleeding","Kidney problems","Heart attack"],"risk_base":3,"usual_dose":"110mg - 150mg"},
    "Atorvastatin (Stroke)":{"class":"Stroke Medicine","common":["Muscle pain","Joint pain","Diarrhea"],"serious":["Severe muscle breakdown","Liver damage","Memory problems"],"risk_base":2,"usual_dose":"40mg - 80mg"},
    # VITAMINS SUPPLEMENTS
    "Vitamin C":{"class":"Supplement","common":["Stomach upset","Diarrhea","Nausea"],"serious":["Kidney stones at high dose","Iron overload"],"risk_base":1,"usual_dose":"500mg - 1000mg"},
    "Vitamin D3":{"class":"Supplement","common":["Nausea","Constipation","Fatigue"],"serious":["Calcium toxicity","Kidney damage at high dose"],"risk_base":1,"usual_dose":"1000 IU - 4000 IU"},
    "Iron Supplement":{"class":"Supplement","common":["Constipation","Stomach pain","Dark stools"],"serious":["Iron toxicity","Liver damage","Stomach bleeding"],"risk_base":1,"usual_dose":"65mg - 200mg"},
    "Calcium Supplement":{"class":"Supplement","common":["Constipation","Bloating","Gas"],"serious":["Kidney stones","High blood calcium","Heart problems"],"risk_base":1,"usual_dose":"500mg - 1000mg"},
    "Zinc Supplement":{"class":"Supplement","common":["Nausea","Stomach upset","Metallic taste"],"serious":["Copper deficiency","Immune suppression at high dose"],"risk_base":1,"usual_dose":"8mg - 11mg"},
    "Folic Acid":{"class":"Supplement","common":["Nausea","Bloating","Loss of appetite"],"serious":["Masks B12 deficiency","Seizure risk in high doses"],"risk_base":1,"usual_dose":"400mcg - 5mg"},
    # RESPIRATORY
    "Montelukast (Asthma)":{"class":"Respiratory","common":["Headache","Stomach pain","Fatigue"],"serious":["Mental health changes","Suicidal thoughts","Severe allergic reaction"],"risk_base":2,"usual_dose":"10mg"},
    "Budesonide (Inhaler)":{"class":"Respiratory","common":["Mouth thrush","Hoarse voice","Cough"],"serious":["Adrenal suppression","Bone loss","Immune suppression"],"risk_base":2,"usual_dose":"200mcg - 400mcg"},
    "Tiotropium":{"class":"Respiratory","common":["Dry mouth","Constipation","Urinary difficulty"],"serious":["Urinary retention","Glaucoma worsening","Severe allergic reaction"],"risk_base":2,"usual_dose":"18mcg"},
    "Theophylline":{"class":"Respiratory","common":["Nausea","Headache","Insomnia","Tremors"],"serious":["Seizures","Heart rhythm problems","Toxicity — narrow safety window"],"risk_base":3,"usual_dose":"100mg - 300mg"},
    # BONE JOINT
    "Alendronate":{"class":"Bone Medicine","common":["Stomach pain","Heartburn","Joint pain"],"serious":["Esophageal ulcers","Jaw bone damage","Unusual thigh fractures"],"risk_base":2,"usual_dose":"70mg weekly"},
    "Calcium + Vitamin D":{"class":"Bone Medicine","common":["Constipation","Bloating","Gas"],"serious":["Kidney stones","High calcium levels"],"risk_base":1,"usual_dose":"500mg + 400 IU"},
    "Colchicine (Gout)":{"class":"Bone Medicine","common":["Diarrhea","Nausea","Stomach pain"],"serious":["Muscle damage","Nerve damage","Blood disorders"],"risk_base":2,"usual_dose":"0.5mg - 1mg"},
    "Allopurinol":{"class":"Bone Medicine","common":["Rash","Nausea","Drowsiness"],"serious":["Severe skin reactions","Liver damage","Kidney damage"],"risk_base":2,"usual_dose":"100mg - 300mg"},
    # EYE
    "Latanoprost (Eye drops)":{"class":"Eye Medicine","common":["Eye redness","Iris color change","Eye irritation"],"serious":["Macular edema","Severe eye inflammation","Vision changes"],"risk_base":1,"usual_dose":"1 drop daily"},
    "Timolol (Eye drops)":{"class":"Eye Medicine","common":["Eye irritation","Blurred vision","Dry eyes"],"serious":["Severe breathing problems in asthma","Low heart rate","Low BP"],"risk_base":2,"usual_dose":"1 drop twice daily"},
    "Chloramphenicol (Eye drops)":{"class":"Eye Medicine","common":["Eye stinging","Temporary blurred vision"],"serious":["Aplastic anemia (very rare)","Severe allergic reaction"],"risk_base":1,"usual_dose":"1 drop every 2 hours"},
}

DRUG_CLASS_MAP = {"Painkiller":0,"Antibiotic":1,"Antiallergy":2,"Gastrointestinal":3,"Antihypertensive":4,"Antidiabetic":5,"Antidepressant":6,"Supplement":7,"Smoking Cessation":8,"Alcohol Treatment":9,"Stroke Medicine":10,"Respiratory":11,"Bone Medicine":12,"Eye Medicine":13}

CATEGORIES = {
    "💊 Painkillers & Fever":["Paracetamol (Acetaminophen)","Ibuprofen","Aspirin","Diclofenac","Tramadol","Naproxen","Codeine","Mefenamic Acid","Ketorolac","Morphine"],
    "🦠 Antibiotics":["Amoxicillin","Ciprofloxacin","Azithromycin","Doxycycline","Metronidazole","Clindamycin","Cephalexin","Erythromycin","Trimethoprim","Nitrofurantoin","Ampicillin","Levofloxacin"],
    "🤧 Cold, Cough & Allergy":["Cetirizine","Loratadine","Chlorpheniramine","Phenylephrine","Dextromethorphan","Bromhexine","Salbutamol","Montelukast","Fexofenadine"],
    "🫃 Stomach & Digestion":["Omeprazole","Pantoprazole","Ondansetron","Domperidone","Ranitidine","Loperamide","Lactulose","Antacid (Aluminium Hydroxide)","Metoclopramide","Bisacodyl"],
    "❤️ Heart & Blood Pressure":["Amlodipine","Atorvastatin","Lisinopril","Ramipril","Metoprolol","Warfarin","Clopidogrel","Digoxin","Furosemide","Spironolactone"],
    "🩸 Diabetes":["Metformin","Glibenclamide","Insulin (Regular)","Sitagliptin","Empagliflozin","Glipizide"],
    "🧠 Mental Health & Sleep":["Sertraline","Diazepam","Alprazolam","Melatonin","Fluoxetine","Haloperidol","Lithium"],
    "🚬 Smoking & Tobacco":["Nicotine Patch","Varenicline (Champix)","Nicotine Gum","Bupropion (Zyban)","Nicotine Lozenge"],
    "🍺 Alcohol Treatment":["Disulfiram (Antabuse)","Naltrexone","Acamprosate","Thiamine (Vitamin B1 for alcohol)"],
    "🧠 Stroke Medicines":["Alteplase (tPA)","Clopidogrel","Rivaroxaban","Dabigatran","Atorvastatin (Stroke)"],
    "🌿 Vitamins & Supplements":["Vitamin C","Vitamin D3","Iron Supplement","Calcium Supplement","Zinc Supplement","Folic Acid"],
    "🫁 Respiratory & Asthma":["Montelukast (Asthma)","Budesonide (Inhaler)","Tiotropium","Theophylline"],
    "🦴 Bone & Joint":["Alendronate","Calcium + Vitamin D","Colchicine (Gout)","Allopurinol"],
    "👁️ Eye Medicines":["Latanoprost (Eye drops)","Timolol (Eye drops)","Chloramphenicol (Eye drops)"],
}

# ══════════════════════════════════════════════════════════════════════════════
# AI DOCTOR — SMART RULE BASED
# ══════════════════════════════════════════════════════════════════════════════
def ask_ai_doctor(question, lang="English"):
    q = question.lower().strip()
    q = re.sub(r'[^\w\s]', ' ', q)

    # Fix spelling mistakes and short forms
    fixes = {"nicotin":"nicotine","nicoten":"nicotine","smokng":"smoking","smokin":"smoking","paracetmol":"paracetamol","parcetamol":"paracetamol","ibuprofin":"ibuprofen","alchohol":"alcohol","alcohal":"alcohol","booze":"alcohol","cigaret":"cigarette","tabaco":"tobacco","tobaco":"tobacco","strok":"stroke","pregnan":"pregnant","diabet":"diabetes","diabetis":"diabetes","kidny":"kidney","kideny":"kidney","lever":"liver","hart":"heart","faver":"fever","medcine":"medicine","emergancy":"emergency","childern":"children","grandma":"elderly","grandpa":"elderly"}
    for wrong,right in fixes.items():
        q = q.replace(wrong, right)

    # Greetings
    greetings = ["hi","hello","hey","hii","helo","hai","vanakkam","நல்லது","good morning","good evening","sup","wassup","yo"]
    if any(g in q.split() for g in greetings) and len(q.split()) < 5:
        if lang=="தமிழ்":
            return "வணக்கம்! நான் உங்கள் AI மருத்துவர். மருந்துகள், பக்க விளைவுகள் அல்லது மருந்து பாதுகாப்பு பற்றி எந்த கேள்வியும் கேளுங்கள்! 😊"
        return "Hello! I'm your AI Doctor 👨‍⚕️ Ask me anything about medicines, side effects, or drug safety — I'm here to help! 😊"

    # Thanks
    if any(w in q for w in ["thank","thanks","thank you","நன்றி","thx","ty"]):
        if lang=="தமிழ்":
            return "மகிழ்ச்சியாக இருக்கிறேன்! உங்கள் ஆரோக்கியம் எனது முக்கிய அக்கறை. வேறு ஏதாவது கேள்வி இருந்தால் கேளுங்கள்! 😊"
        return "You're welcome! Your health is my priority. Feel free to ask anything else! 😊"

    # Funny / dumb questions handled smartly
    food_words = ["biriyani","biryani","food","eat","rice","chapati","dosa","idli","coffee","tea","milk","juice","water","drink"]
    if any(f in q for f in food_words) and any(m in q for m in ["medicine","tablet","paracetamol","ibuprofen","drug","pill"]):
        if lang=="தமிழ்":
            return "நல்ல கேள்வி! பெரும்பாலான மாத்திரைகளை உணவுடன் எடுப்பது நல்லது — குறிப்பாக வயிற்று எரிச்சல் தடுக்க. ஆனால் சில மாத்திரைகளை வெறும் வயிற்றில் எடுக்க வேண்டும். உங்கள் மருந்தின் லேபிளை பாருங்கள் அல்லது மருத்துவரிடம் கேளுங்கள்! நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Good question! Most medicines are actually BETTER taken with food — it protects your stomach. However some medicines need to be taken on an empty stomach. Always check the label or ask your pharmacist! Taking medicine with a full meal like biryani is totally fine for most tablets 😄 Remember: Always consult a real doctor before taking any medicine."

    # Too many tablets
    overdose_hints = ["10 tablet","10 pill","too many","too much","lot of tablet","many tablet","20 tablet","whole bottle","full strip","overdose","ate too many","swallowed"]
    if any(o in q for o in overdose_hints):
        if lang=="தமிழ்":
            return "⚠️ இது மிக முக்கியமான நிலை! அதிக மாத்திரைகள் எடுத்திருந்தால் உடனே 108 (இந்தியா) என்ற எண்ணில் அழைக்கவும். வாந்தி எடுக்க வைக்க முயற்சிக்காதீர்கள். மருந்து பாட்டிலை எடுத்துக்கொண்டு உடனே மருத்துவமனை செல்லுங்கள். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "⚠️ THIS IS SERIOUS! If someone has taken too many tablets, call emergency services IMMEDIATELY — 108 in India. Do NOT try to make them vomit. Take the medicine bottle with you to the hospital. Every second matters — please act fast! Remember: Always consult a real doctor before taking any medicine."

    # Smoking related
    smoke_words = ["smoke","smoking","smokin","cigarette","tobacco","nicotine","nicotin","nicoti","nico","bidi","hookah","vape","vaping","quit smoking","stop smoking","புகை","புகைபிடி","chewing tobacco","gutka","beedi"]
    if any(s in q for s in smoke_words):
        if lang=="தமிழ்":
            return "புகைபிடிப்பை விட சிறந்த முடிவு இல்லை! 🚭 நிறுத்த உதவும் மருந்துகள்: நிகோடின் பேட்ச், நிகோடின் கம், Varenicline (Champix), Bupropion (Zyban). இந்த மருந்துகள் மருத்துவரின் ஆலோசனையுடன் மட்டுமே எடுக்கவும். புகை பிடிக்கும்போது மருந்து எடுப்பது இதயத்தை பாதிக்கும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Quitting smoking is one of the BEST decisions you can make! 🚭 Medicines that help: Nicotine Patch, Nicotine Gum, Varenicline (Champix), Bupropion (Zyban). These work by reducing cravings and withdrawal symptoms. IMPORTANT — never smoke while using nicotine replacement therapy as it causes nicotine overdose. Always get a doctor's guidance before starting. Remember: Always consult a real doctor before taking any medicine."

    # Alcohol related
    alcohol_words = ["alcohol","drink","drinking","beer","wine","whiskey","rum","liquor","drunk","arrack","toddy","குடி","மது","alcohol medicine","stop drinking","quit alcohol"]
    if any(a in q for a in alcohol_words):
        if lang=="தமிழ்":
            return "மது மருந்துடன் கலந்தால் மிகவும் ஆபத்தானது! 🍺⚠️ குறிப்பாக: Paracetamol + மது = கல்லீரல் சேதம், Metronidazole + மது = கடுமையான வாந்தி மற்றும் இதய பிரச்சனை, Diazepam + மது = சுவாசம் நிற்கும் ஆபத்து. மது குடிப்பதை நிறுத்த உதவும் மருந்துகள்: Disulfiram, Naltrexone, Acamprosate. இவை மருத்துவரின் ஆலோசனையுடன் மட்டுமே எடுக்கவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Mixing alcohol with medicines is VERY DANGEROUS! 🍺⚠️ Critical combinations to NEVER do: Paracetamol + alcohol = serious liver damage, Metronidazole + alcohol = violent vomiting and heart problems, Diazepam/Alprazolam + alcohol = can stop breathing (potentially fatal), Warfarin + alcohol = dangerous bleeding risk. For quitting alcohol, medicines like Disulfiram, Naltrexone and Acamprosate can help — but ONLY under doctor supervision. Remember: Always consult a real doctor before taking any medicine."

    # Stroke related
    stroke_words = ["stroke","paralysis","brain attack","clot","blood clot","brain bleed","sudden weakness","face drooping","speech problem","பக்கவாதம்","மூளை"]
    if any(s in q for s in stroke_words):
        if lang=="தமிழ்":
            return "பக்கவாதம் மிக அவசரமான நிலை! 🧠⚠️ FAST அறிகுறிகள் கவனியுங்கள்: F-முகம் கோணல், A-கை தொங்குதல், S-பேச்சு தெளிவற்றது, T-உடனே 108 அழைக்கவும்! பக்கவாதம் வந்த 4.5 மணி நேரத்தில் சிகிச்சை கொடுத்தால் குணமாகலாம். மருந்துகள்: Alteplase, Clopidogrel, Rivaroxaban — இவை மருத்துவமனையில் மட்டுமே கொடுக்கப்படும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Stroke is a MEDICAL EMERGENCY — every minute counts! 🧠⚠️ Remember FAST: F = Face drooping, A = Arm weakness, S = Speech difficulty, T = Time to call 108 immediately! Treatment with Alteplase must happen within 4.5 hours of stroke onset. Medicines used: Alteplase (hospital only), Clopidogrel, Rivaroxaban, Dabigatran — ALL require doctor prescription. After stroke, blood thinners and statins are often prescribed to prevent another stroke. Remember: Always consult a real doctor before taking any medicine."

    # Pregnancy related
    preg_words = ["pregnant","pregnancy","baby","unborn","கர்ப்ப","கர்ப்பிணி","fetus","trimester","expecting"]
    if any(p in q for p in preg_words):
        if lang=="தமிழ்":
            return "கர்ப்பகாலத்தில் மருந்து எடுக்கும்போது மிக கவனமாக இருக்கவும்! 🤰 பாதுகாப்பானவை (மருத்துவர் ஆலோசனையுடன்): Paracetamol. தவிர்க்க வேண்டியவை: Ibuprofen, Aspirin (அதிக அளவு), Tramadol, Codeine, Warfarin, பெரும்பாலான ஆண்டிபயாடிக்குகள். முதல் 3 மாதங்கள் மிக முக்கியம் — கரு உருவாகும் நேரம். எந்த மருந்தும் மருத்துவர் ஆலோசனை இல்லாமல் எடுக்காதீர்கள். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Pregnancy requires EXTRA caution with medicines! 🤰 Generally safer (with doctor advice): Paracetamol, Folic Acid, Iron supplements. AVOID during pregnancy: Ibuprofen (especially after 30 weeks), Aspirin (high dose), Tramadol, Codeine, Warfarin, most antibiotics. The first trimester is most critical as the baby's organs are forming. NEVER take any medicine during pregnancy without consulting your gynecologist first. Remember: Always consult a real doctor before taking any medicine."

    # Children related
    child_words = ["child","baby","infant","kid","children","toddler","குழந்தை","பச்சிளம்","years old","month old","year old","2 year","3 year","5 year","newborn"]
    if any(c in q for c in child_words):
        if lang=="தமிழ்":
            return "குழந்தைகளுக்கு மருந்து மிக கவனமாக கொடுக்க வேண்டும்! 👶 முக்கியமான விஷயங்கள்: அளவு எப்போதும் எடை அல்லது வயதை பொறுத்து இருக்கும். Aspirin 16 வயதுக்கு கீழ் கொடுக்கவே கூடாது. Ibuprofen 6 மாதத்திற்கு கீழ் கொடுக்கக்கூடாது. பெரியவர் மாத்திரை குழந்தைக்கு கொடுக்காதீர்கள். காய்ச்சலுக்கு Paracetamol குழந்தை சிரப் பயன்படுத்துங்கள். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Children need SPECIAL care with medicines! 👶 Key rules: Doses are always based on weight or age — never give adult doses. NEVER give Aspirin to children under 16 — it can cause dangerous Reye's syndrome. Ibuprofen not recommended under 6 months. For fever, use children's Paracetamol syrup at correct weight-based dose. NEVER break adult tablets for children without medical advice. When in doubt, always call your pediatrician. Remember: Always consult a real doctor before taking any medicine."

    # Elderly
    elderly_words = ["old","elderly","grandma","grandpa","grandfather","grandmother","aged","senior","முதியவர்","பாட்டி","தாத்தா","70","80","90 year"]
    if any(e in q for e in elderly_words):
        if lang=="தமிழ்":
            return "முதியவர்களுக்கு மருந்து கவனமாக கொடுக்க வேண்டும்! 👴 காரணம்: சிறுநீரகம் மற்றும் கல்லீரல் மெதுவாக செயல்படும், மருந்து உடலில் அதிக நேரம் தங்கும். அதிக மருந்துகள் ஒன்றாக எடுப்பதால் பக்க விளைவுகள் அதிகமாகும். தலைச்சுற்றல் விழுவதற்கு வழிவகுக்கலாம். மருத்துவரிடம் அனைத்து மருந்துகளையும் பட்டியலிடுங்கள். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Elderly patients need CAREFUL medication management! 👴 Why? Kidneys and liver slow down with age so medicines stay longer in the body causing stronger effects. They're often on multiple medicines increasing interaction risk. Dizziness from medicines is a major fall risk. Tips: Keep a list of ALL medicines taken, attend regular checkups, start with lower doses, never stop medicines suddenly. Remember: Always consult a real doctor before taking any medicine."

    # Kidney
    kidney_words = ["kidney","renal","சிறுநீரக","dialysis","kidney disease","kidney problem","kidney failure"]
    if any(k in q for k in kidney_words):
        if lang=="தமிழ்":
            return "சிறுநீரக நோயாளிகளுக்கு மிக கவனமாக இருக்க வேண்டும்! 🫘 தவிர்க்க வேண்டிய மருந்துகள்: Ibuprofen, Naproxen, Diclofenac (சிறுநீரகத்தை மேலும் பாதிக்கும்). Metformin (சிறுநீரக செயலிழப்பில் தடைசெய்யப்பட்டது). பாதுகாப்பானவை: Paracetamol (சரியான அளவில்). எல்லா மருந்துகளும் மருத்துவர் ஆலோசனையுடன் மட்டுமே எடுக்கவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Kidney disease patients need EXTRA caution! 🫘 AVOID these medicines — they worsen kidney function: Ibuprofen, Naproxen, Diclofenac (NSAIDs damage kidneys), Metformin (banned in kidney failure), Certain antibiotics like Nitrofurantoin. SAFER options: Paracetamol at recommended doses is generally safer. Always tell EVERY doctor about your kidney condition. Get regular kidney function tests. Remember: Always consult a real doctor before taking any medicine."

    # Liver
    liver_words = ["liver","hepatic","கல்லீரல்","jaundice","hepatitis","liver disease","liver problem","liver failure"]
    if any(l in q for l in liver_words):
        if lang=="தமிழ்":
            return "கல்லீரல் நோயாளிகளுக்கு மருந்து மிக ஆபத்தானது! 🫀 தவிர்க்க வேண்டியவை: அதிக அளவு Paracetamol (4g/day க்கு மேல் கூடாது), மது + எந்த மருந்தும், Statins கவனமாக எடுக்கவும், Metronidazole அதிக அளவு. மதுபானம் முற்றிலும் தவிர்க்கவும். கல்லீரல் செயல்பாட்டை தொடர்ந்து சோதிக்கவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Liver disease patients must be very careful with medicines! 🫀 AVOID: High dose Paracetamol (never exceed 2g/day with liver disease), ALL alcohol (even small amounts cause serious damage), Statins need careful monitoring, Metronidazole in high doses. The liver processes most medicines — damaged liver means medicines build up to dangerous levels. NEVER start or stop any medicine without your hepatologist's advice. Regular liver function tests are essential. Remember: Always consult a real doctor before taking any medicine."

    # Heart
    heart_words = ["heart","cardiac","இதய","heart attack","chest pain","palpitation","heart disease","cardiovascular","heartbeat"]
    if any(h in q for h in heart_words):
        if lang=="தமிழ்":
            return "இதய நோயாளிகளுக்கு மருந்து மிக முக்கியமானது! ❤️ தவிர்க்க வேண்டியவை: Ibuprofen, Diclofenac (மாரடைப்பு அபாயம் அதிகரிக்கும்), Naproxen. கவனமாக எடுக்க வேண்டியவை: Decongestants (BP ஐ உயர்த்தும்). இதய மருந்துகளை திடீரென நிறுத்தாதீர்கள் — மிக ஆபத்தானது. மார்பு வலி வந்தால் உடனே 108 அழைக்கவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Heart patients need CAREFUL medication choices! ❤️ AVOID: Ibuprofen, Diclofenac, Naproxen — these NSAIDs significantly increase heart attack and stroke risk. Decongestants like Phenylephrine raise blood pressure. NEVER stop heart medicines suddenly — it can trigger heart attack. If you feel chest pain, jaw pain, left arm pain, or sudden breathlessness — call 108 IMMEDIATELY, don't wait. Remember: Always consult a real doctor before taking any medicine."

    # Diabetes
    diabetes_words = ["diabetes","diabetic","sugar","blood sugar","insulin","நீரிழிவு","glucose","hypoglycemia","low sugar","high sugar"]
    if any(d in q for d in diabetes_words):
        if lang=="தமிழ்":
            return "நீரிழிவு நோயாளிகளுக்கு சில மருந்துகள் ரத்த சர்க்கரையை பாதிக்கும்! 🩸 ரத்த சர்க்கரையை உயர்த்தும் மருந்துகள்: Steroids, Beta blockers, Decongestants. ரத்த சர்க்கரையை குறைக்கும் மருந்துகள் (hypoglycemia): Insulin, Glibenclamide. Insulin எடுக்கும்போது அறிகுறிகள் கவனியுங்கள்: வியர்த்தல், நடுக்கம், தலைச்சுற்றல் = உடனே சர்க்கரை அல்லது juice எடுங்கள். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Diabetic patients must watch out for medicine interactions! 🩸 Medicines that RAISE blood sugar: Steroids, certain antipsychotics, decongestants. Medicines that cause dangerous LOW blood sugar: Insulin, Glibenclamide, Glipizide — always carry a glucose tablet. Signs of low blood sugar: sweating, trembling, dizziness, confusion — eat sugar immediately! Metformin is generally safe but must be stopped before contrast dye procedures. Remember: Always consult a real doctor before taking any medicine."

    # Fever
    fever_words = ["fever","temperature","hot","காய்ச்சல்","pyrexia","high temperature","body heat"]
    if any(f in q for f in fever_words):
        if lang=="தமிழ்":
            return "காய்ச்சலுக்கு சரியான சிகிச்சை: 🌡️ Paracetamol (500mg - 1000mg, 6-8 மணி நேர இடைவெளியில்) முதல் தேர்வு. நிறைய தண்ணீர் குடியுங்கள், ஓய்வு எடுங்கள். உடலை குளிர்ந்த துணியால் துடையுங்கள். எப்போது மருத்துவரை அணுக வேண்டும்: 39.4°C (103°F) க்கு மேல் இருந்தால், 3 நாட்களுக்கு மேல் நீடித்தால், தலைவலி + கழுத்து கடினம் + தடிப்பு இருந்தால் — உடனே மருத்துவரை அணுகவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "For fever, here's what to do! 🌡️ Best medicine: Paracetamol (500mg-1000mg every 6-8 hours, max 4g per day for adults). Stay well hydrated, rest, use cool wet cloth on forehead. See a doctor IMMEDIATELY if: Fever above 39.4°C (103°F), fever lasts more than 3 days, fever with severe headache + stiff neck + rash (could be meningitis!), fever with difficulty breathing. Don't use Aspirin for fever in children EVER. Remember: Always consult a real doctor before taking any medicine."

    # Allergy reactions
    allergy_words = ["allergy","allergic","rash","hives","swelling","itching","ஒவ்வாமை","anaphylaxis","reaction to medicine"]
    if any(a in q for a in allergy_words):
        if lang=="தமிழ்":
            return "மருந்து ஒவ்வாமை மிக முக்கியமான விஷயம்! ⚠️ லேசான அறிகுறிகள்: தோல் தடிப்பு, அரிப்பு, சிவத்தல். கடுமையான அறிகுறிகள் (anaphylaxis) — உடனே 108 அழைக்கவும்: முகம் வீக்கம், தொண்டை வீக்கம், சுவாசிக்க கஷ்டம், திடீர் BP குறைவு. ஒவ்வாமை உள்ள மருந்தை உடனே நிறுத்துங்கள். ஒவ்வாமை வரலாற்றை எல்லா மருத்துவர்களிடமும் சொல்லுங்கள். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "Drug allergies can range from mild to life-threatening! ⚠️ Mild symptoms: Rash, hives, itching — stop the medicine and call your doctor. EMERGENCY symptoms (call 108 immediately!): Face/throat swelling, difficulty breathing, sudden drop in blood pressure, loss of consciousness. This is anaphylaxis — a life-threatening emergency! Always tell EVERY doctor and dentist about any medicine allergy you have. Wear a medical alert bracelet if you have severe allergies. Remember: Always consult a real doctor before taking any medicine."

    # Emergency signs
    emergency_words = ["emergency","urgent","serious","danger","hospital","dying","cant breathe","not breathing","unconscious","collapsed","108","ambulance","help"]
    if any(e in q for e in emergency_words):
        if lang=="தமிழ்":
            return "🚨 அவசரநிலை! உடனே 108 அழைக்கவும்! இந்த அறிகுறிகள் இருந்தால் உடனே மருத்துவமனை செல்லுங்கள்: மார்பு வலி, சுவாசிக்க கஷ்டம், திடீர் உடல் ஒரு பக்கம் செயலிழத்தல், வாய் கோணல், சுயநினைவு இழத்தல், கடுமையான தலைவலி. தாமதிக்காதீர்கள் — உயிர் முக்கியம்! நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "🚨 EMERGENCY! Call 108 immediately if you see: Chest pain or pressure, difficulty breathing, sudden weakness of face/arm/leg on one side, slurred speech, loss of consciousness, severe uncontrolled bleeding, suspected poisoning or overdose. Do NOT drive yourself to the hospital in an emergency — call 108. Time is life — don't delay! Remember: Always consult a real doctor before taking any medicine."

    # Periods / menstrual
    period_words = ["period","periods","menstrual","menstruation","cramps","period pain","monthly","menses","pms","irregular period","heavy bleeding","மாதவிடாய்","period cramp","dysmenorrhea"]
    if any(p in q for p in period_words):
        if lang=="தமிழ்":
            return "மாதவிடாய் வலிக்கு (Dysmenorrhea) உதவும் மருந்துகள்: 🩸 Mefenamic Acid (250mg-500mg) — மிகவும் பயனுள்ளது, வலி தொடங்கும் முன்பே எடுக்கவும். Ibuprofen (200mg-400mg) — வலி மற்றும் வீக்கத்தை குறைக்கும். Paracetamol — லேசான வலிக்கு. வெதுவெதுப்பான தண்ணீர் பை வயிற்றில் வைக்கவும். நிறைய தண்ணீர் குடியுங்கள், ஓய்வு எடுங்கள். கடுமையான வலி, அதிக ரத்தப்போக்கு அல்லது 7 நாட்களுக்கு மேல் நீடித்தால் மருத்துவரை அணுகவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "For period pain (Dysmenorrhea), here's what helps! 🩸 Best medicines: Mefenamic Acid (250mg-500mg) — most effective for period cramps, take at start of pain. Ibuprofen (200mg-400mg) — reduces pain and inflammation, take with food. Paracetamol — for mild pain. Non-medicine tips: Hot water bottle on abdomen, stay hydrated, gentle exercise helps. See a doctor if: pain is severely unbearable, bleeding is very heavy (soaking more than 1 pad per hour), periods are irregular, or pain lasts more than 7 days — could indicate endometriosis or PCOS. Remember: Always consult a real doctor before taking any medicine."

    # Muscle pain body ache
    muscle_words = ["muscle","body ache","bodyache","muscle pain","body pain","sore","soreness","தசை வலி","தசை","ache","sprain","cramp","stiff"]
    if any(m in q for m in muscle_words):
        if lang=="தமிழ்":
            return "தசை வலி மற்றும் உடல் வலிக்கு: 💪 Paracetamol — பாதுகாப்பான முதல் தேர்வு. Ibuprofen அல்லது Diclofenac — வீக்கம் சம்பந்தமான வலிக்கு (உணவுடன் எடுக்கவும்). வெதுவெதுப்பான ஒத்தடம் — தசை பிடிப்பிற்கு. குளிர் ஒத்தடம் — சுளுக்கிற்கு (முதல் 24 மணி நேரம்). ஓய்வு எடுங்கள், நீட்சி உடற்பயிற்சி செய்யுங்கள். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "For muscle pain and body aches! 💪 Best medicines: Paracetamol — safest first choice for general body ache. Ibuprofen or Diclofenac — better for inflammation-related muscle pain, always take with food. Diclofenac gel — apply directly on sore muscle, very effective. Non-medicine tips: Heat pack for muscle spasms, ice pack for sprains (first 24 hours), rest and gentle stretching. See a doctor if pain is severe, follows an injury, or lasts more than a week. Remember: Always consult a real doctor before taking any medicine."

    # Sleep problems
    sleep_words = ["sleep","insomnia","cant sleep","sleepless","தூக்கம்","தூக்கமின்மை","sleeping problem","sleep issue","not sleeping","sleep medicine"]
    if any(s in q for s in sleep_words):
        if lang=="தமிழ்":
            return "தூக்கமின்மைக்கு உதவும் விஷயங்கள்: 😴 மருந்துகள்: Melatonin (0.5mg-5mg) — இயற்கையான தூக்க ஹார்மோன், பாதுகாப்பானது. Diazepam, Alprazolam — மருத்துவர் ஆலோசனையுடன் மட்டுமே (அடிமையாதல் அபாயம்). மருந்தில்லாத வழிகள்: தொலைபேசி பயன்பாட்டை படுக்கை நேரத்திற்கு 1 மணி நேரம் முன்பு நிறுத்துங்கள், ஒரே நேரத்தில் தூங்குங்கள், காஃபின் தவிர்க்கவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "For sleep problems and insomnia! 😴 Safe medicines: Melatonin (0.5mg-5mg) — natural sleep hormone, start with lowest dose, take 30 mins before bed. AVOID without doctor: Diazepam, Alprazolam — highly addictive, only for short-term use under strict medical supervision. Non-medicine tips that actually work: Stop phone/screen 1 hour before bed, sleep and wake at same time daily, avoid caffeine after 3pm, keep room cool and dark, avoid heavy meals at night. If sleep problems persist more than 4 weeks, see a doctor. Remember: Always consult a real doctor before taking any medicine."

    # Nausea vomiting
    nausea_words = ["nausea","vomit","vomiting","nauseous","குமட்டல்","வாந்தி","feel sick","throwing up","motion sickness","travel sick"]
    if any(n in q for n in nausea_words):
        if lang=="தமிழ்":
            return "குமட்டல் மற்றும் வாந்திக்கு: 🤢 Ondansetron (4mg) — மிகவும் பயனுள்ளது. Domperidone (10mg) — சாப்பிடுவதற்கு 30 நிமிடம் முன்பு எடுக்கவும். Metoclopramide — கடுமையான குமட்டலுக்கு. வீட்டு வைத்தியம்: இஞ்சி தேநீர், எலுமிச்சை வாசம், குளிர்ந்த தண்ணீர், சிறிய அளவில் சாப்பிடுங்கள். 24 மணி நேரத்திற்கு மேல் வாந்தி நீடித்தால், இரத்தம் வந்தால் உடனே மருத்துவரை அணுகவும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "For nausea and vomiting! 🤢 Best medicines: Ondansetron (4mg) — very effective, dissolves under tongue. Domperidone (10mg) — take 30 minutes before meals. Metoclopramide — for severe nausea. For motion sickness: take medicine 30 minutes before travel. Home remedies that help: Ginger tea, lemon smell, cold water sips, eat small amounts frequently. See a doctor if: vomiting lasts more than 24 hours, there is blood in vomit, you are severely dehydrated, or you have severe stomach pain. Remember: Always consult a real doctor before taking any medicine."

    # Tooth pain
    tooth_words = ["tooth","teeth","toothache","dental","tooth pain","gum","பல்","பல் வலி","cavity","tooth ache"]
    if any(t in q for t in tooth_words):
        if lang=="தமிழ்":
            return "பல் வலிக்கு: 🦷 Paracetamol (500mg-1000mg) — பாதுகாப்பான முதல் தேர்வு. Ibuprofen (400mg) — வீக்கம் இருந்தால் நல்லது. Clove oil — பஞ்சில் நனைத்து பல்லில் வையுங்கள் — இயற்கையான வலி நிவாரணி. Benzocaine gel — மரப்பு மருந்து. பல் வலி என்பது ஒரு அறிகுறி மட்டுமே — உடனே பல் மருத்துவரை (dentist) அணுகவும். மருந்து வலியை மட்டுமே குறைக்கும், காரணத்தை சரிசெய்யாது. நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "For tooth pain! 🦷 Medicines for relief: Paracetamol (500mg-1000mg) — safest first choice. Ibuprofen (400mg) — better if there is swelling or infection. Clove oil — soak cotton and apply on tooth — natural numbing effect that actually works! Benzocaine gel — numbing gel from pharmacy. IMPORTANT — tooth pain is just a symptom! Always see a dentist — medicine only masks the pain, it does not fix the problem. Ignoring tooth pain can lead to serious infection. Remember: Always consult a real doctor before taking any medicine."

    # Cold and flu
    cold_words = ["cold","flu","running nose","runny nose","blocked nose","sneezing","sore throat","cough","congestion","சளி","இருமல்","தொண்டை வலி","throat pain","nasal"]
    if any(c in q for c in cold_words):
        if lang=="தமிழ்":
            return "சளி மற்றும் காய்ச்சலுக்கு: 🤧 மருந்துகள்: Paracetamol — காய்ச்சல் மற்றும் தலைவலிக்கு. Cetirizine அல்லது Loratadine — மூக்கு ஒழுகுதல் மற்றும் தும்மலுக்கு. Bromhexine — சளியை கரைக்க. Lozenges — தொண்டை வலிக்கு. வீட்டு வைத்தியம்: ஆவி பிடியுங்கள், தேன் + இஞ்சி, உப்பு தண்ணீரில் கொப்பளியுங்கள், நிறைய ஓய்வு எடுங்கள். சாதாரண சளி 7-10 நாட்களில் சரியாகும். நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
        return "For cold and flu! 🤧 Medicines that help: Paracetamol — for fever and headache. Cetirizine or Loratadine — for runny nose and sneezing. Bromhexine — to loosen mucus and phlegm. Throat lozenges — for sore throat. Phenylephrine — for blocked nose (avoid if high BP). Home remedies: Steam inhalation, honey + ginger tea, gargle with warm salt water, rest and lots of fluids. Normal cold resolves in 7-10 days. Antibiotics do NOT work for cold/flu — it's viral! See doctor if fever is very high or symptoms worsen after 5 days. Remember: Always consult a real doctor before taking any medicine."


    # Check specific drug name
    for drug_name, info in DRUG_DATA.items():
        drug_lower = drug_name.lower().replace("(","").replace(")","")
        short_name = drug_lower.split()[0]
        if short_name in q or drug_lower in q:
            common = ", ".join(info["common"])
            serious = ", ".join(info["serious"])
            if lang=="தமிழ்":
                return f"**{drug_name}** ({info['class']}) பற்றி:\n\n📋 **பொதுவான பக்க விளைவுகள்:** {common}\n\n⚠️ **தீவிர பக்க விளைவுகள்:** {serious}\n\n💊 **வழக்கமான அளவு:** {info['usual_dose']}\n\nநினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
            return f"Here's what you need to know about **{drug_name}** ({info['class']}):\n\n📋 **Common side effects:** {common}\n\n⚠️ **Serious side effects to watch:** {serious}\n\n💊 **Usual dose:** {info['usual_dose']}\n\nRemember: Always consult a real doctor before taking any medicine."

    # General fallback
    if lang=="தமிழ்":
        return f"உங்கள் கேள்வி புரிந்தது! மருந்து பாதுகாப்பு பற்றி பொதுவான ஆலோசனை: எப்போதும் மருத்துவரின் பரிந்துரைப்படி மருந்து எடுக்கவும். சரியான அளவை கடைப்பிடிக்கவும். பக்க விளைவுகள் தோன்றினால் உடனே மருத்துவரை அணுகவும். மது மற்றும் புகையை தவிர்க்கவும். நீங்கள் கேட்டது பற்றி கொஞ்சம் விரிவாக கேளுங்கள் — நான் சரியாக பதில் சொல்கிறேன்! நினைவில் கொள்ளுங்கள்: எந்த மருந்தும் எடுக்கும் முன் மருத்துவரை அணுகவும்."
    return f"I understand you're asking about medicine safety! Here's general advice: Always follow your doctor's prescription exactly. Take the correct dose at the right time. Never share your medicines with others. Watch for unusual symptoms and report them to your doctor. Avoid alcohol and smoking while on medication. Could you be more specific about your question? For example — ask about a specific drug name, a condition, or a symptom — I'll give you a more detailed answer! Remember: Always consult a real doctor before taking any medicine."


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💊 Drug Safety AI")
    st.markdown("---")
    page=st.radio("📍 Navigate",["🏠 Home","🔬 Risk Predictor","🤖 AI Doctor"])
    st.markdown("---")
    st.markdown(f"**Accuracy:** {accuracy*100:.1f}%  \n**Drugs:** {len(DRUG_DATA)}  \n**Categories:** {len(CATEGORIES)}")
    st.markdown("---")
    st.caption("International Science Project 2025")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page=="🏠 Home":
    st.markdown("<div style='text-align:center;padding:2rem 0'><div style='font-size:4rem'>💊</div><h1 style='color:#00a884;font-size:2.5rem;font-weight:800'>AI-Powered Personalized<br>Drug Safety Assistant</h1><p style='font-size:1.1rem;color:#666;max-width:600px;margin:auto'>An intelligent system that predicts medicine side effect risks using Machine Learning — helping patients and doctors make safer decisions.</p></div>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ✨ What can this app do?")
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("<div class='feature-card'><div style='font-size:2.5rem'>🔬</div><h3>Risk Predictor</h3><p>Enter your drug and patient details — get instant Low, Medium or High risk prediction with AI confidence score</p></div>",unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='feature-card'><div style='font-size:2.5rem'>🤖</div><h3>AI Doctor</h3><p>Chat with our AI Doctor — ask anything about medicines, side effects, alcohol, smoking, stroke in English or Tamil</p></div>",unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='feature-card'><div style='font-size:2.5rem'>📊</div><h3>Smart Analysis</h3><p>Visual risk charts, dosage trend graphs, smart recommendations and known side effects all in one place</p></div>",unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"### 💊 {len(DRUG_DATA)} Drugs Covered Across {len(CATEGORIES)} Categories")
    cols=st.columns(4)
    for i,cat in enumerate(CATEGORIES.keys()):
        with cols[i%4]:
            st.markdown(f"<div style='padding:0.75rem;border-radius:8px;border:1px solid #ddd;margin:0.3rem 0;text-align:center'><b>{cat}</b><br><span style='color:#00a884;font-size:1.2rem'>{len(CATEGORIES[cat])} drugs</span></div>",unsafe_allow_html=True)
    st.markdown("---")
    s1,s2,s3,s4=st.columns(4)
    for col,val,label in [(s1,str(len(DRUG_DATA)),"Drugs in Database"),(s2,"12","Patient Parameters"),(s3,f"{accuracy*100:.0f}%","Model Accuracy"),(s4,"2","Languages Supported")]:
        with col:
            st.markdown(f"<div style='text-align:center;padding:1rem;border-radius:12px;border:2px solid #00a884'><div style='font-size:2rem;font-weight:800;color:#00a884'>{val}</div><div style='font-size:0.85rem;color:#666'>{label}</div></div>",unsafe_allow_html=True)
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate to **Risk Predictor** or **AI Doctor**")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — RISK PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🔬 Risk Predictor":
    st.markdown('<h2 style="color:#00a884">🔬 Medicine Side Effect Risk Predictor</h2>',unsafe_allow_html=True)
    st.caption("Fill in the details below — our AI predicts side effect risk instantly")
    st.markdown("---")
    col1,col2=st.columns(2)
    with col1:
        st.markdown("### 💊 Step 1 — Select Your Drug")
        category=st.selectbox("Drug Category",list(CATEGORIES.keys()))
        drug_name=st.selectbox("Drug Name",CATEGORIES[category])
        drug_info=DRUG_DATA[drug_name]
        st.success(f"**Usual Dose:** {drug_info['usual_dose']}  |  **Type:** {drug_info['class']}  |  **Base Risk:** {'⭐'*drug_info['risk_base']}")
        dose_mg=st.text_input("Patient's Actual Dose (e.g. 500mg or 1 tablet)",placeholder="e.g. 500mg")
        dosage_level=st.select_slider("Dose Strength Compared to Normal",[1,2,3],format_func=lambda x:{1:"🟢 Low — Less than normal",2:"🟡 Medium — As prescribed",3:"🔴 High — More than normal"}[x])
        st.caption("Low = less than prescribed | Medium = exactly as prescribed | High = more than normal (risky!)")
        num_ing=st.slider("Number of Active Ingredients",1,5,2)
        interactions=st.slider("Other Medicines Taken at the Same Time",0,5,1,help="0=only this drug | 1-2=a few others | 3-5=many medicines (more=higher risk)")
    with col2:
        st.markdown("### 👤 Step 2 — Patient Details")
        age_group=st.selectbox("Patient Age Group",[0,1,2],format_func=lambda x:{0:"👦 Child/Teen (below 18)",1:"🧑 Adult (18-60 years)",2:"👴 Elderly (above 60 years)"}[x])
        st.markdown("**Pre-existing Medical Conditions:**")
        st.caption("Tick all that apply — these affect how the body processes medicine")
        c1,c2=st.columns(2)
        with c1:
            kidney=st.checkbox("🫘 Kidney Disease")
            liver=st.checkbox("🫀 Liver Disease")
            heart=st.checkbox("❤️ Heart Disease")
        with c2:
            diabetes=st.checkbox("🩸 Diabetes")
            pregnant=st.checkbox("🤰 Pregnant")
            allergy=st.checkbox("⚠️ Past Drug Allergy")
        with st.expander(f"📋 Known Side Effects of {drug_name}"):
            c1,c2=st.columns(2)
            with c1:
                st.markdown("**Common (mild):**")
                for s in drug_info["common"]:
                    st.markdown(f"<div class='side-box'>🟡 {s}</div>",unsafe_allow_html=True)
            with c2:
                st.markdown("**Serious (watch out!):**")
                for s in drug_info["serious"]:
                    st.markdown(f"<div class='side-box'>🔴 {s}</div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    if st.button("🔮 Predict My Risk Now",use_container_width=True):
        row={"drug_class":DRUG_CLASS_MAP[drug_info["class"]],"drug_risk_base":drug_info["risk_base"],"num_ingredients":num_ing,"patient_age_group":age_group,"has_kidney_issue":int(kidney),"has_liver_issue":int(liver),"has_heart_issue":int(heart),"has_diabetes":int(diabetes),"is_pregnant":int(pregnant),"dosage_level":dosage_level,"drug_interactions":interactions,"allergy_history":int(allergy)}
        inp=pd.DataFrame([row])
        for c in feature_cols:
            if c not in inp.columns: inp[c]=0
        inp=inp[feature_cols]
        pred=model.predict(inp)[0]; proba=model.predict_proba(inp)[0]
        risk_pct=round(max(proba)*100,1)
        risk_names={0:"LOW",1:"MEDIUM",2:"HIGH"}
        risk_colors={0:"#00c853",1:"#ffa000",2:"#d32f2f"}
        bg_colors={0:"#e8f5e9",1:"#fff8e1",2:"#ffebee"}
        st.markdown("---")
        st.markdown("### 📋 Your Risk Result")
        m1,m2,m3=st.columns(3)
        for co,val,lbl,clr in [(m1,risk_names[pred],"Risk Level",risk_colors[pred]),(m2,f"{risk_pct}%","AI Confidence","#00a884"),(m3,f"{accuracy*100:.1f}%","Model Accuracy","#7c83fd")]:
            with co:
                st.markdown(f"<div class='metric-card'><div class='metric-value' style='color:{clr}'>{val}</div><div class='metric-label'>{lbl}</div></div>",unsafe_allow_html=True)
        emoji={0:"🟢",1:"🟡",2:"🔴"}[pred]
        msg={0:f"{drug_name} appears relatively safe for this patient.",1:f"Moderate risk for {drug_name}. Medical advice recommended.",2:f"HIGH risk with {drug_name}. See a doctor immediately!"}[pred]
        st.markdown(f"<div style='padding:1.5rem;border-radius:12px;text-align:center;border:2px solid {risk_colors[pred]};background:{bg_colors[pred]}'><h2 style='color:{risk_colors[pred]}'>{emoji} {risk_names[pred]} RISK</h2><p>{msg}</p>{f'<p><b>Dose entered:</b> {dose_mg}</p>' if dose_mg else ''}</div>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("### ⚠️ Side Effects to Watch")
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Common (mild, happens often):**")
            for s in drug_info["common"]:
                st.markdown(f"<div class='side-box'>🟡 {s}</div>",unsafe_allow_html=True)
        with c2:
            st.markdown("**Serious (watch out!):**")
            icon={0:"🔵",1:"⚠️",2:"🚨"}[pred]
            for s in drug_info["serious"]:
                st.markdown(f"<div class='side-box' style='border-left:3px solid {risk_colors[pred]}'>{icon} {s}</div>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("### 💡 What Should You Do?")
        recs={0:["✅ Continue medication exactly as your doctor prescribed.","✅ Take with food to avoid stomach upset.","✅ Routine checkup every 3 months is enough.","✅ Drink plenty of water daily."],1:["⚠️ Visit your doctor before continuing.","⚠️ Get kidney and liver tests every 4-6 weeks.","⚠️ Don't add new medicines without doctor approval.","⚠️ Report any unusual symptoms immediately."],2:["🚨 See a specialist doctor immediately.","🚨 Ask about switching to a safer alternative.","🚨 Do NOT take other medicines without supervision.","🚨 Get emergency blood tests done now.","🚨 Hospital admission may be needed."]}
        for r in recs[pred]:
            st.markdown(f"<div class='recommend-box'>{r}</div>",unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**📊 Risk Probability**")
            st.caption("How likely is each risk level for this patient?")
            fig,ax=plt.subplots(figsize=(5,3))
            bars=ax.barh(["🟢 Low","🟡 Medium","🔴 High"],proba,color=["#00c853","#ffa000","#d32f2f"],height=0.5)
            ax.set_xlim(0,1); ax.set_xlabel("← Less likely     More likely →",fontsize=9)
            ax.spines[['top','right','left']].set_visible(False)
            for bar,p in zip(bars,proba):
                ax.text(bar.get_width()+0.01,bar.get_y()+bar.get_height()/2,f"{p*100:.0f}%",va="center",fontsize=9)
            st.pyplot(fig)
        with c2:
            st.markdown("**🧠 What Influenced This Result?**")
            st.caption("Longer bar = bigger influence on prediction")
            feat_imp=pd.Series(model.feature_importances_,index=feature_cols).sort_values(ascending=False).head(5)
            readable={"drug_risk_base":"Drug's own risk","dosage_level":"Dosage strength","has_kidney_issue":"Kidney condition","has_liver_issue":"Liver condition","drug_interactions":"Other drugs taken","allergy_history":"Allergy history","patient_age_group":"Patient age","has_heart_issue":"Heart condition","has_diabetes":"Diabetes","is_pregnant":"Pregnancy","num_ingredients":"No. of ingredients","drug_class":"Drug type"}
            feat_imp.index=[readable.get(i,i) for i in feat_imp.index]
            fig2,ax2=plt.subplots(figsize=(5,3))
            feat_imp.plot(kind="barh",ax=ax2,color="#00a884")
            ax2.set_xlabel("Impact →",fontsize=9); ax2.invert_yaxis()
            ax2.spines[['top','right','bottom']].set_visible(False)
            st.pyplot(fig2)
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown("**📈 How Risk Changes With Dose**")
        st.caption("What happens if the dose increases beyond normal?")
        dosage_risks=[]
        for d in [1,2,3]:
            tr=row.copy(); tr["dosage_level"]=d
            ti=pd.DataFrame([tr])[feature_cols]
            dosage_risks.append(model.predict_proba(ti)[0][2]*100)
        fig3,ax3=plt.subplots(figsize=(8,3))
        xlabels=["🟢 Low Dose\n(Less than normal)","🟡 Medium Dose\n(As prescribed)","🔴 High Dose\n(More than normal)"]
        ax3.plot(xlabels,dosage_risks,color="#d32f2f",marker="o",linewidth=2.5,markersize=10)
        ax3.fill_between(xlabels,dosage_risks,alpha=0.1,color="#d32f2f")
        ax3.set_ylabel("High Risk %",fontsize=9)
        ax3.set_title(f"{drug_name} — Risk increases as dose goes higher",fontsize=10)
        ax3.spines[['top','right']].set_visible(False)
        for x,y in zip(xlabels,dosage_risks):
            ax3.annotate(f"{y:.0f}%",(x,y),textcoords="offset points",xytext=(0,12),ha="center",fontsize=9)
        st.pyplot(fig3)
        st.success("✅ For research and educational purposes only. Always consult a certified doctor.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — AI DOCTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page=="🤖 AI Doctor":
    st.markdown('<h2 style="color:#00a884">🤖 AI Doctor — Ask Me Anything!</h2>',unsafe_allow_html=True)
    st.caption("Chat with our AI Doctor about medicines, side effects, alcohol, smoking, stroke — in English or Tamil")
    st.markdown("<div style='background:#e8f5e9;padding:1rem;border-radius:12px;border-left:4px solid #00a884;margin-bottom:1rem'><b>👨‍⚕️ Hi! I'm your AI Doctor.</b> Ask me about any medicine, symptom, or health concern — I'll give you clear, simple advice!</div>",unsafe_allow_html=True)
    with col2:
        chat_lang="English"

    if "messages" not in st.session_state:
        st.session_state.messages=[]

    # Quick questions
    st.markdown("**⚡ Quick Questions — click any:**")
    q1,q2,q3,q4,q5,q6=st.columns(6)
    quick=""
    with q1:
        if st.button("🤒 Fever help"): quick="I have fever what medicine should I take"
    with q2:
        if st.button("🚬 Quit smoking"): quick="How to quit smoking what medicines help"
    with q3:
        if st.button("🍺 Alcohol + medicine"): quick="Is it safe to drink alcohol with medicine"
    with q4:
        if st.button("🧠 Stroke signs"): quick="What are stroke symptoms and medicines"
    with q5:
        if st.button("👶 Child fever"): quick="My child has fever what medicine is safe"
    with q6:
        if st.button("🚨 Emergency"): quick="What are emergency signs I need to go to hospital"

    # Chat display
    st.markdown("**💬 Conversation:**")
    if not st.session_state.messages:
        st.markdown("<div class='chat-ai'>👨‍⚕️ Hello! I'm your AI Doctor 😊 Ask me anything about:<br>💊 Medicine side effects &nbsp;|&nbsp; 🚬 Smoking &nbsp;|&nbsp; 🍺 Alcohol &nbsp;|&nbsp; 🧠 Stroke<br>👶 Children &nbsp;|&nbsp; 👴 Elderly &nbsp;|&nbsp; 🤰 Pregnancy &nbsp;|&nbsp; 🚨 Emergencies<br><br>I speak English and Tamil!</div>",unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if msg["role"]=="user":
            st.markdown(f"<div class='chat-user'>👤 {msg['content']}</div>",unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-ai'>👨‍⚕️ {msg['content']}</div>",unsafe_allow_html=True)

    c1,c2=st.columns([4,1])
    with c1:
        user_input=st.text_input("Type your question here...",placeholder="e.g. Can I take Paracetamol and Ibuprofen together?",label_visibility="collapsed")
    with c2:
        send=st.button("Send 📤",use_container_width=True)

    question=quick if quick else (user_input if send and user_input else "")
    if question:
        st.session_state.messages.append({"role":"user","content":question})
        with st.spinner("👨‍⚕️ Thinking..."):
            reply=ask_ai_doctor(question,chat_lang)
        st.session_state.messages.append({"role":"assistant","content":reply})
        st.rerun()

    if st.session_state.messages:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages=[]
            st.rerun()

    st.markdown("---")
    st.warning("⚠️ AI Doctor provides general information only. Always consult a real certified doctor for medical decisions.")
