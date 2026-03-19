import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle, warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Drug Safety Assistant", page_icon="💊", layout="wide")

# ── Language Toggle ─────────────────────────────────────────────────────────
LANG = st.sidebar.radio("🌐 Language / மொழி", ["English", "தமிழ்"])

T = {
    "English": {
        "title": "💊 AI-Powered Personalized Drug Safety Assistant",
        "subtitle": "Enter your drug and patient details below to instantly check side effect risk using Artificial Intelligence",
        "drug_section": "💊 Step 1 — Select Your Drug",
        "patient_section": "👤 Step 2 — Enter Patient Details",
        "category_label": "Drug Category (What type of medicine?)",
        "drug_label": "Drug Name (Select the exact medicine)",
        "dosage_label": "Dosage Level (How strong is the dose?)",
        "dosage_help": "Low = less than prescribed | Medium = as prescribed | High = more than prescribed",
        "ingredients_label": "Number of Active Ingredients (How many chemicals are in this drug?)",
        "interactions_label": "Other Drugs Taken Together (How many other medicines is the patient taking at the same time?)",
        "interactions_help": "0 = only this drug | 1-2 = few others | 3-5 = many other drugs (more = higher risk)",
        "age_label": "Patient Age Group",
        "conditions_label": "Pre-existing Medical Conditions (Does the patient already have these?)",
        "known_effects": "📋 View Known Side Effects of",
        "common_effects": "Common Side Effects (Mild, happens often)",
        "serious_effects": "Serious Side Effects (Rare but dangerous)",
        "predict_btn": "🔮 Check My Risk Now",
        "result_title": "📋 Your Risk Assessment Result",
        "risk_level": "Risk Level",
        "confidence": "AI Confidence",
        "accuracy": "Model Accuracy",
        "watch_title": "⚠️ Side Effects to Watch For",
        "recommend_title": "💡 What Should You Do?",
        "prob_title": "📊 Risk Probability Chart\n(How likely is each risk level?)",
        "factors_title": "🧠 What Caused This Risk?\n(Top factors the AI considered)",
        "trend_title": "📈 How Risk Changes With Dose\n(What happens if dose increases?)",
        "disclaimer": "✅ For research and educational purposes only. Always consult a certified doctor.",
        "drug_class_label": "Drug Type",
        "base_risk_label": "Base Risk Rating",
        "low": "LOW", "medium": "MEDIUM", "high": "HIGH",
        "dosage_options": {1: "🟢 Low — Below normal dose", 2: "🟡 Medium — Standard prescribed dose", 3: "🔴 High — Above normal dose"},
        "age_options": {0: "👦 Child / Teen (below 18)", 1: "🧑 Adult (18 to 60 years)", 2: "👴 Elderly (above 60 years)"},
        "conditions": ["🫘 Kidney Disease", "🫀 Liver Disease", "❤️ Heart Disease", "🩸 Diabetes", "🤰 Pregnant", "⚠️ Past Drug Allergy"],
        "low_msg": "appears relatively safe for this patient. Standard precautions apply.",
        "med_msg": "carries moderate risk for this patient. Medical advice recommended.",
        "high_msg": "carries HIGH risk for this patient. Immediate doctor consultation required!",
        "recs": {
            0: ["✅ Continue medication as prescribed by your doctor.",
                "✅ Take medicine with food to reduce stomach upset.",
                "✅ Routine checkup every 3 months is enough.",
                "✅ Drink plenty of water throughout the day."],
            1: ["⚠️ Visit your doctor before continuing this medicine.",
                "⚠️ Get kidney and liver tests every 4 to 6 weeks.",
                "⚠️ Do not add new medicines without doctor approval.",
                "⚠️ Report any new symptoms immediately to your doctor."],
            2: ["🚨 See a specialist doctor immediately — do not delay.",
                "🚨 Ask doctor about switching to a safer alternative medicine.",
                "🚨 Do NOT take any other medicines without supervision.",
                "🚨 Get emergency blood tests and organ function tests done.",
                "🚨 Hospital admission may be needed if symptoms appear."]
        },
        "model_info": "### 📊 Model Information",
        "algo": "Algorithm", "samples": "Training Samples",
        "features": "Features", "drugs": "Total Drugs",
        "env": "Environment", "deploy": "Deployment",
        "categories_heading": "### 📋 Drug Categories",
    },
    "தமிழ்": {
        "title": "💊 AI சக்தியுள்ள மருந்து பாதுகாப்பு உதவியாளர்",
        "subtitle": "உங்கள் மருந்து மற்றும் நோயாளி விவரங்களை உள்ளிட்டு பக்க விளைவு அபாயத்தை உடனே சரிபாருங்கள்",
        "drug_section": "💊 படி 1 — உங்கள் மருந்தை தேர்ந்தெடுக்கவும்",
        "patient_section": "👤 படி 2 — நோயாளி விவரங்களை உள்ளிடவும்",
        "category_label": "மருந்து வகை (எந்த வகை மருந்து?)",
        "drug_label": "மருந்தின் பெயர் (சரியான மருந்தை தேர்ந்தெடுக்கவும்)",
        "dosage_label": "மோதிரை அளவு (டோஸ் எவ்வளவு வலிமையானது?)",
        "dosage_help": "குறைவு = பரிந்துரைக்கப்பட்டதை விட குறைவு | நடுத்தரம் = பரிந்துரைக்கப்பட்டபடி | அதிகம் = பரிந்துரைக்கப்பட்டதை விட அதிகம்",
        "ingredients_label": "செயலில் உள்ள பொருட்களின் எண்ணிக்கை",
        "interactions_label": "ஒரே நேரத்தில் எடுக்கப்படும் மற்ற மருந்துகள்",
        "interactions_help": "0 = இந்த மருந்து மட்டும் | 1-2 = சில மருந்துகள் | 3-5 = பல மருந்துகள் (அதிகம் = அதிக அபாயம்)",
        "age_label": "நோயாளியின் வயது பிரிவு",
        "conditions_label": "ஏற்கனவே உள்ள மருத்துவ நிலைமைகள்",
        "known_effects": "📋 இந்த மருந்தின் அறியப்பட்ட பக்க விளைவுகள்",
        "common_effects": "பொதுவான பக்க விளைவுகள் (லேசானவை, அடிக்கடி நடக்கும்)",
        "serious_effects": "தீவிர பக்க விளைவுகள் (அரிதானவை ஆனால் ஆபத்தானவை)",
        "predict_btn": "🔮 என் அபாயத்தை இப்போது சரிபாருங்கள்",
        "result_title": "📋 உங்கள் அபாய மதிப்பீட்டு முடிவு",
        "risk_level": "அபாய நிலை",
        "confidence": "AI நம்பகத்தன்மை",
        "accuracy": "மாதிரி துல்லியம்",
        "watch_title": "⚠️ கவனிக்க வேண்டிய பக்க விளைவுகள்",
        "recommend_title": "💡 நீங்கள் என்ன செய்ய வேண்டும்?",
        "prob_title": "📊 அபாய நிகழ்தகவு விளக்கப்படம்\n(ஒவ்வொரு அபாய நிலையும் எவ்வளவு சாத்தியம்?)",
        "factors_title": "🧠 இந்த அபாயத்திற்கு என்ன காரணம்?\n(AI கருதிய முக்கிய காரணிகள்)",
        "trend_title": "📈 டோஸ் அதிகரிக்கும்போது அபாயம் எப்படி மாறுகிறது?",
        "disclaimer": "✅ ஆராய்ச்சி மற்றும் கல்வி நோக்கங்களுக்காக மட்டுமே. எப்போதும் சான்றளிக்கப்பட்ட மருத்துவரை அணுகவும்.",
        "drug_class_label": "மருந்து வகை",
        "base_risk_label": "அடிப்படை அபாய மதிப்பீடு",
        "low": "குறைவு", "medium": "நடுத்தரம்", "high": "அதிகம்",
        "dosage_options": {1: "🟢 குறைவு — சாதாரண அளவுக்கு கீழே", 2: "🟡 நடுத்தரம் — பரிந்துரைக்கப்பட்ட அளவு", 3: "🔴 அதிகம் — சாதாரண அளவுக்கு மேலே"},
        "age_options": {0: "👦 குழந்தை / இளைஞர் (18 வயதுக்கு கீழ்)", 1: "🧑 பெரியவர் (18 முதல் 60 வயது)", 2: "👴 முதியவர் (60 வயதுக்கு மேல்)"},
        "conditions": ["🫘 சிறுநீரக நோய்", "🫀 கல்லீரல் நோய்", "❤️ இதய நோய்", "🩸 நீரிழிவு", "🤰 கர்ப்பிணி", "⚠️ மருந்து ஒவ்வாமை வரலாறு"],
        "low_msg": "இந்த நோயாளிக்கு ஒப்பீட்டளவில் பாதுகாப்பானது.",
        "med_msg": "இந்த நோயாளிக்கு மிதமான அபாயம் உள்ளது. மருத்துவ ஆலோசனை பரிந்துரைக்கப்படுகிறது.",
        "high_msg": "இந்த நோயாளிக்கு அதிக அபாயம் உள்ளது! உடனடியாக மருத்துவரை அணுகவும்!",
        "recs": {
            0: ["✅ மருத்துவர் பரிந்துரைத்தபடி மருந்தை தொடரவும்.",
                "✅ வயிறு வலியை குறைக்க உணவுடன் மருந்து எடுக்கவும்.",
                "✅ 3 மாதத்திற்கு ஒரு முறை வழக்கமான பரிசோதனை போதுமானது.",
                "✅ நாள் முழுவதும் நிறைய தண்ணீர் குடிக்கவும்."],
            1: ["⚠️ இந்த மருந்தை தொடர்வதற்கு முன் மருத்துவரை அணுகவும்.",
                "⚠️ 4 முதல் 6 வாரங்களுக்கு ஒரு முறை சிறுநீரகம் மற்றும் கல்லீரல் பரிசோதனை செய்யவும்.",
                "⚠️ மருத்துவர் ஒப்புதல் இல்லாமல் புதிய மருந்துகளை சேர்க்காதீர்கள்.",
                "⚠️ புதிய அறிகுறிகள் தோன்றினால் உடனே மருத்துவரிடம் தெரிவிக்கவும்."],
            2: ["🚨 உடனடியாக நிபுணர் மருத்துவரை சந்தியுங்கள் — தாமதிக்காதீர்கள்.",
                "🚨 பாதுகாப்பான மாற்று மருந்துகளைப் பற்றி மருத்துவரிடம் கேளுங்கள்.",
                "🚨 மேற்பார்வை இல்லாமல் வேறு மருந்துகள் எடுக்காதீர்கள்.",
                "🚨 அவசர இரத்த பரிசோதனை மற்றும் உறுப்பு செயல்பாடு பரிசோதனை செய்யவும்.",
                "🚨 அறிகுறிகள் தோன்றினால் மருத்துவமனையில் அனுமதி தேவைப்படலாம்."]
        },
        "model_info": "### 📊 மாதிரி தகவல்",
        "algo": "வழிமுறை", "samples": "பயிற்சி மாதிரிகள்",
        "features": "அம்சங்கள்", "drugs": "மொத்த மருந்துகள்",
        "env": "சூழல்", "deploy": "வரிசைப்படுத்தல்",
        "categories_heading": "### 📋 மருந்து வகைகள்",
    }
}

tx = T[LANG]

DRUG_DATA = {
    "Paracetamol (Acetaminophen)": {"class":"Painkiller","common_side_effects":["Nausea / குமட்டல்","Stomach pain / வயிற்று வலி","Loss of appetite / பசியின்மை"],"serious_side_effects":["Liver damage / கல்லீரல் சேதம்","Kidney damage / சிறுநீரக சேதம்","Severe allergic reaction / கடுமையான ஒவ்வாமை"],"risk_base":1},
    "Ibuprofen": {"class":"Painkiller","common_side_effects":["Stomach upset / வயிற்று அசெளகரியம்","Heartburn / நெஞ்செரிச்சல்","Dizziness / தலைச்சுற்றல்","Headache / தலைவலி"],"serious_side_effects":["Stomach bleeding / வயிற்று இரத்தப்போக்கு","Kidney problems / சிறுநீரக பிரச்சனை","Heart attack risk / மாரடைப்பு அபாயம்"],"risk_base":2},
    "Aspirin": {"class":"Painkiller","common_side_effects":["Stomach irritation / வயிற்று எரிச்சல்","Nausea / குமட்டல்","Heartburn / நெஞ்செரிச்சல்"],"serious_side_effects":["Internal bleeding / உள் இரத்தப்போக்கு","Reye's syndrome in children / குழந்தைகளில் ரேய்ஸ் நோய்க்குறி","Stroke risk / பக்கவாதம் அபாயம்"],"risk_base":2},
    "Diclofenac": {"class":"Painkiller","common_side_effects":["Stomach pain / வயிற்று வலி","Nausea / குமட்டல்","Headache / தலைவலி"],"serious_side_effects":["Stomach ulcers / வயிற்று புண்கள்","Heart attack / மாரடைப்பு","Kidney failure / சிறுநீரக செயலிழப்பு"],"risk_base":2},
    "Tramadol": {"class":"Painkiller","common_side_effects":["Nausea / குமட்டல்","Dizziness / தலைச்சுற்றல்","Constipation / மலச்சிக்கல்"],"serious_side_effects":["Seizures / வலிப்பு","Addiction risk / அடிமையாதல் அபாயம்","Breathing problems / சுவாச பிரச்சனை"],"risk_base":3},
    "Naproxen": {"class":"Painkiller","common_side_effects":["Stomach upset / வயிறு அசெளகரியம்","Heartburn / நெஞ்செரிச்சல்","Drowsiness / தூக்கம்"],"serious_side_effects":["GI bleeding / இரைப்பை இரத்தப்போக்கு","Kidney damage / சிறுநீரக சேதம்","Heart problems / இதய பிரச்சனை"],"risk_base":2},
    "Codeine": {"class":"Painkiller","common_side_effects":["Constipation / மலச்சிக்கல்","Drowsiness / தூக்கம்","Nausea / குமட்டல்"],"serious_side_effects":["Addiction / அடிமையாதல்","Breathing problems / சுவாச பிரச்சனை","Liver damage / கல்லீரல் சேதம்"],"risk_base":3},
    "Mefenamic Acid": {"class":"Painkiller","common_side_effects":["Stomach upset / வயிற்று அசெளகரியம்","Diarrhea / வயிற்றுப்போக்கு","Dizziness / தலைச்சுற்றல்"],"serious_side_effects":["Kidney failure / சிறுநீரக செயலிழப்பு","Stomach bleeding / வயிற்று இரத்தப்போக்கு","Seizures / வலிப்பு"],"risk_base":2},
    "Amoxicillin": {"class":"Antibiotic","common_side_effects":["Diarrhea / வயிற்றுப்போக்கு","Stomach upset / வயிறு அசெளகரியம்","Skin rash / தோல் தடிப்பு"],"serious_side_effects":["Severe allergic reaction / கடுமையான ஒவ்வாமை","Liver problems / கல்லீரல் பிரச்சனை","Colitis / பெருங்குடல் அழற்சி"],"risk_base":1},
    "Ciprofloxacin": {"class":"Antibiotic","common_side_effects":["Nausea / குமட்டல்","Diarrhea / வயிற்றுப்போக்கு","Headache / தலைவலி"],"serious_side_effects":["Tendon rupture / தசைநார் கிழிதல்","Nerve damage / நரம்பு சேதம்","Heart rhythm problems / இதய தாள பிரச்சனை"],"risk_base":3},
    "Azithromycin": {"class":"Antibiotic","common_side_effects":["Nausea / குமட்டல்","Diarrhea / வயிற்றுப்போக்கு","Stomach pain / வயிற்று வலி"],"serious_side_effects":["Heart rhythm problems / இதய தாள பிரச்சனை","Liver damage / கல்லீரல் சேதம்","Severe allergic reaction / கடுமையான ஒவ்வாமை"],"risk_base":2},
    "Doxycycline": {"class":"Antibiotic","common_side_effects":["Nausea / குமட்டல்","Sun sensitivity / சூரிய ஒளி உணர்திறன்","Stomach upset / வயிறு அசெளகரியம்"],"serious_side_effects":["Esophageal damage / உணவுக்குழாய் சேதம்","Liver toxicity / கல்லீரல் நச்சுத்தன்மை","Intracranial pressure / மண்டை ஓட்டு அழுத்தம்"],"risk_base":2},
    "Metronidazole": {"class":"Antibiotic","common_side_effects":["Nausea / குமட்டல்","Metallic taste / உலோக சுவை","Headache / தலைவலி"],"serious_side_effects":["Nerve damage / நரம்பு சேதம்","Seizures / வலிப்பு","Severe skin reactions / தீவிர தோல் வினைகள்"],"risk_base":2},
    "Clindamycin": {"class":"Antibiotic","common_side_effects":["Diarrhea / வயிற்றுப்போக்கு","Nausea / குமட்டல்","Stomach pain / வயிற்று வலி"],"serious_side_effects":["Severe colitis / தீவிர பெருங்குடல் அழற்சி","Allergic reactions / ஒவ்வாமை வினைகள்","Liver problems / கல்லீரல் பிரச்சனை"],"risk_base":2},
    "Cephalexin": {"class":"Antibiotic","common_side_effects":["Diarrhea / வயிற்றுப்போக்கு","Nausea / குமட்டல்","Stomach upset / வயிறு அசெளகரியம்"],"serious_side_effects":["Severe allergic reaction / கடுமையான ஒவ்வாமை","Kidney problems / சிறுநீரக பிரச்சனை","Colitis / பெருங்குடல் அழற்சி"],"risk_base":1},
    "Erythromycin": {"class":"Antibiotic","common_side_effects":["Nausea / குமட்டல்","Stomach cramps / வயிற்று பிடிப்பு","Diarrhea / வயிற்றுப்போக்கு"],"serious_side_effects":["Heart rhythm problems / இதய தாள பிரச்சனை","Liver damage / கல்லீரல் சேதம்","Hearing loss / செவிட்டுத்தன்மை"],"risk_base":2},
    "Trimethoprim": {"class":"Antibiotic","common_side_effects":["Nausea / குமட்டல்","Rash / தடிப்பு","Headache / தலைவலி"],"serious_side_effects":["Kidney problems / சிறுநீரக பிரச்சனை","Blood disorders / இரத்த கோளாறுகள்","Severe skin reactions / தீவிர தோல் வினைகள்"],"risk_base":2},
    "Nitrofurantoin": {"class":"Antibiotic","common_side_effects":["Nausea / குமட்டல்","Headache / தலைவலி","Urine discoloration / சிறுநீர் நிற மாற்றம்"],"serious_side_effects":["Lung toxicity / நுரையீரல் நச்சுத்தன்மை","Liver damage / கல்லீரல் சேதம்","Nerve damage / நரம்பு சேதம்"],"risk_base":2},
    "Ampicillin": {"class":"Antibiotic","common_side_effects":["Diarrhea / வயிற்றுப்போக்கு","Rash / தடிப்பு","Nausea / குமட்டல்"],"serious_side_effects":["Severe allergic reaction / கடுமையான ஒவ்வாமை","Colitis / பெருங்குடல் அழற்சி","Seizures / வலிப்பு"],"risk_base":1},
    "Levofloxacin": {"class":"Antibiotic","common_side_effects":["Nausea / குமட்டல்","Diarrhea / வயிற்றுப்போக்கு","Headache / தலைவலி"],"serious_side_effects":["Tendon rupture / தசைநார் கிழிதல்","Heart problems / இதய பிரச்சனை","Mental health effects / மன நல விளைவுகள்"],"risk_base":3},
    "Cetirizine": {"class":"Antiallergy","common_side_effects":["Drowsiness / தூக்கம்","Dry mouth / வாய் வறட்சி","Headache / தலைவலி"],"serious_side_effects":["Severe allergic reaction / கடுமையான ஒவ்வாமை","Fast heartbeat / வேகமான இதயத்துடிப்பு","Tremors / நடுக்கம்"],"risk_base":1},
    "Loratadine": {"class":"Antiallergy","common_side_effects":["Headache / தலைவலி","Dry mouth / வாய் வறட்சி","Fatigue / சோர்வு"],"serious_side_effects":["Fast heartbeat / வேகமான இதயத்துடிப்பு","Liver problems / கல்லீரல் பிரச்சனை","Severe allergic reaction / கடுமையான ஒவ்வாமை"],"risk_base":1},
    "Chlorpheniramine": {"class":"Antiallergy","common_side_effects":["Drowsiness / தூக்கம்","Dry mouth / வாய் வறட்சி","Dizziness / தலைச்சுற்றல்"],"serious_side_effects":["Urinary retention / சிறுநீர் தேக்கம்","Confusion in elderly / முதியோரில் குழப்பம்","Vision problems / பார்வை பிரச்சனை"],"risk_base":1},
    "Phenylephrine": {"class":"Antiallergy","common_side_effects":["Headache / தலைவலி","Nausea / குமட்டல்","Increased BP / உயர் இரத்த அழுத்தம்"],"serious_side_effects":["Severe hypertension / கடுமையான உயர் இரத்த அழுத்தம்","Heart attack / மாரடைப்பு","Stroke / பக்கவாதம்"],"risk_base":2},
    "Dextromethorphan": {"class":"Antiallergy","common_side_effects":["Drowsiness / தூக்கம்","Dizziness / தலைச்சுற்றல்","Nausea / குமட்டல்"],"serious_side_effects":["Serotonin syndrome / செரோடோனின் நோய்க்குறி","Hallucinations / மாயத்தோற்றம்","Dependency / சார்பு"],"risk_base":2},
    "Bromhexine": {"class":"Antiallergy","common_side_effects":["Nausea / குமட்டல்","Diarrhea / வயிற்றுப்போக்கு","Dizziness / தலைச்சுற்றல்"],"serious_side_effects":["Severe skin reactions / தீவிர தோல் வினைகள்","Liver problems / கல்லீரல் பிரச்சனை"],"risk_base":1},
    "Salbutamol (Albuterol)": {"class":"Antiallergy","common_side_effects":["Tremors / நடுக்கம்","Headache / தலைவலி","Fast heartbeat / வேகமான இதயத்துடிப்பு"],"serious_side_effects":["Severe chest pain / தீவிர மார்பு வலி","Irregular heartbeat / ஒழுங்கற்ற இதயத்துடிப்பு","Low potassium / குறைந்த பொட்டாசியம்"],"risk_base":2},
    "Omeprazole": {"class":"Gastrointestinal","common_side_effects":["Headache / தலைவலி","Nausea / குமட்டல்","Diarrhea / வயிற்றுப்போக்கு"],"serious_side_effects":["Kidney disease / சிறுநீரக நோய்","Low magnesium / குறைந்த மெக்னீசியம்","Bone fractures / எலும்பு முறிவுகள்"],"risk_base":1},
    "Pantoprazole": {"class":"Gastrointestinal","common_side_effects":["Headache / தலைவலி","Diarrhea / வயிற்றுப்போக்கு","Nausea / குமட்டல்"],"serious_side_effects":["Kidney inflammation / சிறுநீரக அழற்சி","Low magnesium / குறைந்த மெக்னீசியம்","C. diff infection / சி. டிஃப் தொற்று"],"risk_base":1},
    "Ondansetron": {"class":"Gastrointestinal","common_side_effects":["Headache / தலைவலி","Constipation / மலச்சிக்கல்","Fatigue / சோர்வு"],"serious_side_effects":["Heart rhythm problems / இதய தாள பிரச்சனை","Serotonin syndrome / செரோடோனின் நோய்க்குறி","Severe allergic reaction / கடுமையான ஒவ்வாமை"],"risk_base":2},
    "Domperidone": {"class":"Gastrointestinal","common_side_effects":["Dry mouth / வாய் வறட்சி","Headache / தலைவலி","Diarrhea / வயிற்றுப்போக்கு"],"serious_side_effects":["Heart rhythm problems / இதய தாள பிரச்சனை","Sudden cardiac death / திடீர் இதய மரணம்","Hormonal effects / ஹார்மோன் விளைவுகள்"],"risk_base":2},
    "Ranitidine": {"class":"Gastrointestinal","common_side_effects":["Headache / தலைவலி","Diarrhea / வயிற்றுப்போக்கு","Nausea / குமட்டல்"],"serious_side_effects":["Liver problems / கல்லீரல் பிரச்சனை","Blood disorders / இரத்த கோளாறுகள்","Kidney problems / சிறுநீரக பிரச்சனை"],"risk_base":1},
    "Loperamide": {"class":"Gastrointestinal","common_side_effects":["Constipation / மலச்சிக்கல்","Dizziness / தலைச்சுற்றல்","Nausea / குமட்டல்"],"serious_side_effects":["Heart rhythm problems / இதய தாள பிரச்சனை","Toxic megacolon / நச்சு மெகாகோலன்","Ileus / குடல் அடைப்பு"],"risk_base":2},
    "Lactulose": {"class":"Gastrointestinal","common_side_effects":["Bloating / வீக்கம்","Diarrhea / வயிற்றுப்போக்கு","Stomach cramps / வயிற்று பிடிப்பு"],"serious_side_effects":["Severe electrolyte imbalance / கடுமையான எலக்ட்ரோலைட் சமநிலையின்மை","Dehydration / நீர்ச்சத்து குறைபாடு"],"risk_base":1},
    "Amlodipine": {"class":"Antihypertensive","common_side_effects":["Swollen ankles / கணுக்கால் வீக்கம்","Flushing / சிவத்தல்","Headache / தலைவலி"],"serious_side_effects":["Severe low BP / கடுமையான குறைந்த இரத்த அழுத்தம்","Chest pain / மார்பு வலி","Heart failure / இதய செயலிழப்பு"],"risk_base":2},
    "Atorvastatin": {"class":"Antihypertensive","common_side_effects":["Muscle pain / தசை வலி","Joint pain / மூட்டு வலி","Diarrhea / வயிற்றுப்போக்கு"],"serious_side_effects":["Severe muscle breakdown / தீவிர தசை சிதைவு","Liver damage / கல்லீரல் சேதம்","Memory problems / நினைவாற்றல் பிரச்சனை"],"risk_base":2},
    "Lisinopril": {"class":"Antihypertensive","common_side_effects":["Dry cough / வறட்டு இருமல்","Dizziness / தலைச்சுற்றல்","Headache / தலைவலி"],"serious_side_effects":["Angioedema / ஆஞ்சியோஈடிமா","Kidney failure / சிறுநீரக செயலிழப்பு","High potassium / உயர் பொட்டாசியம்"],"risk_base":2},
    "Ramipril": {"class":"Antihypertensive","common_side_effects":["Cough / இருமல்","Dizziness / தலைச்சுற்றல்","Fatigue / சோர்வு"],"serious_side_effects":["Angioedema / ஆஞ்சியோஈடிமா","Kidney problems / சிறுநீரக பிரச்சனை","Low BP / குறைந்த இரத்த அழுத்தம்"],"risk_base":2},
    "Metoprolol": {"class":"Antihypertensive","common_side_effects":["Fatigue / சோர்வு","Dizziness / தலைச்சுற்றல்","Cold hands / குளிர்ந்த கைகள்"],"serious_side_effects":["Severe low heart rate / கடுமையான குறைந்த இதய துடிப்பு","Heart failure / இதய செயலிழப்பு","Depression / மனச்சோர்வு"],"risk_base":2},
    "Warfarin": {"class":"Antihypertensive","common_side_effects":["Easy bruising / எளிதில் நீலமடித்தல்","Bleeding gums / ஈறுகளில் இரத்தப்போக்கு","Fatigue / சோர்வு"],"serious_side_effects":["Severe internal bleeding / கடுமையான உள் இரத்தப்போக்கு","Brain hemorrhage / மூளை இரத்தப்போக்கு","Stroke / பக்கவாதம்"],"risk_base":3},
    "Metformin": {"class":"Antidiabetic","common_side_effects":["Nausea / குமட்டல்","Diarrhea / வயிற்றுப்போக்கு","Stomach pain / வயிற்று வலி"],"serious_side_effects":["Lactic acidosis / லாக்டிக் அமிலத்தன்மை","Vitamin B12 deficiency / B12 குறைபாடு","Kidney stress / சிறுநீரக அழுத்தம்"],"risk_base":2},
    "Glibenclamide": {"class":"Antidiabetic","common_side_effects":["Low blood sugar / குறைந்த இரத்த சர்க்கரை","Nausea / குமட்டல்","Weight gain / எடை அதிகரிப்பு"],"serious_side_effects":["Severe hypoglycemia / கடுமையான ஹைபோகிளைசீமியா","Liver damage / கல்லீரல் சேதம்","Blood disorders / இரத்த கோளாறுகள்"],"risk_base":3},
    "Insulin (Regular)": {"class":"Antidiabetic","common_side_effects":["Low blood sugar / குறைந்த இரத்த சர்க்கரை","Injection site pain / ஊசி இடம் வலி","Weight gain / எடை அதிகரிப்பு"],"serious_side_effects":["Severe hypoglycemia / கடுமையான ஹைபோகிளைசீமியா","Hypokalemia / குறைந்த பொட்டாசியம்","Lipodystrophy / லிபோடிஸ்ட்ரோபி"],"risk_base":3},
    "Sitagliptin": {"class":"Antidiabetic","common_side_effects":["Runny nose / மூக்கு ஒழுகுதல்","Headache / தலைவலி","Stomach pain / வயிற்று வலி"],"serious_side_effects":["Pancreatitis / கணைய அழற்சி","Kidney problems / சிறுநீரக பிரச்சனை","Severe joint pain / தீவிர மூட்டு வலி"],"risk_base":2},
    "Sertraline": {"class":"Antidepressant","common_side_effects":["Nausea / குமட்டல்","Insomnia / தூக்கமின்மை","Dizziness / தலைச்சுற்றல்"],"serious_side_effects":["Suicidal thoughts in youth / இளைஞர்களில் தற்கொலை எண்ணங்கள்","Serotonin syndrome / செரோடோனின் நோய்க்குறி","Bleeding risk / இரத்தப்போக்கு அபாயம்"],"risk_base":2},
    "Diazepam": {"class":"Antidepressant","common_side_effects":["Drowsiness / தூக்கம்","Dizziness / தலைச்சுற்றல்","Fatigue / சோர்வு"],"serious_side_effects":["Addiction / அடிமையாதல்","Respiratory depression / சுவாச மந்தம்","Memory impairment / நினைவாற்றல் குறைபாடு"],"risk_base":3},
    "Alprazolam": {"class":"Antidepressant","common_side_effects":["Drowsiness / தூக்கம்","Dizziness / தலைச்சுற்றல்","Memory issues / நினைவாற்றல் பிரச்சனை"],"serious_side_effects":["Severe addiction / கடுமையான அடிமையாதல்","Withdrawal seizures / திரும்பப்பெறும் வலிப்பு","Respiratory depression / சுவாச மந்தம்"],"risk_base":3},
    "Melatonin": {"class":"Antidepressant","common_side_effects":["Drowsiness / தூக்கம்","Headache / தலைவலி","Dizziness / தலைச்சுற்றல்"],"serious_side_effects":["Hormonal effects / ஹார்மோன் விளைவுகள்","Depression worsening / மனச்சோர்வு மோசமாதல்","Vivid dreams / தெளிவான கனவுகள்"],"risk_base":1},
    "Fluoxetine": {"class":"Antidepressant","common_side_effects":["Nausea / குமட்டல்","Headache / தலைவலி","Insomnia / தூக்கமின்மை"],"serious_side_effects":["Serotonin syndrome / செரோடோனின் நோய்க்குறி","Suicidal thoughts / தற்கொலை எண்ணங்கள்","Bleeding risk / இரத்தப்போக்கு அபாயம்"],"risk_base":2},
    "Vitamin C (Ascorbic Acid)": {"class":"Supplement","common_side_effects":["Stomach upset / வயிறு அசெளகரியம்","Diarrhea / வயிற்றுப்போக்கு","Nausea / குமட்டல்"],"serious_side_effects":["Kidney stones at high dose / அதிக அளவில் சிறுநீரக கற்கள்","Iron overload / இரும்பு அதிகப்படிதல்","Digestive problems / செரிமான பிரச்சனை"],"risk_base":1},
    "Vitamin D3": {"class":"Supplement","common_side_effects":["Nausea / குமட்டல்","Constipation / மலச்சிக்கல்","Fatigue / சோர்வு"],"serious_side_effects":["Calcium toxicity / கால்சியம் நச்சுத்தன்மை","Kidney damage at high dose / அதிக அளவில் சிறுநீரக சேதம்","Heart rhythm problems / இதய தாள பிரச்சனை"],"risk_base":1},
    "Iron Supplement": {"class":"Supplement","common_side_effects":["Constipation / மலச்சிக்கல்","Stomach pain / வயிற்று வலி","Dark stools / கருமையான மலம்"],"serious_side_effects":["Iron toxicity / இரும்பு நச்சுத்தன்மை","Liver damage / கல்லீரல் சேதம்","Stomach bleeding / வயிற்று இரத்தப்போக்கு"],"risk_base":1},
}

DRUG_CLASS_MAP = {"Painkiller":0,"Antibiotic":1,"Antiallergy":2,"Gastrointestinal":3,"Antihypertensive":4,"Antidiabetic":5,"Antidepressant":6,"Supplement":7}

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
    data = {"drug_class":np.random.choice(list(range(8)),n),"drug_risk_base":np.random.choice([1,2,3],n),"num_ingredients":np.random.randint(1,6,n),"patient_age_group":np.random.choice([0,1,2],n),"has_kidney_issue":np.random.choice([0,1],n,p=[0.8,0.2]),"has_liver_issue":np.random.choice([0,1],n,p=[0.85,0.15]),"has_heart_issue":np.random.choice([0,1],n,p=[0.85,0.15]),"has_diabetes":np.random.choice([0,1],n,p=[0.8,0.2]),"is_pregnant":np.random.choice([0,1],n,p=[0.9,0.1]),"dosage_level":np.random.choice([1,2,3],n),"drug_interactions":np.random.randint(0,6,n),"allergy_history":np.random.choice([0,1],n,p=[0.85,0.15])}
    df = pd.DataFrame(data)
    risk = df["drug_risk_base"]+(df["dosage_level"]==3).astype(int)*2+df["has_kidney_issue"]*2+df["has_liver_issue"]*2+df["has_heart_issue"]+df["has_diabetes"]+df["is_pregnant"]*2+(df["drug_interactions"]>3).astype(int)*2+df["allergy_history"]*2+(df["patient_age_group"]==2).astype(int)
    df["risk_label"] = pd.cut(risk,bins=[-1,4,8,20],labels=[0,1,2]).astype(int)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop("risk_label",axis=1); y = df["risk_label"]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    model = RandomForestClassifier(n_estimators=200,random_state=42)
    model.fit(X_train,y_train)
    acc = accuracy_score(y_test,model.predict(X_test))
    pickle.dump(model,open("model.pkl","wb"))
    return model,X.columns.tolist(),acc

df = load_data()
model,feature_cols,accuracy = train_model(df)

with st.sidebar:
    st.markdown(tx["model_info"])
    st.markdown(f"""<div style='background:#1e2130;padding:1rem;border-radius:10px;text-align:center;margin-bottom:0.5rem'>
    <div style='font-size:2rem;font-weight:700;color:#00d4aa'>{accuracy*100:.1f}%</div>
    <div style='font-size:0.85rem;color:#aaa'>{tx['accuracy']}</div></div>""",unsafe_allow_html=True)
    st.markdown(f"""
**{tx['algo']}:** Random Forest  
**{tx['samples']}:** 640  
**{tx['features']}:** 12  
**{tx['drugs']}:** 50  
**{tx['env']}:** Cloud (Google Colab)  
**{tx['deploy']}:** Streamlit  
    """)
    st.markdown("---")
    st.markdown(tx["categories_heading"])
    for cat in CATEGORIES:
        st.markdown(f"{cat} — {len(CATEGORIES[cat])}")
    st.markdown("---")
    st.caption("AI Drug Safety Assistant | International Science Project 2025")

st.markdown(f'<p style="font-size:2rem;font-weight:700;color:#00d4aa;text-align:center">{tx["title"]}</p>',unsafe_allow_html=True)
st.markdown(f'<p style="font-size:1rem;color:#aaa;text-align:center;margin-bottom:1rem">{tx["subtitle"]}</p>',unsafe_allow_html=True)
st.markdown("---")

col1,col2 = st.columns(2)

with col1:
    st.markdown(f"### {tx['drug_section']}")
    category  = st.selectbox(tx["category_label"], list(CATEGORIES.keys()))
    drug_name = st.selectbox(tx["drug_label"], CATEGORIES[category])
    drug_info = DRUG_DATA[drug_name]
    dosage_level = st.selectbox(tx["dosage_label"],[1,2,3],format_func=lambda x:tx["dosage_options"][x],help=tx["dosage_help"])
    num_ing = st.slider(tx["ingredients_label"],1,5,2)
    interactions = st.slider(tx["interactions_label"],0,5,1,help=tx["interactions_help"])
    st.info(f"**{tx['drug_class_label']}:** {drug_info['class']}  |  **{tx['base_risk_label']}:** {'⭐'*drug_info['risk_base']}")

with col2:
    st.markdown(f"### {tx['patient_section']}")
    age_group = st.selectbox(tx["age_label"],[0,1,2],format_func=lambda x:tx["age_options"][x])
    st.markdown(f"**{tx['conditions_label']}:**")
    conds = tx["conditions"]
    c1,c2 = st.columns(2)
    with c1:
        kidney = st.checkbox(conds[0])
        liver  = st.checkbox(conds[1])
        heart  = st.checkbox(conds[2])
    with c2:
        diabetes = st.checkbox(conds[3])
        pregnant = st.checkbox(conds[4])
        allergy  = st.checkbox(conds[5])

with st.expander(f"{tx['known_effects']} {drug_name}"):
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"**{tx['common_effects']}:**")
        for s in drug_info["common_side_effects"]:
            st.markdown(f"<div style='background:#1e2130;border:1px solid #333;padding:0.5rem 1rem;border-radius:8px;margin:0.3rem 0'>🟡 {s}</div>",unsafe_allow_html=True)
    with c2:
        st.markdown(f"**{tx['serious_effects']}:**")
        for s in drug_info["serious_side_effects"]:
            st.markdown(f"<div style='background:#1e2130;border:1px solid #333;padding:0.5rem 1rem;border-radius:8px;margin:0.3rem 0'>🔴 {s}</div>",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)

if st.button(tx["predict_btn"],use_container_width=True):
    row = {"drug_class":DRUG_CLASS_MAP[drug_info["class"]],"drug_risk_base":drug_info["risk_base"],"num_ingredients":num_ing,"patient_age_group":age_group,"has_kidney_issue":int(kidney),"has_liver_issue":int(liver),"has_heart_issue":int(heart),"has_diabetes":int(diabetes),"is_pregnant":int(pregnant),"dosage_level":dosage_level,"drug_interactions":interactions,"allergy_history":int(allergy)}
    inp = pd.DataFrame([row])
    for c in feature_cols:
        if c not in inp.columns: inp[c]=0
    inp = inp[feature_cols]
    pred  = model.predict(inp)[0]
    proba = model.predict_proba(inp)[0]
    risk_pct = round(max(proba)*100,1)

    st.markdown("---")
    st.subheader(tx["result_title"])

    risk_names  = {0:tx["low"],1:tx["medium"],2:tx["high"]}
    risk_colors = {0:"#00c853",1:"#ffa000",2:"#d32f2f"}

    m1,m2,m3 = st.columns(3)
    for col_obj,val,label,color in [
        (m1,risk_names[pred],tx["risk_level"],risk_colors[pred]),
        (m2,f"{risk_pct}%",tx["confidence"],"#00d4aa"),
        (m3,f"{accuracy*100:.1f}%",tx["accuracy"],"#7c83fd")]:
        with col_obj:
            st.markdown(f"<div style='background:#1e2130;padding:1rem;border-radius:10px;text-align:center'><div style='font-size:2rem;font-weight:700;color:{color}'>{val}</div><div style='font-size:0.85rem;color:#aaa'>{label}</div></div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    css = {0:"low",1:"med",2:"high"}[pred]
    emoji = {0:"🟢",1:"🟡",2:"🔴"}[pred]
    msg = {0:tx["low_msg"],1:tx["med_msg"],2:tx["high_msg"]}[pred]
    bg  = {0:"#1a3d2b",1:"#3d2e00",2:"#3d1a1a"}[pred]
    border = risk_colors[pred]
    st.markdown(f"<div style='padding:1.5rem;border-radius:12px;text-align:center;border:2px solid {border};background:{bg}'><h2>{emoji} {risk_names[pred]} RISK / அபாயம்</h2><p style='color:#ccc'><b>{drug_name}</b> {msg}</p></div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.subheader(tx["watch_title"])
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"**{tx['common_effects']}:**")
        for s in drug_info["common_side_effects"]:
            st.markdown(f"<div style='background:#1e2130;border:1px solid #333;padding:0.5rem 1rem;border-radius:8px;margin:0.3rem 0'>🟡 {s}</div>",unsafe_allow_html=True)
    with c2:
        st.markdown(f"**{tx['serious_effects']}:**")
        icon  = {0:"🔵",1:"⚠️",2:"🚨"}[pred]
        color = risk_colors[pred]
        for s in drug_info["serious_side_effects"]:
            st.markdown(f"<div style='background:#1e2130;border-left:3px solid {color};padding:0.5rem 1rem;border-radius:8px;margin:0.3rem 0'>{icon} {s}</div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.subheader(tx["recommend_title"])
    for r in tx["recs"][pred]:
        st.markdown(f"<div style='background:#1a1f35;border-left:4px solid #00d4aa;padding:1rem 1.5rem;border-radius:8px;margin:0.5rem 0'>{r}</div>",unsafe_allow_html=True)

    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    with c1:
        st.markdown(f"**{tx['prob_title']}**")
        fig,ax = plt.subplots(figsize=(5,3))
        fig.patch.set_facecolor('#1e2130'); ax.set_facecolor('#1e2130')
        risk_labels_chart = [f"🟢 {tx['low']}", f"🟡 {tx['medium']}", f"🔴 {tx['high']}"]
        bars = ax.barh(risk_labels_chart, proba, color=["#00c853","#ffa000","#d32f2f"], height=0.5)
        ax.set_xlim(0,1)
        ax.set_xlabel("← Less likely        More likely →", color="white", fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_visible(False)
        for bar,p in zip(bars,proba):
            ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
                    f"{p*100:.0f}% chance", va="center", color="white", fontsize=9)
        st.pyplot(fig)
        st.caption("This chart shows how likely each risk level is for this patient and drug combination.")

    with c2:
        st.markdown(f"**{tx['factors_title']}**")
        feat_imp = pd.Series(model.feature_importances_,index=feature_cols).sort_values(ascending=False).head(5)
        readable = {"drug_risk_base":"Drug's own risk","dosage_level":"Dosage strength","has_kidney_issue":"Kidney condition","has_liver_issue":"Liver condition","drug_interactions":"Other drugs taken","allergy_history":"Allergy history","patient_age_group":"Patient age","has_heart_issue":"Heart condition","has_diabetes":"Diabetes","is_pregnant":"Pregnancy","num_ingredients":"No. of ingredients","drug_class":"Drug type"}
        feat_imp.index = [readable.get(i,i) for i in feat_imp.index]
        fig2,ax2 = plt.subplots(figsize=(5,3))
        fig2.patch.set_facecolor('#1e2130'); ax2.set_facecolor('#1e2130')
        feat_imp.plot(kind="barh",ax=ax2,color="#00d4aa")
        ax2.set_xlabel("Impact on prediction →", color="white", fontsize=9)
        ax2.tick_params(colors="white"); ax2.invert_yaxis()
        for spine in ax2.spines.values(): spine.set_visible(False)
        st.pyplot(fig2)
        st.caption("Longer bar = this factor had more influence on the AI's decision.")

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown(f"**{tx['trend_title']}**")
    dosage_risks=[]
    for d in [1,2,3]:
        tr = row.copy(); tr["dosage_level"]=d
        ti = pd.DataFrame([tr])[feature_cols]
        dosage_risks.append(model.predict_proba(ti)[0][2]*100)
    fig3,ax3 = plt.subplots(figsize=(8,3))
    fig3.patch.set_facecolor('#1e2130'); ax3.set_facecolor('#1e2130')
    xlabels = [f"🟢 Low Dose\n(Below normal)","🟡 Medium Dose\n(As prescribed)","🔴 High Dose\n(Above normal)"]
    ax3.plot(xlabels, dosage_risks, color="#d32f2f", marker="o", linewidth=2.5, markersize=10)
    ax3.fill_between(xlabels, dosage_risks, alpha=0.15, color="#d32f2f")
    ax3.set_ylabel("High Risk Probability (%)", color="white")
    ax3.set_title(f"{drug_name} — Risk increases as dose increases", color="white", fontsize=10)
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values(): spine.set_visible(False)
    for x,y in zip(xlabels,dosage_risks):
        ax3.annotate(f"{y:.0f}% high risk",(x,y),textcoords="offset points",xytext=(0,12),ha="center",color="white",fontsize=9)
    st.pyplot(fig3)
    st.caption("This graph shows how the risk of serious side effects increases when the dose is increased beyond normal.")

    st.success(tx["disclaimer"])
