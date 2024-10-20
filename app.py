import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import google.generativeai as genai
import fitz  # PyMuPDF for PDF handling
import pytesseract  # Tesseract for OCR
from PIL import Image
from transformers import pipeline
import io


api_key="AIzaSyAVQbBfLAIbuv7rEJGqC5nOnPGmiuuKnBU"
genai.configure(api_key=api_key)
# Load the model
# Initialize the Flask application
app = Flask(__name__)

model = genai.GenerativeModel("gemini-1.5-flash")

model_path = './svc.pkl'  # Change this to the actual path where your model is saved
with open(model_path, 'rb') as file:
    svc = pickle.load(file)

# Load additional data needed for the response
description = pd.read_csv('./Dataset/description.csv')
precautions = pd.read_csv('./Dataset/precautions_df.csv')
medications = pd.read_csv('./Dataset/medications.csv')
diets = pd.read_csv('./Dataset/diets.csv')
workout = pd.read_csv('./Dataset/workout_df.csv')

description.columns = description.columns.str.strip()
precautions.columns = precautions.columns.str.strip()
medications.columns = medications.columns.str.strip()
diets.columns = diets.columns.str.strip()
workout.columns = workout.columns.str.strip()

# Define the symptoms dictionary
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
    'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91,
    'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101,
    'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104,
    'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114,
    'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
    'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119,
    'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
    'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
    'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# Define diseases list
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    14: 'Drug Reaction', 33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes ',
    17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ',
    30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue',
    37: 'Typhoid', 40: 'Hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C',
    21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis',
    10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemorrhoids(piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism',
    25: 'Hypoglycemia', 31: 'Osteoarthritis', 5: 'Arthritis', 0: '(vertigo) Paroxysmal Positional Vertigo',
    2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'
}



summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    images = []

    # Loop through all pages in the PDF
    for page_num in range(doc.page_count):
        page = doc[page_num]
        # Extract text from the page
        full_text += page.get_text("text")
        
        # Extract images from the page
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(Image.open(io.BytesIO(image_bytes)))

    doc.close()
    return full_text, images

def extract_text_from_images(images):
    ocr_text = ""
    for img in images:
        ocr_text += pytesseract.image_to_string(img) + "\n"
    return ocr_text

# Function to chunk the text for summarization
def chunk_text(text, max_chunk_size=1024):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for space
        if current_length >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Function to summarize each chunk of text
def summarize_chunks(chunks):
    summarized_text = ""
    for chunk in chunks:
        try:
            # Summarize each chunk
            summary = summarizer(chunk, max_length=150, min_length=25, do_sample=False)
            summarized_text += summary[0]['summary_text'] + " "
        except Exception as e:
            summarized_text += f"\nError in summarization: {str(e)}"
    return summarized_text

# Main function to process the uploaded PDF
def process_upload(file_path):
    try:
        # Extract text and images from the PDF
        text, images = extract_text_from_pdf(file_path)

        # Extract text from the images using OCR
        ocr_text = extract_text_from_images(images)

        # Combine text from the PDF and from OCR
        full_text = text + "\n" + ocr_text

        # Check if full_text is non-empty and large enough
        if len(full_text.strip()) == 0:
            return "No extractable text found in the PDF."

        # Chunk the text if it's too long
        chunks = chunk_text(full_text)

        # Summarize each chunk
        summary = summarize_chunks(chunks)

        return full_text, summary

    except Exception as e:
        return f"Error in processing PDF: {str(e)}", None




# generate the summary
def get_summary(extracted_text):
    
    # Make the request to the Gemini API
    response = model.generate_content(f"Give the Summary for the PDF File extracted text: {extracted_text}")
    return response.text

# Define prediction function
def given_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Define helper function for detailed info
def helper(predicted_disease):
    desc = description[description['Disease'] == predicted_disease]['Description'].values
    desc = " ".join(desc) if desc else "No description available."

    # Check if 'Disease' exists in precautions DataFrame
    if 'Disease' in precautions.columns:
        pre = precautions[precautions['Disease'] == predicted_disease].iloc[:, 1:6]
    else:
        pre = pd.DataFrame()  # Create an empty DataFrame if 'Disease' doesn't exist

    pre_list = pre.dropna().values.flatten().tolist()

    # Repeat similar checks for medications, diets, and workout DataFrames
    if 'Disease' in medications.columns:
        med = medications[medications['Disease'] == predicted_disease].iloc[:, 1:6]
    else:
        med = pd.DataFrame()

    med_list = med.dropna().values.flatten().tolist()

    if 'Disease' in diets.columns:
        diet = diets[diets['Disease'] == predicted_disease].iloc[:, 1:6]
    else:
        diet = pd.DataFrame()

    diet_list = diet.dropna().values.flatten().tolist()

    if 'Disease' in workout.columns:
        workout_info = workout[workout['Disease'] == predicted_disease].iloc[:, 1:6]
    else:
        workout_info = pd.DataFrame()

    workout_list = workout_info.dropna().values.flatten().tolist()

    response = {
        'disease': predicted_disease,
        'description': desc,
        'precautions': pre_list,
        'medications': med_list,
        'workout': workout_list,
        'diets': diet_list
    }
    return response

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symptoms = data.get('symptoms', [])
    
    predicted_disease = given_predicted_value(symptoms)
    response = helper(predicted_disease)
    
    return jsonify(response)


@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Check if the request has a file
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file"}), 400
            
            # Extract text from the PDF file
            extracted_text, summary = process_upload(file)
        
        # Check if the request has text data
        elif 'text' in request.json:
            extracted_text = request.json.get('text')
            if not extracted_text:
                return jsonify({"error": "No text provided"}), 400
        else:
            return jsonify({"error": "No file or text provided"}), 400
        
        # Get the summary
        summary = get_summary(extracted_text)
        
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
