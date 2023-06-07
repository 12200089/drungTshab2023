from flask import Flask, render_template,jsonify
import joblib
from flask import Flask, render_template, redirect, url_for, request, send_from_directory
from flask import Flask, render_template, request, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
# from keras_preprocessing import brain
# from keras.models import load_model
import sys
import os
import glob
import numpy as np
from markupsafe import escape
import uuid
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Frame, Table, TableStyle


global_user_inputs = []
global_response = ""
# Load the trained disease prediction model
random = RandomForestClassifier()
# model.load('rf_model.pkl')
import pickle
# Load the pickle model from a file
with open('random_final_m20.pkl', 'rb') as f:
    model5 = pickle.load(f)

import csv
# Load the diagnoses and treatments from a CSV file
diagnoses = {}
with open('des_treat.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        diagnoses[row['diagnosis']] = [row['description'], row['treatment']]

app = Flask(__name__)  #instance of flask
app.debug = True

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    intent = req['queryResult']['intent']['displayName']

    global global_user_inputs, global_response  

    if intent == 'initiate_conversation-yes':
       
        response_map = {'yes': 1, 'no': 0}

        symptom1 = response_map[req['queryResult']['parameters']['symptom1']] if 'symptom1' in req['queryResult']['parameters'] else 0
        symptom2 = response_map[req['queryResult']['parameters']['symptom2']] if 'symptom2' in req['queryResult']['parameters'] else 0
        symptom3 = response_map[req['queryResult']['parameters']['symptom3']] if 'symptom3' in req['queryResult']['parameters'] else 0
        symptom4 = response_map[req['queryResult']['parameters']['symptom4']] if 'symptom4' in req['queryResult']['parameters'] else 0
        symptom5 = response_map[req['queryResult']['parameters']['symptom5']] if 'symptom5' in req['queryResult']['parameters'] else 0
        symptom6 = response_map[req['queryResult']['parameters']['symptom6']] if 'symptom6' in req['queryResult']['parameters'] else 0
        symptom7 = response_map[req['queryResult']['parameters']['symptom7']] if 'symptom7' in req['queryResult']['parameters'] else 0
        symptom8 = response_map[req['queryResult']['parameters']['symptom8']] if 'symptom8' in req['queryResult']['parameters'] else 0
        symptom9 = response_map[req['queryResult']['parameters']['symptom9']] if 'symptom9' in req['queryResult']['parameters'] else 0
        symptom10 = response_map[req['queryResult']['parameters']['symptom10']] if 'symptom10' in req['queryResult']['parameters'] else 0
        symptom11 = response_map[req['queryResult']['parameters']['symptom11']] if 'symptom11' in req['queryResult']['parameters'] else 0
        symptom12 = response_map[req['queryResult']['parameters']['symptom12']] if 'symptom12' in req['queryResult']['parameters'] else 0
        symptom13 = response_map[req['queryResult']['parameters']['symptom13']] if 'symptom13' in req['queryResult']['parameters'] else 0
        symptom14 = response_map[req['queryResult']['parameters']['symptom14']] if 'symptom14' in req['queryResult']['parameters'] else 0
        symptom15 = response_map[req['queryResult']['parameters']['symptom15']] if 'symptom15' in req['queryResult']['parameters'] else 0
        symptom16 = response_map[req['queryResult']['parameters']['symptom16']] if 'symptom16' in req['queryResult']['parameters'] else 0
        symptom17 = response_map[req['queryResult']['parameters']['symptom17']] if 'symptom17' in req['queryResult']['parameters'] else 0
        symptom18 = response_map[req['queryResult']['parameters']['symptom18']] if 'symptom18' in req['queryResult']['parameters'] else 0
        symptom19 = response_map[req['queryResult']['parameters']['symptom19']] if 'symptom19' in req['queryResult']['parameters'] else 0
        symptom20 = response_map[req['queryResult']['parameters']['symptom20']] if 'symptom20' in req['queryResult']['parameters'] else 0
   

        global_user_inputs = []  # List to store user inputs
        # Append user inputs to the list
        global_user_inputs.append(('1. Are you suffering from fever?', convert_symptom_value(symptom1)))
        global_user_inputs.append(('2. Are you feeling nauseous or experiencing any urge to vomit?', convert_symptom_value(symptom2)))
        global_user_inputs.append(('3. Do you experience feelings of fatigue or exhaustion?', convert_symptom_value(symptom3)))
        global_user_inputs.append(('4. Are you suffering from headache?', convert_symptom_value(symptom4)))
        global_user_inputs.append(('5. Are you having joint pain?', convert_symptom_value(symptom5)))
        global_user_inputs.append(('6. Are you experiencing sign of chest pain?', convert_symptom_value(symptom6)))
        global_user_inputs.append(('7. Are you experiencing any itching sensation?', convert_symptom_value(symptom7)))
        global_user_inputs.append(('8. Are you experiencing loss of appetite?', convert_symptom_value(symptom8)))
        global_user_inputs.append(('9. Are you suffering from stomach pain?', convert_symptom_value(symptom9)))
        global_user_inputs.append(('10. Are you experiencing any sign of abdominal pain?', convert_symptom_value(symptom10)))
        global_user_inputs.append(('11. Are you experiencing any feelings of nausea or discomfort in your stomach?', convert_symptom_value(symptom11)))
        global_user_inputs.append(('12. Are you feeling dizzy or unstable on your feet / loss of balance?', convert_symptom_value(symptom12)))
        global_user_inputs.append(('13. Are you suffering from diarrhea?', convert_symptom_value(symptom13)))
        global_user_inputs.append(('14. Do you notice any abnormal yellow coloring in your eyes?', convert_symptom_value(symptom14)))
        global_user_inputs.append(('15. Well, are you experiencing any indications of muscle pain?', convert_symptom_value(symptom15)))
        global_user_inputs.append(('16. Are you feeling generally unwell or experiencing any symptoms of discomfort or malaise?', convert_symptom_value(symptom16)))
        global_user_inputs.append(('17. Are you noticing any indication of neck pain?', convert_symptom_value(symptom17)))
        global_user_inputs.append(('18. Are you experiencing any difficulties or irregularities in passing stool or constipation?', convert_symptom_value(symptom18)))
        global_user_inputs.append(('19. Are you experiencing any symptoms of feeling cold or shivering?', convert_symptom_value(symptom19)))
        global_user_inputs.append(('20. Are you experiencing a constant urge to urinate frequently?', convert_symptom_value(symptom20)))

        # Call your disease prediction model with the input symptoms
        diagnosis, description, treatment = predict_disease(symptom1, symptom2, symptom3, symptom4, symptom5,symptom6, symptom7, symptom8, symptom9, symptom10,symptom11, symptom12, symptom13, symptom14, symptom15,symptom16, symptom17, symptom18, symptom19, symptom20)

        response = {
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [
                            "Based on your symptoms, it looks like you have " + diagnosis +
                            ".\n\nKnow more about " + diagnosis + ": \n " + description + 
                            ".\n\nI recommend the following treatment: \n" + treatment + "."
                        ]
                    }
                },
                
                {
                    "text": {
                        "text": [
                            "Do you want to download the result as a PDF file?"
                        ]
                    }
                }
            ]
        }

        global_response = response

    elif intent == 'download-result-yes':
    
        if global_user_inputs and global_response:
        

            pdf_filename=generate_pdf(global_user_inputs, global_response)

            # Step 5: Prepare a response message confirming the download
            response = {

                "fulfillmentMessages": [
                    {
                        "text": {
                            "text": [
                                # "The result has been downloaded as "+pdf_file+"."
                                "The result has been downloaded as "+pdf_filename+"."
                            ]
                        }
                    }
                ]
            }
        else:
            # Step 6: Prepare a response message indicating missing information
            response  = {
                "fulfillmentMessages": [
                    {
                        "text": {
                            "text": [
                                global_user_inputs,  global_response
                            ]
                        }
                    }
                ]
            }
            
        return jsonify(response)


    elif intent == 'Q&A':
        question = req['queryResult']['parameters']['question']
        treatment = req['queryResult']['parameters'].get('treatment')
        
        # Call your Q&A function
        question, treatment, ans, treat = Ques_Ans(question, treatment)

        if treatment:
            response = {
                "fulfillmentMessages": [
                    {
                        "text": {
                            "text": [
                                "The solution for "+ question + " are : " + treat
                            ]
                        }
                    }
                ]
            }
        else:
            response = {
                "fulfillmentMessages": [
                    {
                        "text": {
                            "text": [
                                question + ". \n\n " + ans
                            ]
                        }
                    }
                ]
            }
            print(response)

    elif intent == 'my_abilities - custom':
        response = {
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [
                            "I can predict following diseases \n 1. Malaria,   \n  2. Allergy \n 3. Hypothyroidism, \n  4. Psoriasis \n 5. GERD, \n  6. Chronic cholestasis \n 7. hepatitis A, \n  8. Osteoarthristis \n 9. (vertigo) Paroymsal  Positional Vertigo,\n10. Hypoglycemia, \n 11. Acne,   \n  12. Diabetes \n 13. Impetigo,  \n 14. Hypertension \n 15. Peptic ulcer diseae, \n  16. Dimorphic hemmorhoids(piles) \n 17. Common Cold, \n  18. Chicken pox \n 19. Cervical spondylosis, \n  20. Hyperthyroidism \n 21. Urinary tract infection,\n22. Varicose veins \n 23. AIDS, \n  24. Paralysis (brain hemorrhage) \n 25. Typhoid, \n  26. Hepatitis B \n 27. Fungal infection, \n  28. Hepatitis C \n 29. Migraine, \n  30. Bronchial Asthma \n 31. Alcoholic hepatitis, \n  32. Jaundice \n 33. Hepatitis E, \n  34. Dengue \n 35. Hepatitis D,\n 36. Heart attack, \n  37. Pneumonia \n 38. Arthritis, \n  39. Gastroenteritis \n 40. Tuberculosis, \n 41. Drug Reaction"
                        ]
                    }
                }
            ]
        }
    
    else:
        response = {
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [
                            "Sorry, I couldn't understand your query."
                        ]
                    }
                }
            ]
        }

    return jsonify(response)

def convert_symptom_value(symptom):
    if symptom == 1:
        return "yes"
    else:
        return "no"
    
def generate_pdf(user_inputs, response):
    # Create a PDF file
    random_name = str(uuid.uuid4())[:3]  # Use the first 3 characters of the UUID for a random name
    pdf_filename = f"result_{random_name}.pdf"
    pdf = SimpleDocTemplate(pdf_filename, pagesize=letter)

    # Set up the styles for paragraph formatting
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle('heading_style', parent=styles['Title'], spaceAfter=0.4*inch)
    response_style = ParagraphStyle('response_style', parent=styles['Normal'], spaceAfter=0.2*inch)

    # Build the content for the PDF
    content = []

    # Add heading title to the content
    heading_text = 'Your Prediction Result'
    heading = Paragraph(heading_text, heading_style)
    content.append(heading)

    # Add user inputs to the content
    for label, value in user_inputs:
        input_text = f"{label}: {value}"
        input_paragraph = Paragraph(input_text, styles['Normal'])
        content.append(input_paragraph)

    # Add a separator
    content.append(Spacer(1, 0.2*inch))

    # Add the response to the content
    response_text = response["fulfillmentMessages"][0]["text"]["text"][0]
    response_lines = response_text.split('\n')
    for line in response_lines:
        response_paragraph = Paragraph(line, response_style)
        content.append(response_paragraph)

    # Create a table to enclose the content with a border
    table_data = [[cell] for cell in content]
    table_style = TableStyle([('BOX', (0, 0), (-1, -1), 1, colors.black)])
    table = Table(table_data, style=table_style, colWidths=500)

    # Build the PDF by adding the table
    pdf.build([table])

    return pdf_filename

def Ques_Ans(question, treatment):
    
    if question in diagnoses:
        ans = diagnoses[question][0]
        treat = diagnoses[question][1]
            
    else:       
        ans = 'No answer available \n\nI can give answer related to \n 1. Malaria,   \n  2. Allergy \n 3. Hypothyroidism, \n  4. Psoriasis \n 5. GERD, \n  6. Chronic cholestasis \n 7. hepatitis A, \n  8. Osteoarthristis \n 9. (vertigo) Paroymsal  Positional Vertigo,\n10. Hypoglycemia, \n 11. Acne,   \n  12. Diabetes \n 13. Impetigo,  \n 14. Hypertension \n 15. Peptic ulcer diseae, \n  16. Dimorphic hemmorhoids(piles) \n 17. Common Cold, \n  18. Chicken pox \n 19. Cervical spondylosis, \n  20. Hyperthyroidism \n 21. Urinary tract infection,\n22. Varicose veins \n 23. AIDS, \n  24. Paralysis (brain hemorrhage) \n 25. Typhoid, \n  26. Hepatitis B \n 27. Fungal infection, \n  28. Hepatitis C \n 29. Migraine, \n  30. Bronchial Asthma \n 31. Alcoholic hepatitis, \n  32. Jaundice \n 33. Hepatitis E, \n  34. Dengue \n 35. Hepatitis D,\n 36. Heart attack, \n  37. Pneumonia \n 38. Arthritis, \n  39. Gastroenteritis \n 40. Tuberculosis, \n 41. Drug Reaction \n  If there is no answer, you might have made mistake in spelling and letter case, please checked and retry '
        treat = 'No treatment available'

    return question, treatment, ans, treat

def predict_disease(symptom1, symptom2, symptom3, symptom4, symptom5, symptom6, symptom7, symptom8, symptom9, symptom10, symptom11, symptom12, symptom13, symptom14, symptom15, symptom16, symptom17, symptom18, symptom19, symptom20):
    # Convert the input symptoms to a feature vector
    features = [symptom1, symptom2, symptom3, symptom4, symptom5, symptom6, symptom7, symptom8, symptom9, symptom10, symptom11, symptom12, symptom13, symptom14, symptom15, symptom16, symptom17, symptom18, symptom19, symptom20]
  
    diagnosis = model5.predict([features])[0]

    if diagnosis in diagnoses:
        description = diagnoses[diagnosis][0]
        treatment = diagnoses[diagnosis][1]
       
    else:
        treatment = 'No treatment recommendation'
        description = 'No description available'

    return diagnosis,description, treatment



filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model1 = pickle.load(open('heartdisease_model.sav', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

os.environ["CUDA_VISIBLE_DEVICES"]="-1"


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['brainimg'],filename)

@app.route("/")  
def base():
    return render_template('home.html')

@app.route("/service")
def service():
    return render_template('service.html')

@app.route("/about")
def about():
    return render_template('about.html') 
  
@app.route("/contact")
def contact():
    return render_template('contact.html')

@app.route("/voice")
def voice():
    return render_template('voice.html')

@app.route("/form")
def form():
    return render_template('form.html')

@app.route("/BreastC")
# @login_required
def cancer():
    return render_template("BreastC.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
    if output == 4:
        res_val = "a high risk of Breast Cancer"
        coment="Keep a healthy weight. Be physically active. Choose not to drink alcohol, or drink alcohol in moderation. If you are taking, or have been told to take, hormone replacement therapy or oral contraceptives (birth control pills), ask your doctor about the risks and find out if it is right for you."
    else:
        res_val = "a low risk of Breast Cancer"
        coment="You should limit alcohol, Maintain a healthy weight, Be physically active,Breast-feed dan Limit postmenopausal hormone therapy. Maintaining a healthy weight also is a key factor in breast cancer prevention."

    return render_template('breastC_result.html', prediction_text='Patient has {}'.format(res_val), coment=coment)


@app.route("/diabetes")
# @login_required
def diabetes():
    return render_template("diabetes.html")

##################################################################################

df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building

X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#####################################################################


@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('dia_result.html', prediction=my_prediction)


############################################################################################################

@app.route("/heart")
# @login_required
def heart():
    return render_template("heart.html")

@app.route('/predictheart', methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

   
    features_name = ["age", "sex", "cp", "thalach","exang", "oldpeak", "slope", "ca" ,"thal"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model1.predict(df)

    if output == 1:
        res_val = "a high risk of Heart Disease"
        recommend="Regular physical activity can lower your risk for heart disease. Drinking too much alcohol can raise blood pressure levels and the risk for heart disease. It also increases levels of triglycerides, a fatty substance in the blood which can increase the risk for heart disease."
    else:
        res_val = "a low risk of Heart Disease"
        recommend ="Be sure to eat plenty of fresh fruits and vegetables and fewer processed foods. Eating lots of foods high in saturated fat and trans fat may contribute to heart disease. Eating foods high in fiber and low in saturated fats, trans fat, and cholesterol can help prevent high cholesterol."

    return render_template('heart_result.html', Heart_prediction_text='Patient has {}'.format(res_val),recommend=recommend)




# model4 = load_model("BT.h5")


# def model_predict(img_path, model4):
#     img = brain.load_img(img_path, target_size=(200,200)) #target_size must agree with what the trained model expects!!

#     # Preprocessing the image
#     img = image.img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = img.astype('float32')/255
   
#     preds = model4.predict(img)

   
   
#     pred = np.argmax(preds,axis = 1)
#     return pred


# @app.route('/BrainTumor', methods=['GET'])
# def BrainTumor():
#     return render_template("BrainTumor.html")


# @app.route('/Bpredict', methods=['GET', 'POST'])
# def upload():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         pred = model_predict(file_path, model4)
#         os.remove(file_path)#removes file from the server after prediction has been returned

#         # Arrange the correct return according to the model. 
# 		# In this model 1 is Pneumonia and 0 is Normal.
#         str0 = 'Base on your MRI image it looks like you are suffering from Glioma Brain Tumor'
#         str1 = 'Base on your MRI image it looks like you are suffering from Meningioma Brain Tumor'
#         str3 = 'Base on your MRI image it looks like you are suffering from pituitary Brain Tumor'
#         str2 = 'Base on your MRI image it looks like you are not suffering from Brain Tumour'
#         if pred[0] == 0:
#             return str0
#         elif pred[0] == 1:
#             return str1
#         elif pred[0]==3:
#             return str3
#         else:
#             return str2
#     return None


@app.route("/learn")
# @login_required
def learn():
    return render_template("LearnDi.html")

if __name__ == '__main__':
    app.run( debug=True)