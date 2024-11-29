from flask import Flask, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import google.generativeai as genai
from PIL import Image


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Configure upload folder
UPLOAD_FOLDER = 'C:\\Users\\sudhu\\OneDrive\\Desktop\\disease_prediction\\uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the models
MODEL_PATH_1 = 'C:\\Users\\sudhu\\OneDrive\\Desktop\\disease_prediction\\Apple-leaf-MobileVnet.h5'
MODEL_PATH_2 = 'C:\\Users\\sudhu\\OneDrive\\Desktop\\disease_prediction\\Corn-leaf-MobileVnet.h5'
MODEL_PATH_3 = 'C:\\Users\\sudhu\\OneDrive\\Desktop\\disease_prediction\\Grape-leaf-MobileVnet.h5'
MODEL_PATH_4 = 'C:\\Users\\sudhu\\OneDrive\\Desktop\\disease_prediction\\Potato-leaf-MobileVnet.h5'
MODEL_PATH_5 = 'C:\\Users\\sudhu\\OneDrive\\Desktop\\disease_prediction\\Paddy-leaf-MobileVnet.h5'

model1 = load_model(MODEL_PATH_1)
model2 = load_model(MODEL_PATH_2)
model3 = load_model(MODEL_PATH_3)
model4 = load_model(MODEL_PATH_4)
model5 = load_model(MODEL_PATH_5)

# Define the class names
CLASS_NAMES_1 = ['Apple__Apple_scab', 'Apple__Black_rot', 'Apple__Cedar_apple_rust', 'Apple__healthy', 'Potato__Early_blight', 'Potato__Late_Blight']
CLASS_NAMES_2 = ['Corn_(maize)__healthy', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn_(maize)__Common_rust_', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot']
CLASS_NAMES_3 = ['Grapes__Black_rot', 'Grapes__Esca_(Black_Measles)', 'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape__healthy']
CLASS_NAMES_4 = ['Potato__Early_blight', 'Potato__Late_blight', 'Potato__healthy']
CLASS_NAMES_5 = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

@app.route('/submit', methods=['POST'])
def submit_form():
    global history
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    image = request.files['image']
    image_filename = image.filename
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

    try:
        image.save(image_path)
        print(f"Image saved to: {image_path}")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    # Predict the disease
    name = request.form['name']
    prediction = predict_disease(image_path, name)

    history.append({
        "role": "user",
        "parts": [prediction]
    })

    return jsonify({'message': 'Form data has been saved to the database', 'prediction': prediction, 'image_url': f'/uploads/{image_filename}'}), 200

def predict_disease(image_path, name):
    global history
    try:
        if name.lower()=='paddy':
            img = load_img(image_path, target_size=(256, 256))
        else:
            img = load_img(image_path, target_size=(224, 224))
  # Resize to 256x256
  # Adjust target_size as per your models
    except Exception as e:
        return f"Error loading image: {e}"

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if needed

    # Select the model based on the name
    if name.lower() in ['corn', 'maize']:
        predictions = model2.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        return CLASS_NAMES_2[predicted_class]
    elif name.lower() == 'grapes':
        predictions = model3.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        return CLASS_NAMES_3[predicted_class]
    elif name.lower() == 'potato':
        predictions = model4.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        return CLASS_NAMES_4[predicted_class]
    elif name.lower() == 'paddy':
        predictions = model5.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        return CLASS_NAMES_5[predicted_class]
    else:
        predictions = model1.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        return CLASS_NAMES_1[predicted_class]

# GEMINI API Configuration
genai.configure(api_key='AIzaSyDpOc3A020S4XpEAYXQ-_zgwaBUpPtQB_A')

# Initialize history list to maintain conversation history
history = []

# Create the model with configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 150,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-exp-0827",
    generation_config=generation_config,
    system_instruction="You're a chatbot integrated in a plant disease prediction software. You will be provided a predicted disease. Then you have to answer the farmers' questions related to that disease. You should not mention 'I know about it' or similar. You should reply as if you already know. You should give the causes, symptoms, and solutions for that disease. If you don't know the predicted disease, tell the user that due to some technical error, the predicted disease was not found and ask 'Can you please provide me the name of the predicted disease?' Then provide the solution, symptoms for the disease. Your name is Groot AI. Use emojis to make the replies attractive. Keep in mind that the users are farmers. Do not reply to anything other than plant disease and solution.",
)

# Initialize the chat session as None globally
chat_session = None

@app.route('/chatbot', methods=['POST'])
def chatbot():
    global history, chat_session

    user_message = request.json.get('message')

    # Append the user's message to the history
    history.append({
        "role": "user",
        "parts": [user_message]
    })

    if chat_session is None:
        chat_session = model.start_chat(history=history)
    else:
        chat_session.history = history

    response = chat_session.send_message(user_message)
    
    # Break long responses into smaller segments
    response_text = response.text
    max_characters_per_message = 200  # Adjust this based on screen size

    response_parts = [response_text[i:i + max_characters_per_message] for i in range(0, len(response_text), max_characters_per_message)]

    history.append({
        "role": "model",
        "parts": response_parts
    })

    # Send the first part of the response and inform the frontend to show the rest gradually if needed
    return jsonify({"response": response_parts[0], "followup": response_parts[1:] if len(response_parts) > 1 else []})

@app.route('/uploads/<filename>')
def serve_image(filename):
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
        else:
            abort(404)
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        abort(500)

@app.route('/')
def index():
    return send_from_directory('C:\\Users\\sudhu\\OneDrive\\Desktop\\disease_prediction\\plant-disease-detection\\public', 'index.html')


if __name__ == '__main__':
    app.run(debug=True)
