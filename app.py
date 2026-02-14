from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

app = Flask(__name__)

# Global variables for models
crop_model = None
fert_model = None
soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']
fertilizers = ['Urea', 'DAP', '14-35-14', '28-28', '17-17-17', '20-20', '10-26-26']

def initialize_models():
    """Initialize models with error handling"""
    global crop_model, fert_model
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        import joblib
        
        # Check if models already exist
        if os.path.exists('crop_model.pkl') and os.path.exists('fertilizer_model.pkl'):
            print("Loading existing models...")
            crop_model = joblib.load('crop_model.pkl')
            fert_model = joblib.load('fertilizer_model.pkl')
            global crop_le, fert_le, soil_le
            crop_le = joblib.load('crop_label_encoder.pkl')
            fert_le = joblib.load('fertilizer_label_encoder.pkl')
            soil_le = joblib.load('soil_label_encoder.pkl')
            print(" Models loaded successfully!")
            return True
        
        # Generate training data
        print("Generating training data...")
        dummy_data = []
        for _ in range(1000):
            temp = round(random.uniform(15, 40), 2)
            humidity = round(random.uniform(40, 90), 2)
            moisture = round(random.uniform(20, 80), 2)
            soil = random.choice(soil_types)
            nitrogen = round(random.uniform(0, 100), 2)
            potassium = round(random.uniform(0, 100), 2)
            phosphorous = round(random.uniform(0, 100), 2)
            crop = random.choice(crop_types)
            fertilizer = random.choice(fertilizers)
            dummy_data.append([temp, humidity, moisture, soil, crop, nitrogen, potassium, phosphorous, fertilizer])
        
        df = pd.DataFrame(dummy_data, columns=['Temparature', 'Humidity', 'Moisture', 'Soil Type', 
                                                'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous', 
                                                'Fertilizer Name'])
        
        # Encode categorical variables
        label_encoders = {}
        for col in ['Soil Type', 'Crop Type', 'Fertilizer Name']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Train crop model
        print("Training crop prediction model...")
        X_crop = df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
        y_crop = df['Crop Type']
        crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
        crop_model.fit(X_crop, y_crop)
        
        # Train fertilizer model
        print("Training fertilizer prediction model...")
        X_fert = df[['Temparature', 'Humidity', 'Moisture', 'Soil Type', 'Crop Type', 
                     'Nitrogen', 'Potassium', 'Phosphorous']]
        y_fert = df['Fertilizer Name']
        fert_model = RandomForestClassifier(n_estimators=100, random_state=42)
        fert_model.fit(X_fert, y_fert)
        
        # Save models
        joblib.dump(crop_model, 'crop_model.pkl')
        joblib.dump(fert_model, 'fertilizer_model.pkl')
        joblib.dump(label_encoders['Crop Type'], 'crop_label_encoder.pkl')
        joblib.dump(label_encoders['Fertilizer Name'], 'fertilizer_label_encoder.pkl')
        joblib.dump(label_encoders['Soil Type'], 'soil_label_encoder.pkl')
        
        crop_le = label_encoders['Crop Type']
        fert_le = label_encoders['Fertilizer Name']
        soil_le = label_encoders['Soil Type']
        
        print(" Models trained and saved successfully!")
        return True
        
    except Exception as e:
        print(f" Error initializing models: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    try:
        data = request.json
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        moisture = float(data['moisture'])
        soil_type = data['soil_type']
        nitrogen = float(data['nitrogen'])
        potassium = float(data['potassium'])
        phosphorous = float(data['phosphorous'])
        
        # Encode soil type
        soil_encoded = soil_le.transform([soil_type])[0]
        
        # Prepare input
        input_data = np.array([[temp, humidity, moisture, soil_encoded, nitrogen, potassium, phosphorous]])
        
        # Predict
        crop_pred = crop_model.predict(input_data)
        crop_name = crop_le.inverse_transform(crop_pred)[0]
        
        # Get prediction probability
        proba = crop_model.predict_proba(input_data)
        confidence = round(max(proba[0]) * 100, 2)
        
        return jsonify({
            'success': True,
            'crop': crop_name,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        data = request.json
        temp = float(data['temperature'])
        humidity = float(data['humidity'])
        moisture = float(data['moisture'])
        soil_type = data['soil_type']
        crop_type = data['crop_type']
        nitrogen = float(data['nitrogen'])
        potassium = float(data['potassium'])
        phosphorous = float(data['phosphorous'])
        
        # Encode categorical variables
        soil_encoded = soil_le.transform([soil_type])[0]
        crop_encoded = crop_le.transform([crop_type])[0]
        
        # Prepare input
        input_data = np.array([[temp, humidity, moisture, soil_encoded, crop_encoded, 
                               nitrogen, potassium, phosphorous]])
        
        # Predict
        fert_pred = fert_model.predict(input_data)
        fert_name = fert_le.inverse_transform(fert_pred)[0]
        
        # Get prediction probability
        proba = fert_model.predict_proba(input_data)
        confidence = round(max(proba[0]) * 100, 2)
        
        return jsonify({
            'success': True,
            'fertilizer': fert_name,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_options', methods=['GET'])
def get_options():
    try:
        return jsonify({
            'soil_types': soil_types,
            'crop_types': crop_types
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'running',
        'models_loaded': crop_model is not None and fert_model is not None
    })

if __name__ == '__main__':
    print("=" * 50)
    print(" Soil Detection System Starting...")
    print("=" * 50)
    
    # Initialize models
    if initialize_models():
        print("\n" + "=" * 50)
        print(" Server is ready!")
        print(" Open your browser and go to: http://127.0.0.1:5000")
        print("=" * 50 + "\n")
        app.run(debug=True, port=5000, use_reloader=False)
    else:
        print(" Failed to initialize models. Please install required packages:")
        print("pip install flask pandas numpy scikit-learn joblib")
