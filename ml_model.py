import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
import os

class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol',
            'fasting_bs', 'resting_ecg', 'max_hr', 'exercise_angina',
            'oldpeak', 'st_slope'
        ]
        self.model_path = 'heart_disease_model.pkl'
        self.scaler_path = 'heart_disease_scaler.pkl'
        
        # Load or train the model
        self._load_or_train_model()
    
    def _create_synthetic_training_data(self):
        """Create synthetic training data based on medical knowledge"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic health data
        data = []
        labels = []
        
        for i in range(n_samples):
            # Age: 20-80 years
            age = np.random.randint(20, 81)
            
            # Sex: 0=female, 1=male (males have higher risk)
            sex = np.random.randint(0, 2)
            
            # Chest pain type: 0-3 (higher types indicate more severe pain)
            chest_pain_type = np.random.randint(0, 4)
            
            # Resting BP: 90-200 mmHg
            resting_bp = np.random.randint(90, 201)
            
            # Cholesterol: 100-600 mg/dl
            cholesterol = np.random.randint(100, 601)
            
            # Fasting blood sugar > 120 mg/dl: 0=no, 1=yes
            fasting_bs = np.random.randint(0, 2)
            
            # Resting ECG: 0-2
            resting_ecg = np.random.randint(0, 3)
            
            # Max heart rate: 60-220 bpm
            max_hr = np.random.randint(60, 221)
            
            # Exercise induced angina: 0=no, 1=yes
            exercise_angina = np.random.randint(0, 2)
            
            # ST depression: 0-6.2
            oldpeak = np.random.uniform(0, 6.2)
            
            # ST slope: 0-2
            st_slope = np.random.randint(0, 3)
            
            # Calculate risk based on medical knowledge
            risk_score = 0
            
            # Age factor
            if age > 65:
                risk_score += 3
            elif age > 55:
                risk_score += 2
            elif age > 45:
                risk_score += 1
            
            # Sex factor (males higher risk)
            if sex == 1:
                risk_score += 1
            
            # Chest pain factor
            risk_score += chest_pain_type
            
            # Blood pressure factor
            if resting_bp > 160:
                risk_score += 3
            elif resting_bp > 140:
                risk_score += 2
            elif resting_bp > 120:
                risk_score += 1
            
            # Cholesterol factor
            if cholesterol > 300:
                risk_score += 3
            elif cholesterol > 240:
                risk_score += 2
            elif cholesterol > 200:
                risk_score += 1
            
            # Other factors
            if fasting_bs == 1:
                risk_score += 1
            if resting_ecg > 0:
                risk_score += 1
            if max_hr < 100:
                risk_score += 2
            if exercise_angina == 1:
                risk_score += 2
            if oldpeak > 2:
                risk_score += 2
            if st_slope > 1:
                risk_score += 1
            
            # Convert risk score to binary classification
            # Add some randomness to make it more realistic
            noise = np.random.normal(0, 1)
            final_risk = risk_score + noise
            
            heart_disease = 1 if final_risk > 6 else 0
            
            data.append([age, sex, chest_pain_type, resting_bp, cholesterol,
                        fasting_bs, resting_ecg, max_hr, exercise_angina,
                        oldpeak, st_slope])
            labels.append(heart_disease)
        
        return np.array(data), np.array(labels)
    
    def _train_model(self):
        """Train the heart disease prediction model"""
        logging.info("Training heart disease prediction model...")
        
        # Create synthetic training data
        X, y = self._create_synthetic_training_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Model training completed. Accuracy: {accuracy:.4f}")
        logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Save the model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return accuracy
    
    def _load_or_train_model(self):
        """Load existing model or train a new one"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logging.info("Loaded existing heart disease prediction model")
            else:
                self._train_model()
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self._train_model()
    
    def predict(self, health_data):
        """Make prediction based on health data"""
        try:
            # Convert health data to feature array
            features = np.array([[
                health_data.age,
                health_data.sex,
                health_data.chest_pain_type,
                health_data.resting_bp,
                health_data.cholesterol,
                health_data.fasting_bs,
                health_data.resting_ecg,
                health_data.max_hr,
                health_data.exercise_angina,
                health_data.oldpeak,
                health_data.st_slope
            ]])
            
            # Scale the features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            confidence = self.model.predict_proba(features_scaled)[0].max()
            
            logging.info(f"Prediction: {prediction}, Confidence: {confidence:.4f}")
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            # Return conservative prediction in case of error
            return 1, 0.5
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_features
    
    def explain_prediction(self, health_data):
        """Provide explanation for the prediction"""
        prediction, confidence = self.predict(health_data)
        
        explanations = []
        
        # Age factor
        if health_data.age > 65:
            explanations.append("Advanced age (>65) increases heart disease risk")
        elif health_data.age > 55:
            explanations.append("Age over 55 is a moderate risk factor")
        
        # Gender factor
        if health_data.sex == 1:
            explanations.append("Male gender is associated with higher heart disease risk")
        
        # Blood pressure
        if health_data.resting_bp > 160:
            explanations.append("High blood pressure (>160 mmHg) significantly increases risk")
        elif health_data.resting_bp > 140:
            explanations.append("Elevated blood pressure (>140 mmHg) is a risk factor")
        
        # Cholesterol
        if health_data.cholesterol > 300:
            explanations.append("Very high cholesterol (>300 mg/dl) is a major risk factor")
        elif health_data.cholesterol > 240:
            explanations.append("High cholesterol (>240 mg/dl) increases risk")
        
        # Exercise capacity
        if health_data.max_hr < 100:
            explanations.append("Low maximum heart rate may indicate poor cardiac fitness")
        
        # Exercise-induced symptoms
        if health_data.exercise_angina == 1:
            explanations.append("Exercise-induced chest pain is a significant warning sign")
        
        # ST depression
        if health_data.oldpeak > 2:
            explanations.append("Significant ST depression indicates possible coronary disease")
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'explanations': explanations
        }
