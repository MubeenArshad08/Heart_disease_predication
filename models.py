from datetime import datetime
from app import db
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(20))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    appointments = db.relationship('Appointment', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class HealthData(db.Model):
    __tablename__ = 'health_data'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)  # 1 = male, 0 = female
    chest_pain_type = db.Column(db.Integer, nullable=False)  # 0-3
    resting_bp = db.Column(db.Integer, nullable=False)  # Resting blood pressure
    cholesterol = db.Column(db.Integer, nullable=False)  # Serum cholesterol
    fasting_bs = db.Column(db.Integer, nullable=False)  # Fasting blood sugar > 120 mg/dl
    resting_ecg = db.Column(db.Integer, nullable=False)  # Resting ECG results
    max_hr = db.Column(db.Integer, nullable=False)  # Maximum heart rate achieved
    exercise_angina = db.Column(db.Integer, nullable=False)  # Exercise induced angina
    oldpeak = db.Column(db.Float, nullable=False)  # ST depression induced by exercise
    st_slope = db.Column(db.Integer, nullable=False)  # Slope of peak exercise ST segment
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<HealthData {self.id} for User {self.user_id}>'

class Prediction(db.Model):
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    health_data_id = db.Column(db.Integer, db.ForeignKey('health_data.id'), nullable=False)
    prediction_result = db.Column(db.Integer, nullable=False)  # 0 = no disease, 1 = disease
    confidence_score = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)  # Low, Medium, High
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    health_data = db.relationship('HealthData', backref='predictions')
    
    def __repr__(self):
        return f'<Prediction {self.id} - Result: {self.prediction_result}>'

class Appointment(db.Model):
    __tablename__ = 'appointments'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    doctor_name = db.Column(db.String(100), nullable=False)
    appointment_date = db.Column(db.DateTime, nullable=False)
    appointment_time = db.Column(db.String(10), nullable=False)
    reason = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, confirmed, rejected, completed, cancelled
    doctor_notes = db.Column(db.Text)  # Doctor's notes or rejection reason
    confirmation_date = db.Column(db.DateTime)  # When doctor confirmed/rejected
    confirmed_by = db.Column(db.String(100))  # Doctor who confirmed
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    prediction = db.relationship('Prediction', backref='appointments')
    
    def __repr__(self):
        return f'<Appointment {self.id} with Dr. {self.doctor_name}>'
    
    @property
    def is_confirmed(self):
        return self.status == 'confirmed'
    
    @property
    def is_pending(self):
        return self.status == 'pending'
    
    @property
    def is_rejected(self):
        return self.status == 'rejected'
    
    @property
    def status_badge_color(self):
        status_colors = {
            'pending': 'warning',
            'confirmed': 'success',
            'rejected': 'danger',
            'completed': 'info',
            'cancelled': 'secondary'
        }
        return status_colors.get(self.status, 'secondary')

class Notification(db.Model):
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    appointment_id = db.Column(db.Integer, db.ForeignKey('appointments.id'), nullable=True)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), nullable=False)  # appointment_confirmed, appointment_rejected, etc.
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='notifications')
    appointment = db.relationship('Appointment', backref='notifications')
    
    def __repr__(self):
        return f'<Notification {self.id} for User {self.user_id}>'

class AIConsultation(db.Model):
    __tablename__ = 'ai_consultations'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='ai_consultations')
    prediction = db.relationship('Prediction', backref='ai_consultations')
    
    def __repr__(self):
        return f'<AIConsultation {self.id} for User {self.user_id}>'
