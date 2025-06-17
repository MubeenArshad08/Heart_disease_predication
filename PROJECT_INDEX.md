# HeartGuardian Project - Complete File Index

## 📁 Project Overview
HeartGuardian is a comprehensive heart disease prediction web application built with Flask, featuring ML-powered risk assessment, appointment management, and admin controls.

## 🏗️ Project Structure

```
HeartGuardian/
├── 📄 Core Application Files
├── 📁 templates/          # HTML Templates
├── 📁 static/            # CSS, JS, Assets
├── 📁 instance/          # Database & Instance Files
├── 📄 ML Models          # Machine Learning Models
├── 📄 Documentation      # Project Documentation
└── 📄 Configuration      # Project Configuration
```

---

## 📄 Core Application Files

### 1. **app.py** (1,021 bytes, 37 lines)
**Purpose**: Main Flask application initialization
**Key Components**:
- Flask app configuration
- Database initialization
- Secret key setup
- Database creation

**Key Functions**:
```python
# Main Flask app instance
app = Flask(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///heart_disease_app.db'

# Database initialization
db = SQLAlchemy(app)
```

### 2. **main.py** (127 bytes, 6 lines)
**Purpose**: Application entry point
**Key Components**:
- Flask app execution
- Debug mode configuration

### 3. **models.py** (6.5KB, 153 lines)
**Purpose**: Database models and relationships
**Key Models**:

#### User Model
```python
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    phone = db.Column(db.String(20))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

#### HealthData Model
```python
class HealthData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.Integer, nullable=False)  # 1 = male, 0 = female
    chest_pain_type = db.Column(db.Integer, nullable=False)  # 0-3
    resting_bp = db.Column(db.Integer, nullable=False)
    cholesterol = db.Column(db.Integer, nullable=False)
    fasting_bs = db.Column(db.Integer, nullable=False)
    resting_ecg = db.Column(db.Integer, nullable=False)
    max_hr = db.Column(db.Integer, nullable=False)
    exercise_angina = db.Column(db.Integer, nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    st_slope = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

#### Prediction Model
```python
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    health_data_id = db.Column(db.Integer, db.ForeignKey('health_data.id'), nullable=False)
    prediction_result = db.Column(db.Integer, nullable=False)  # 0 = no disease, 1 = disease
    confidence_score = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)  # Low, Medium, High
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

#### Appointment Model
```python
class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    doctor_name = db.Column(db.String(100), nullable=False)
    appointment_date = db.Column(db.DateTime, nullable=False)
    appointment_time = db.Column(db.String(10), nullable=False)
    reason = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, confirmed, rejected, completed, cancelled
    doctor_notes = db.Column(db.Text)
    confirmation_date = db.Column(db.DateTime)
    confirmed_by = db.Column(db.String(100))
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

#### Notification Model
```python
class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    appointment_id = db.Column(db.Integer, db.ForeignKey('appointments.id'), nullable=True)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(50), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

#### AIConsultation Model
```python
class AIConsultation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

### 4. **forms.py** (6.2KB, 121 lines)
**Purpose**: Flask-WTF form definitions
**Key Forms**:

#### LoginForm
```python
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')
```

#### RegisterForm
```python
class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    first_name = StringField('First Name', validators=[DataRequired(), Length(min=2, max=50)])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(min=2, max=50)])
    phone = StringField('Phone Number', validators=[Optional(), Length(min=10, max=15)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')
```

#### PredictionForm
```python
class PredictionForm(FlaskForm):
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=1, max=120)])
    sex = SelectField('Sex', choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    chest_pain_type = SelectField('Chest Pain Type', choices=[
        ('0', 'Typical Angina'),
        ('1', 'Atypical Angina'),
        ('2', 'Non-anginal Pain'),
        ('3', 'Asymptomatic')
    ], validators=[DataRequired()])
    resting_bp = IntegerField('Resting Blood Pressure (mm Hg)', validators=[DataRequired(), NumberRange(min=90, max=200)])
    cholesterol = IntegerField('Serum Cholesterol (mg/dl)', validators=[DataRequired(), NumberRange(min=100, max=600)])
    fasting_bs = BooleanField('Fasting Blood Sugar > 120 mg/dl')
    resting_ecg = SelectField('Resting ECG Results', choices=[
        ('0', 'Normal'),
        ('1', 'ST-T Wave Abnormality'),
        ('2', 'Left Ventricular Hypertrophy')
    ], validators=[DataRequired()])
    max_hr = IntegerField('Maximum Heart Rate Achieved', validators=[DataRequired(), NumberRange(min=60, max=202)])
    exercise_angina = BooleanField('Exercise Induced Angina')
    oldpeak = FloatField('ST Depression Induced by Exercise', validators=[DataRequired(), NumberRange(min=0.0, max=6.2)])
    st_slope = SelectField('Slope of Peak Exercise ST Segment', choices=[
        ('0', 'Upsloping'),
        ('1', 'Flat'),
        ('2', 'Downsloping')
    ], validators=[DataRequired()])
    submit = SubmitField('Predict Heart Disease Risk')
```

#### AppointmentForm
```python
class AppointmentForm(FlaskForm):
    doctor_name = SelectField('Doctor', choices=[
        ('Dr. Sarah Johnson', 'Dr. Sarah Johnson - Cardiologist'),
        ('Dr. Michael Chen', 'Dr. Michael Chen - Internal Medicine'),
        ('Dr. Emily Rodriguez', 'Dr. Emily Rodriguez - Cardiothoracic Surgeon'),
        ('Dr. David Thompson', 'Dr. David Thompson - Preventive Medicine'),
        ('Dr. Lisa Wang', 'Dr. Lisa Wang - Cardiac Rehabilitation')
    ], validators=[DataRequired()])
    appointment_date = DateField('Appointment Date', validators=[DataRequired()])
    appointment_time = SelectField('Appointment Time', choices=[
        ('09:00', '9:00 AM'), ('09:30', '9:30 AM'), ('10:00', '10:00 AM'),
        ('10:30', '10:30 AM'), ('11:00', '11:00 AM'), ('11:30', '11:30 AM'),
        ('14:00', '2:00 PM'), ('14:30', '2:30 PM'), ('15:00', '3:00 PM'),
        ('15:30', '3:30 PM'), ('16:00', '4:00 PM'), ('16:30', '4:30 PM')
    ], validators=[DataRequired()])
    reason = TextAreaField('Reason for Appointment', validators=[DataRequired(), Length(min=10, max=500)])
```

### 5. **routes.py** (31KB, 764 lines)
**Purpose**: All Flask routes and application logic
**Key Routes**:

#### Authentication Routes
- `GET/POST /login` - User login
- `GET/POST /register` - User registration
- `GET /logout` - User logout

#### Main Application Routes
- `GET /` - Homepage
- `GET /dashboard` - User dashboard
- `GET/POST /predict` - Heart disease prediction
- `GET /result/<id>` - Prediction results
- `GET/POST /book_appointment/<id>` - Book appointment
- `GET/POST /ai_assistant/<id>` - AI consultation

#### Admin Routes
- `GET /admin` - Admin dashboard
- `GET /admin/users` - User management
- `POST /admin/user/<id>/toggle-admin` - Toggle admin status
- `POST /admin/user/<id>/delete` - Delete user
- `GET /admin/predictions` - Predictions management
- `POST /admin/prediction/<id>/delete` - Delete prediction
- `GET /admin/appointments` - Appointments management
- `POST /admin/appointment/<id>/delete` - Delete appointment
- `GET /admin/system` - System settings
- `POST /admin/system/clear-notifications` - Clear notifications
- `POST /admin/system/export-data` - Export data

#### Doctor Routes
- `GET /doctor/appointments` - Doctor appointment management
- `POST /doctor/appointment/<id>/confirm` - Confirm appointment
- `POST /doctor/appointment/<id>/reject` - Reject appointment
- `POST /doctor/appointment/<id>/complete` - Complete appointment

#### User Routes
- `POST /appointment/<id>/cancel` - Cancel appointment
- `GET /notifications` - View notifications
- `POST /notifications/mark-read/<id>` - Mark notification as read

### 6. **ml_model.py** (9.7KB, 277 lines)
**Purpose**: Machine learning model implementation
**Key Components**:

#### HeartDiseasePredictor Class
```python
class HeartDiseasePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load('heart_disease_model.pkl')
            self.scaler = joblib.load('heart_disease_scaler.pkl')
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def predict(self, health_data):
        """Make prediction on health data"""
        # Feature extraction and preprocessing
        features = self.extract_features(health_data)
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        confidence = self.model.predict_proba(features_scaled)[0].max()
        
        return prediction, confidence
    
    def extract_features(self, health_data):
        """Extract features from health data"""
        return [
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
        ]
```

### 7. **migrate_db.py** (2.4KB, 56 lines)
**Purpose**: Database migration script
**Key Functions**:
```python
def migrate_database():
    """Migrate the database to add new fields"""
    with app.app_context():
        # Add new columns to appointments table
        # Update existing appointments status
        # Create notifications table
```

---

## 📁 Templates Directory

### 1. **base.html** (5.1KB, 127 lines)
**Purpose**: Base template with common layout
**Key Features**:
- Bootstrap 5 integration
- Font Awesome icons
- Navigation bar
- Flash messages
- Common CSS/JS includes

### 2. **index.html** (8.2KB, 190 lines)
**Purpose**: Landing page
**Key Features**:
- Hero section with call-to-action
- Feature highlights
- How it works section
- Testimonials
- Contact information

### 3. **login.html** (4.3KB, 104 lines)
**Purpose**: User login page
**Key Features**:
- Login form
- Remember me functionality
- Registration link
- Password reset (placeholder)

### 4. **register.html** (9.5KB, 205 lines)
**Purpose**: User registration page
**Key Features**:
- Registration form with validation
- Password confirmation
- Terms and conditions
- Login link

### 5. **dashboard.html** (20KB, 402 lines)
**Purpose**: User dashboard
**Key Features**:
- User statistics
- Recent predictions
- Upcoming appointments
- Quick actions
- Notification bell
- Recent appointments history

### 6. **predict.html** (17KB, 342 lines)
**Purpose**: Heart disease prediction form
**Key Features**:
- Comprehensive health data form
- Real-time validation
- Progress indicators
- Help tooltips
- Form validation

### 7. **result.html** (18KB, 355 lines)
**Purpose**: Prediction results display
**Key Features**:
- Risk level visualization
- Detailed analysis
- Recommendations
- Action buttons
- Progress bars

### 8. **book_appointment.html** (17KB, 351 lines)
**Purpose**: Appointment booking
**Key Features**:
- Doctor selection
- Date/time picker
- Reason for appointment
- Confirmation process info
- Assessment summary

### 9. **ai_assistant.html** (20KB, 464 lines)
**Purpose**: AI consultation interface
**Key Features**:
- Chat interface
- Question input
- AI responses
- Conversation history
- Health recommendations

### 10. **notifications.html** (4.8KB, 100 lines)
**Purpose**: User notifications
**Key Features**:
- Notification list
- Read/unread status
- Mark as read functionality
- Notification types

### 11. **doctor_appointments.html** (23KB, 432 lines)
**Purpose**: Doctor appointment management
**Key Features**:
- Pending appointments
- Confirmed appointments
- Action buttons (confirm/reject/complete)
- Patient information
- Appointment details

### 12. **admin.html** (15KB, 307 lines)
**Purpose**: Admin dashboard
**Key Features**:
- System statistics
- Navigation cards
- Risk distribution
- Recent activity
- Quick actions

### 13. **admin_users.html** (13KB, 229 lines)
**Purpose**: User management
**Key Features**:
- User list with pagination
- Admin role management
- User deletion
- Activity tracking
- Statistics

### 14. **admin_predictions.html** (13KB, 234 lines)
**Purpose**: Predictions management
**Key Features**:
- Prediction list with filtering
- Risk level filtering
- Prediction details
- Action buttons
- Pagination

### 15. **admin_appointments.html** (17KB, 287 lines)
**Purpose**: Appointments management
**Key Features**:
- Appointment list with filtering
- Status filtering
- Appointment actions
- Patient information
- Pagination

### 16. **admin_system.html** (14KB, 302 lines)
**Purpose**: System settings
**Key Features**:
- System health monitoring
- Database information
- Maintenance actions
- System statistics
- Configuration details

### 17. **404.html** (1.6KB, 43 lines)
**Purpose**: 404 error page
**Key Features**:
- Error message
- Navigation back to home
- Search functionality

### 18. **500.html** (1.8KB, 46 lines)
**Purpose**: 500 error page
**Key Features**:
- Error message
- Contact information
- Navigation back to home

---

## 📁 Static Directory

### 1. **static/css/style.css** (8.3KB, 450 lines)
**Purpose**: Custom CSS styles
**Key Features**:
- Custom color scheme
- Component styling
- Responsive design
- Animation effects
- Form styling

### 2. **static/js/main.js** (19KB, 589 lines)
**Purpose**: JavaScript functionality
**Key Features**:
- Form validation
- Dynamic content loading
- Chart generation
- Interactive elements
- AJAX functionality

---

## 📁 Instance Directory

### 1. **instance/heart_disease_app.db** (36KB, 98 lines)
**Purpose**: SQLite database
**Key Tables**:
- users
- health_data
- predictions
- appointments
- notifications
- ai_consultations

---

## 📄 ML Models

### 1. **heart_disease_model.pkl** (545KB, 795 lines)
**Purpose**: Trained machine learning model
**Type**: Pickled scikit-learn model
**Algorithm**: Likely Random Forest or similar ensemble method

### 2. **heart_disease_scaler.pkl** (863B, 6 lines)
**Purpose**: Feature scaler for model input
**Type**: Pickled StandardScaler
**Function**: Normalizes input features

---

## 📄 Configuration Files

### 1. **pyproject.toml** (560B, 25 lines)
**Purpose**: Python project configuration
**Key Dependencies**:
- Flask
- SQLAlchemy
- scikit-learn
- joblib
- Flask-WTF
- Flask-Login

### 2. **uv.lock** (109KB, 786 lines)
**Purpose**: Dependency lock file
**Function**: Ensures reproducible builds

### 3. **.replit** (655B, 35 lines)
**Purpose**: Replit configuration
**Function**: Development environment setup

---

## 📄 Documentation

### 1. **DOCTOR_CONFIRMATION_SYSTEM.md** (7.4KB, 207 lines)
**Purpose**: Documentation for appointment confirmation system
**Key Topics**:
- System overview
- How it works
- Features
- Database schema
- Routes and endpoints
- User interface
- Setup instructions

---

## 🔧 Key Features Summary

### **Core Functionality**:
- ✅ Heart disease risk prediction using ML
- ✅ User authentication and registration
- ✅ Appointment booking and management
- ✅ Doctor confirmation system
- ✅ AI consultation assistant
- ✅ Notification system
- ✅ Comprehensive admin panel

### **Technical Stack**:
- **Backend**: Flask, SQLAlchemy, SQLite
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **ML**: scikit-learn, joblib
- **Authentication**: Flask-Login
- **Forms**: Flask-WTF
- **Icons**: Font Awesome

### **Database Schema**:
- **6 Main Tables**: users, health_data, predictions, appointments, notifications, ai_consultations
- **Relationships**: Foreign keys with cascade deletes
- **Indexing**: Primary keys and unique constraints

### **Security Features**:
- Password hashing with Werkzeug
- CSRF protection
- Session management
- Role-based access control
- Admin-only functions

### **User Experience**:
- Responsive design
- Real-time validation
- Interactive charts
- Progress indicators
- Notification system
- Auto-refresh functionality

This comprehensive index covers all files, their purposes, key components, and functionality in the HeartGuardian project. 