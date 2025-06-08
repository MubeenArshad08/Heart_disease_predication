from flask import render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
from datetime import datetime, timedelta
import logging

from app import app, db
from models import User, HealthData, Prediction, Appointment, AIConsultation
from forms import LoginForm, RegisterForm, PredictionForm, AppointmentForm
from ml_model import HeartDiseasePredictor

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

# Initialize ML predictor
predictor = HeartDiseasePredictor()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    """Home page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember_me.data)
            next_page = request.args.get('next')
            flash('Logged in successfully!', 'success')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegisterForm()
    if form.validate_on_submit():
        # Check if user already exists
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists', 'danger')
            return render_template('register.html', form=form)
        
        if User.query.filter_by(email=form.email.data).first():
            flash('Email already registered', 'danger')
            return render_template('register.html', form=form)
        
        # Create new user
        user = User(
            username=form.username.data,
            email=form.email.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            phone=form.phone.data
        )
        user.set_password(form.password.data)
        
        try:
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logging.error(f"Registration error: {e}")
            flash('Registration failed. Please try again.', 'danger')
    
    return render_template('register.html', form=form)

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    # Get user's recent predictions
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                       .order_by(Prediction.created_at.desc())\
                                       .limit(5).all()
    
    # Get user's upcoming appointments
    upcoming_appointments = Appointment.query.filter_by(user_id=current_user.id)\
                                           .filter(Appointment.appointment_date >= datetime.utcnow())\
                                           .order_by(Appointment.appointment_date)\
                                           .limit(3).all()
    
    # Statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    high_risk_predictions = Prediction.query.filter_by(user_id=current_user.id, risk_level='High').count()
    
    return render_template('dashboard.html', 
                         recent_predictions=recent_predictions,
                         upcoming_appointments=upcoming_appointments,
                         total_predictions=total_predictions,
                         high_risk_predictions=high_risk_predictions)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Heart disease prediction"""
    form = PredictionForm()
    
    if form.validate_on_submit():
        try:
            # Save health data
            health_data = HealthData(
                user_id=current_user.id,
                age=form.age.data,
                sex=1 if form.sex.data == 'male' else 0,
                chest_pain_type=form.chest_pain_type.data,
                resting_bp=form.resting_bp.data,
                cholesterol=form.cholesterol.data,
                fasting_bs=1 if form.fasting_bs.data else 0,
                resting_ecg=form.resting_ecg.data,
                max_hr=form.max_hr.data,
                exercise_angina=1 if form.exercise_angina.data else 0,
                oldpeak=form.oldpeak.data,
                st_slope=form.st_slope.data
            )
            
            db.session.add(health_data)
            db.session.flush()  # Get the ID without committing
            
            # Make prediction
            prediction_result, confidence = predictor.predict(health_data)
            
            # Determine risk level
            if prediction_result == 1:
                if confidence >= 0.8:
                    risk_level = 'High'
                elif confidence >= 0.6:
                    risk_level = 'Medium'
                else:
                    risk_level = 'Low'
            else:
                risk_level = 'Low'
            
            # Save prediction
            prediction = Prediction(
                user_id=current_user.id,
                health_data_id=health_data.id,
                prediction_result=prediction_result,
                confidence_score=confidence,
                risk_level=risk_level
            )
            
            db.session.add(prediction)
            db.session.commit()
            
            flash('Prediction completed successfully!', 'success')
            return redirect(url_for('result', prediction_id=prediction.id))
            
        except Exception as e:
            db.session.rollback()
            logging.error(f"Prediction error: {e}")
            flash('Prediction failed. Please try again.', 'danger')
    
    return render_template('predict.html', form=form)

@app.route('/result/<int:prediction_id>')
@login_required
def result(prediction_id):
    """Show prediction results"""
    prediction = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first_or_404()
    return render_template('result.html', prediction=prediction)

@app.route('/book_appointment/<int:prediction_id>', methods=['GET', 'POST'])
@login_required
def book_appointment(prediction_id):
    """Book appointment with doctor"""
    prediction = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first_or_404()
    
    form = AppointmentForm()
    if form.validate_on_submit():
        try:
            appointment = Appointment(
                user_id=current_user.id,
                prediction_id=prediction_id,
                doctor_name=form.doctor_name.data,
                appointment_date=form.appointment_date.data,
                appointment_time=form.appointment_time.data,
                reason=form.reason.data
            )
            
            db.session.add(appointment)
            db.session.commit()
            
            flash('Appointment booked successfully!', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            db.session.rollback()
            logging.error(f"Appointment booking error: {e}")
            flash('Failed to book appointment. Please try again.', 'danger')
    
    return render_template('book_appointment.html', form=form, prediction=prediction)

@app.route('/ai_assistant/<int:prediction_id>', methods=['GET', 'POST'])
@login_required
def ai_assistant(prediction_id):
    """AI assistant consultation"""
    prediction = Prediction.query.filter_by(id=prediction_id, user_id=current_user.id).first_or_404()
    
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if question:
            # Generate AI response based on prediction
            ai_response = generate_ai_response(prediction, question)
            
            # Save consultation
            consultation = AIConsultation(
                user_id=current_user.id,
                prediction_id=prediction_id,
                question=question,
                ai_response=ai_response
            )
            
            db.session.add(consultation)
            db.session.commit()
            
            flash('AI consultation recorded successfully!', 'success')
    
    # Get previous consultations
    consultations = AIConsultation.query.filter_by(user_id=current_user.id, prediction_id=prediction_id)\
                                       .order_by(AIConsultation.created_at.desc()).all()
    
    return render_template('ai_assistant.html', prediction=prediction, consultations=consultations)

@app.route('/admin')
@login_required
def admin():
    """Admin dashboard"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Statistics
    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    high_risk_patients = Prediction.query.filter_by(risk_level='High').count()
    total_appointments = Appointment.query.count()
    
    # Recent activity
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    recent_registrations = User.query.order_by(User.created_at.desc()).limit(10).all()
    
    return render_template('admin.html',
                         total_users=total_users,
                         total_predictions=total_predictions,
                         high_risk_patients=high_risk_patients,
                         total_appointments=total_appointments,
                         recent_predictions=recent_predictions,
                         recent_registrations=recent_registrations)

def generate_ai_response(prediction, question):
    """Generate AI response based on prediction and question"""
    # Simple rule-based responses based on prediction results
    if prediction.prediction_result == 1:  # High risk
        if 'exercise' in question.lower():
            return "Based on your heart disease risk assessment, I recommend consulting with your doctor before starting any new exercise program. Light activities like walking may be beneficial, but medical supervision is important."
        elif 'diet' in question.lower():
            return "A heart-healthy diet is crucial. Consider reducing sodium, saturated fats, and processed foods. Increase fruits, vegetables, whole grains, and lean proteins. Please discuss specific dietary changes with your healthcare provider."
        elif 'medication' in question.lower():
            return "I cannot provide specific medication advice. Please consult with your doctor immediately about your heart disease risk and potential medication needs."
        elif 'symptoms' in question.lower():
            return "Watch for symptoms like chest pain, shortness of breath, fatigue, dizziness, or irregular heartbeat. Seek immediate medical attention if you experience any concerning symptoms."
        else:
            return "Given your elevated heart disease risk, I strongly recommend scheduling an appointment with a cardiologist for comprehensive evaluation and treatment planning. Early intervention can significantly improve outcomes."
    else:  # Low risk
        if 'exercise' in question.lower():
            return "Regular exercise is excellent for heart health! Aim for at least 150 minutes of moderate aerobic activity per week. Activities like brisk walking, swimming, or cycling are great choices."
        elif 'diet' in question.lower():
            return "Maintain a balanced diet rich in fruits, vegetables, whole grains, and lean proteins. Limit processed foods, excessive sodium, and saturated fats to keep your heart healthy."
        elif 'prevention' in question.lower():
            return "Continue your healthy lifestyle! Regular exercise, balanced diet, stress management, adequate sleep, and avoiding smoking are key to preventing heart disease."
        else:
            return "Your assessment shows lower heart disease risk, which is great! Continue maintaining a healthy lifestyle with regular exercise, balanced nutrition, and routine medical check-ups."

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500
