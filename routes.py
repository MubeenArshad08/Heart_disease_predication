from flask import render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import check_password_hash
from datetime import datetime, timedelta
import logging
import os

from app import app, db
from models import User, HealthData, Prediction, Appointment, AIConsultation, Notification
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
    # Get recent predictions
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id)\
                                       .order_by(Prediction.created_at.desc()).limit(5).all()
    
    # Get upcoming appointments (confirmed and pending)
    upcoming_appointments = Appointment.query.filter(
        Appointment.user_id == current_user.id,
        Appointment.status.in_(['pending', 'confirmed']),
        Appointment.appointment_date >= datetime.now().date()
    ).order_by(Appointment.appointment_date, Appointment.appointment_time).all()
    
    # Get recent appointments (all statuses)
    recent_appointments = Appointment.query.filter_by(user_id=current_user.id)\
                                         .order_by(Appointment.created_at.desc()).limit(10).all()
    
    # Get unread notifications
    unread_notifications = Notification.query.filter_by(user_id=current_user.id, is_read=False)\
                                            .order_by(Notification.created_at.desc()).limit(5).all()
    
    # Statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    high_risk_predictions = Prediction.query.filter_by(user_id=current_user.id, risk_level='High').count()
    
    return render_template('dashboard.html', 
                         recent_predictions=recent_predictions,
                         upcoming_appointments=upcoming_appointments,
                         recent_appointments=recent_appointments,
                         unread_notifications=unread_notifications,
                         total_predictions=total_predictions,
                         high_risk_predictions=high_risk_predictions)

@app.route('/notifications')
@login_required
def notifications():
    """User notifications page"""
    # Get all notifications for user
    notifications = Notification.query.filter_by(user_id=current_user.id)\
                                    .order_by(Notification.created_at.desc()).all()
    
    return render_template('notifications.html', notifications=notifications)

@app.route('/notifications/mark-read/<int:notification_id>', methods=['POST'])
@login_required
def mark_notification_read(notification_id):
    """Mark a notification as read"""
    notification = Notification.query.filter_by(id=notification_id, user_id=current_user.id).first_or_404()
    
    try:
        notification.is_read = True
        db.session.commit()
        flash('Notification marked as read.', 'success')
    except Exception as e:
        db.session.rollback()
        flash('Failed to mark notification as read.', 'danger')
    
    return redirect(url_for('notifications'))

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
                reason=form.reason.data,
                status='pending'  # Start with pending status
            )
            
            db.session.add(appointment)
            db.session.commit()
            
            flash('Appointment request submitted successfully! The doctor will review and confirm your appointment.', 'success')
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            db.session.rollback()
            logging.error(f"Appointment booking error: {e}")
            flash('Failed to book appointment. Please try again.', 'danger')
    
    return render_template('book_appointment.html', form=form, prediction=prediction)

@app.route('/doctor/appointments')
@login_required
def doctor_appointments():
    """Doctor's appointment management dashboard"""
    if not current_user.is_admin:
        flash('Access denied. Doctor privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all pending appointments
    pending_appointments = Appointment.query.filter_by(status='pending').order_by(Appointment.created_at.desc()).all()
    
    # Get confirmed appointments for today and upcoming
    today = datetime.now().date()
    confirmed_appointments = Appointment.query.filter(
        Appointment.status == 'confirmed',
        Appointment.appointment_date >= today
    ).order_by(Appointment.appointment_date, Appointment.appointment_time).all()
    
    # Get recent completed appointments
    completed_appointments = Appointment.query.filter_by(status='completed').order_by(Appointment.confirmation_date.desc()).limit(10).all()
    
    return render_template('doctor_appointments.html',
                         pending_appointments=pending_appointments,
                         confirmed_appointments=confirmed_appointments,
                         completed_appointments=completed_appointments)

@app.route('/doctor/appointment/<int:appointment_id>/confirm', methods=['POST'])
@login_required
def confirm_appointment(appointment_id):
    """Doctor confirms an appointment"""
    if not current_user.is_admin:
        flash('Access denied. Doctor privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    appointment = Appointment.query.get_or_404(appointment_id)
    
    if appointment.status != 'pending':
        flash('This appointment has already been processed.', 'warning')
        return redirect(url_for('doctor_appointments'))
    
    try:
        appointment.status = 'confirmed'
        appointment.confirmation_date = datetime.utcnow()
        appointment.confirmed_by = current_user.first_name + ' ' + current_user.last_name
        
        db.session.commit()
        
        # Send confirmation email (placeholder for now)
        send_appointment_confirmation_email(appointment)
        
        # Create notification for user
        notification = Notification(
            user_id=appointment.user_id,
            appointment_id=appointment.id,
            title="Appointment Confirmed",
            message=f"Your appointment with {appointment.doctor_name} on {appointment.appointment_date.strftime('%B %d, %Y')} at {appointment.appointment_time} has been confirmed.",
            type="appointment_confirmed"
        )
        db.session.add(notification)
        
        flash(f'Appointment with {appointment.user.first_name} {appointment.user.last_name} has been confirmed.', 'success')
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Appointment confirmation error: {e}")
        flash('Failed to confirm appointment. Please try again.', 'danger')
    
    return redirect(url_for('doctor_appointments'))

@app.route('/doctor/appointment/<int:appointment_id>/reject', methods=['POST'])
@login_required
def reject_appointment(appointment_id):
    """Doctor rejects an appointment"""
    if not current_user.is_admin:
        flash('Access denied. Doctor privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    appointment = Appointment.query.get_or_404(appointment_id)
    
    if appointment.status != 'pending':
        flash('This appointment has already been processed.', 'warning')
        return redirect(url_for('doctor_appointments'))
    
    rejection_reason = request.form.get('rejection_reason', '').strip()
    
    if not rejection_reason:
        flash('Please provide a reason for rejection.', 'warning')
        return redirect(url_for('doctor_appointments'))
    
    try:
        appointment.status = 'rejected'
        appointment.doctor_notes = rejection_reason
        appointment.confirmation_date = datetime.utcnow()
        appointment.confirmed_by = current_user.first_name + ' ' + current_user.last_name
        
        db.session.commit()
        
        # Send rejection email (placeholder for now)
        send_appointment_rejection_email(appointment)
        
        # Create notification for user
        notification = Notification(
            user_id=appointment.user_id,
            appointment_id=appointment.id,
            title="Appointment Rejected",
            message=f"Your appointment with {appointment.doctor_name} has been rejected. Reason: {rejection_reason}",
            type="appointment_rejected"
        )
        db.session.add(notification)
        
        flash(f'Appointment with {appointment.user.first_name} {appointment.user.last_name} has been rejected.', 'success')
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Appointment rejection error: {e}")
        flash('Failed to reject appointment. Please try again.', 'danger')
    
    return redirect(url_for('doctor_appointments'))

@app.route('/doctor/appointment/<int:appointment_id>/complete', methods=['POST'])
@login_required
def complete_appointment(appointment_id):
    """Doctor marks appointment as completed"""
    if not current_user.is_admin:
        flash('Access denied. Doctor privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    appointment = Appointment.query.get_or_404(appointment_id)
    
    if appointment.status != 'confirmed':
        flash('Only confirmed appointments can be marked as completed.', 'warning')
        return redirect(url_for('doctor_appointments'))
    
    try:
        appointment.status = 'completed'
        appointment.confirmation_date = datetime.utcnow()
        
        db.session.commit()
        
        flash(f'Appointment with {appointment.user.first_name} {appointment.user.last_name} has been marked as completed.', 'success')
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Appointment completion error: {e}")
        flash('Failed to complete appointment. Please try again.', 'danger')
    
    return redirect(url_for('doctor_appointments'))

@app.route('/appointment/<int:appointment_id>/cancel', methods=['POST'])
@login_required
def cancel_appointment(appointment_id):
    """User cancels their own appointment"""
    appointment = Appointment.query.filter_by(id=appointment_id, user_id=current_user.id).first_or_404()
    
    if appointment.status not in ['pending', 'confirmed']:
        flash('This appointment cannot be cancelled.', 'warning')
        return redirect(url_for('dashboard'))
    
    try:
        appointment.status = 'cancelled'
        appointment.confirmation_date = datetime.utcnow()
        
        db.session.commit()
        
        flash('Your appointment has been cancelled successfully.', 'success')
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Appointment cancellation error: {e}")
        flash('Failed to cancel appointment. Please try again.', 'danger')
    
    return redirect(url_for('dashboard'))

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
    print("DEBUG: Admin route accessed!")
    """Admin dashboard"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Statistics
    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    high_risk_patients = Prediction.query.filter_by(risk_level='High').count()
    medium_risk_patients = Prediction.query.filter_by(risk_level='Medium').count()
    low_risk_patients = Prediction.query.filter_by(risk_level='Low').count()
    total_appointments = Appointment.query.count()
    pending_appointments = Appointment.query.filter_by(status='pending').count()
    confirmed_appointments = Appointment.query.filter_by(status='confirmed').count()
    
    # Recent activity
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    recent_registrations = User.query.order_by(User.created_at.desc()).limit(10).all()
    recent_appointments = Appointment.query.order_by(Appointment.created_at.desc()).limit(10).all()
    
    return render_template('admin.html',
                         total_users=total_users,
                         total_predictions=total_predictions,
                         high_risk_patients=high_risk_patients,
                         medium_risk_patients=medium_risk_patients,
                         low_risk_patients=low_risk_patients,
                         total_appointments=total_appointments,
                         pending_appointments=pending_appointments,
                         confirmed_appointments=confirmed_appointments,
                         recent_predictions=recent_predictions,
                         recent_registrations=recent_registrations,
                         recent_appointments=recent_appointments)

@app.route('/admin/users')
@login_required
def admin_users():
    """Admin user management"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all users with pagination
    page = request.args.get('page', 1, type=int)
    users = User.query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    seven_days_ago = datetime.utcnow() - timedelta(days=7)
    return render_template('admin_users.html', users=users, seven_days_ago=seven_days_ago)

@app.route('/admin/user/<int:user_id>/toggle-admin', methods=['POST'])
@login_required
def toggle_admin_status(user_id):
    """Toggle admin status for a user"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    if user_id == current_user.id:
        flash('You cannot modify your own admin status.', 'warning')
        return redirect(url_for('admin_users'))
    
    user = User.query.get_or_404(user_id)
    
    try:
        user.is_admin = not user.is_admin
        db.session.commit()
        
        status = "granted" if user.is_admin else "revoked"
        flash(f'Admin privileges {status} for {user.first_name} {user.last_name}.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash('Failed to update admin status.', 'danger')
    
    return redirect(url_for('admin_users'))

@app.route('/admin/user/<int:user_id>/delete', methods=['POST'])
@login_required
def delete_user(user_id):
    """Delete a user"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    if user_id == current_user.id:
        flash('You cannot delete your own account.', 'warning')
        return redirect(url_for('admin_users'))
    
    user = User.query.get_or_404(user_id)
    
    try:
        # Delete user and all associated data (cascade)
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.first_name} {user.last_name} has been deleted.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash('Failed to delete user.', 'danger')
    
    return redirect(url_for('admin_users'))

@app.route('/admin/predictions')
@login_required
def admin_predictions():
    """Admin predictions management"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all predictions with pagination
    page = request.args.get('page', 1, type=int)
    risk_filter = request.args.get('risk', 'all')
    
    query = Prediction.query
    
    if risk_filter != 'all':
        query = query.filter_by(risk_level=risk_filter.title())
    
    predictions = query.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('admin_predictions.html', predictions=predictions, risk_filter=risk_filter)

@app.route('/admin/prediction/<int:prediction_id>/delete', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    """Delete a prediction"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    prediction = Prediction.query.get_or_404(prediction_id)
    
    try:
        db.session.delete(prediction)
        db.session.commit()
        flash('Prediction has been deleted.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash('Failed to delete prediction.', 'danger')
    
    return redirect(url_for('admin_predictions'))

@app.route('/admin/appointments')
@login_required
def admin_appointments():
    """Admin appointments management"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all appointments with pagination
    page = request.args.get('page', 1, type=int)
    status_filter = request.args.get('status', 'all')
    
    query = Appointment.query
    
    if status_filter != 'all':
        query = query.filter_by(status=status_filter)
    
    appointments = query.order_by(Appointment.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('admin_appointments.html', appointments=appointments, status_filter=status_filter)

@app.route('/admin/appointment/<int:appointment_id>/delete', methods=['POST'])
@login_required
def delete_appointment(appointment_id):
    """Delete an appointment"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    appointment = Appointment.query.get_or_404(appointment_id)
    
    try:
        db.session.delete(appointment)
        db.session.commit()
        flash('Appointment has been deleted.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash('Failed to delete appointment.', 'danger')
    
    return redirect(url_for('admin_appointments'))

@app.route('/admin/system')
@login_required
def admin_system():
    """Admin system settings and maintenance"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get system statistics
    total_users = User.query.count()
    total_predictions = Prediction.query.count()
    total_appointments = Appointment.query.count()
    total_notifications = Notification.query.count()
    
    # Get database info
    from sqlalchemy import inspect
    inspector = inspect(db.engine)
    tables = inspector.get_table_names()
    
    return render_template('admin_system.html',
                         total_users=total_users,
                         total_predictions=total_predictions,
                         total_appointments=total_appointments,
                         total_notifications=total_notifications,
                         tables=tables)

@app.route('/admin/system/clear-notifications', methods=['POST'])
@login_required
def clear_notifications():
    """Clear all notifications"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        Notification.query.delete()
        db.session.commit()
        flash('All notifications have been cleared.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash('Failed to clear notifications.', 'danger')
    
    return redirect(url_for('admin_system'))

@app.route('/admin/system/export-data', methods=['POST'])
@login_required
def export_data():
    """Export system data"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # This would implement data export functionality
        flash('Data export functionality would be implemented here.', 'info')
        
    except Exception as e:
        flash('Failed to export data.', 'danger')
    
    return redirect(url_for('admin_system'))

@app.route('/admin/system/backup-database', methods=['POST'])
@login_required
def backup_database():
    """Create database backup"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        import shutil
        from datetime import datetime
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'heart_disease_app_backup_{timestamp}.db'
        backup_path = os.path.join('backups', backup_filename)
        
        # Create backups directory if it doesn't exist
        os.makedirs('backups', exist_ok=True)
        
        # Copy database file
        shutil.copy2('instance/heart_disease_app.db', backup_path)
        
        flash(f'Database backup created: {backup_filename}', 'success')
        
    except Exception as e:
        flash(f'Failed to create backup: {str(e)}', 'danger')
    
    return redirect(url_for('admin_system'))

@app.route('/admin/users/search')
@login_required
def admin_users_search():
    """Search users in admin panel"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    query = request.args.get('q', '').strip()
    page = request.args.get('page', 1, type=int)
    
    if query:
        # Search by username, email, or name
        users = User.query.filter(
            db.or_(
                User.username.ilike(f'%{query}%'),
                User.email.ilike(f'%{query}%'),
                User.first_name.ilike(f'%{query}%'),
                User.last_name.ilike(f'%{query}%')
            )
        ).order_by(User.created_at.desc()).paginate(
            page=page, per_page=20, error_out=False)
    else:
        users = User.query.order_by(User.created_at.desc()).paginate(
            page=page, per_page=20, error_out=False)
    
    return render_template('admin_users.html', users=users, search_query=query)

@app.route('/admin/predictions/search')
@login_required
def admin_predictions_search():
    """Search predictions in admin panel"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    query = request.args.get('q', '').strip()
    risk_filter = request.args.get('risk', 'all')
    page = request.args.get('page', 1, type=int)
    
    # Build query
    if query:
        # Search by user information
        predictions = Prediction.query.join(User).filter(
            db.or_(
                User.username.ilike(f'%{query}%'),
                User.email.ilike(f'%{query}%'),
                User.first_name.ilike(f'%{query}%'),
                User.last_name.ilike(f'%{query}%')
            )
        )
    else:
        predictions = Prediction.query
    
    # Apply risk filter
    if risk_filter != 'all':
        predictions = predictions.filter_by(risk_level=risk_filter.title())
    
    predictions = predictions.order_by(Prediction.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('admin_predictions.html', 
                         predictions=predictions, 
                         risk_filter=risk_filter, 
                         search_query=query)

@app.route('/admin/appointments/search')
@login_required
def admin_appointments_search():
    """Search appointments in admin panel"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    query = request.args.get('q', '').strip()
    status_filter = request.args.get('status', 'all')
    page = request.args.get('page', 1, type=int)
    
    # Build query
    if query:
        # Search by user information or doctor name
        appointments = Appointment.query.join(User).filter(
            db.or_(
                User.username.ilike(f'%{query}%'),
                User.email.ilike(f'%{query}%'),
                User.first_name.ilike(f'%{query}%'),
                User.last_name.ilike(f'%{query}%'),
                Appointment.doctor_name.ilike(f'%{query}%')
            )
        )
    else:
        appointments = Appointment.query
    
    # Apply status filter
    if status_filter != 'all':
        appointments = appointments.filter_by(status=status_filter)
    
    appointments = appointments.order_by(Appointment.created_at.desc()).paginate(
        page=page, per_page=20, error_out=False)
    
    return render_template('admin_appointments.html', 
                         appointments=appointments, 
                         status_filter=status_filter, 
                         search_query=query)

@app.route('/admin/system/logs')
@login_required
def admin_logs():
    """View system logs"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    # This would show system logs
    # For now, we'll show a placeholder
    logs = [
        {'timestamp': '2024-01-15 10:30:00', 'level': 'INFO', 'message': 'System started successfully'},
        {'timestamp': '2024-01-15 10:35:00', 'level': 'INFO', 'message': 'New user registered: john_doe'},
        {'timestamp': '2024-01-15 10:40:00', 'level': 'WARNING', 'message': 'High risk prediction detected for user: jane_smith'},
    ]
    
    return render_template('admin_logs.html', logs=logs)

@app.route('/admin/system/clear-logs', methods=['POST'])
@login_required
def clear_logs():
    """Clear system logs"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('dashboard'))
    
    try:
        # This would clear system logs
        flash('System logs cleared successfully.', 'success')
    except Exception as e:
        flash(f'Failed to clear logs: {str(e)}', 'danger')
    
    return redirect(url_for('admin_logs'))

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

def send_appointment_confirmation_email(appointment):
    """Send confirmation email to patient (placeholder)"""
    # In a real implementation, this would send an email
    # For now, we'll just log the action
    logging.info(f"Appointment confirmation email would be sent to {appointment.user.email}")
    logging.info(f"Appointment details: {appointment.doctor_name} on {appointment.appointment_date} at {appointment.appointment_time}")

def send_appointment_rejection_email(appointment):
    """Send rejection email to patient (placeholder)"""
    # In a real implementation, this would send an email
    # For now, we'll just log the action
    logging.info(f"Appointment rejection email would be sent to {appointment.user.email}")
    logging.info(f"Rejection reason: {appointment.doctor_notes}")

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500
