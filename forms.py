from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, IntegerField, FloatField, SelectField, TextAreaField, DateTimeField, DateField, TimeField
from wtforms.validators import DataRequired, Email, EqualTo, Length, NumberRange, ValidationError, InputRequired
from datetime import datetime, date, time

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    first_name = StringField('First Name', validators=[DataRequired(), Length(min=1, max=50)])
    last_name = StringField('Last Name', validators=[DataRequired(), Length(min=1, max=50)])
    phone = StringField('Phone Number', validators=[Length(max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])

class PredictionForm(FlaskForm):
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=1, max=120)])
    
    sex = SelectField('Sex', choices=[('male', 'Male'), ('female', 'Female')], validators=[DataRequired()])
    
    chest_pain_type = SelectField('Chest Pain Type', 
                                 choices=[
                                     (0, 'Typical Angina'),
                                     (1, 'Atypical Angina'),
                                     (2, 'Non-Anginal Pain'),
                                     (3, 'Asymptomatic')
                                 ], 
                                 coerce=int, validators=[DataRequired()])
    
    resting_bp = IntegerField('Resting Blood Pressure (mmHg)', 
                             validators=[DataRequired(), NumberRange(min=50, max=300)])
    
    cholesterol = IntegerField('Serum Cholesterol (mg/dl)', 
                              validators=[DataRequired(), NumberRange(min=50, max=1000)])
    
    fasting_bs = BooleanField('Fasting Blood Sugar > 120 mg/dl')
    
    resting_ecg = SelectField('Resting ECG Results',
                             choices=[
                                 (0, 'Normal'),
                                 (1, 'ST-T Wave Abnormality'),
                                 (2, 'Left Ventricular Hypertrophy')
                             ],
                             coerce=int, validators=[InputRequired()], default=0)
    
    max_hr = IntegerField('Maximum Heart Rate Achieved', 
                         validators=[DataRequired(), NumberRange(min=60, max=250)])
    
    exercise_angina = BooleanField('Exercise Induced Angina')
    
    oldpeak = FloatField('ST Depression (Exercise vs Rest)', 
                        validators=[InputRequired(), NumberRange(min=0, max=10)])
    
    st_slope = SelectField('ST Segment Slope',
                          choices=[
                              (0, 'Upsloping'),
                              (1, 'Flat'),
                              (2, 'Downsloping')
                          ],
                          coerce=int, validators=[InputRequired()], default=0)

    def validate_oldpeak(self, field):
        if field.data is None or field.data == '':
            raise ValidationError('This field is required.')
        try:
            value = float(field.data)
        except (TypeError, ValueError):
            raise ValidationError('Invalid number.')
        if not (0.0 <= value <= 10.0):
            raise ValidationError('ST depression value must be between 0.0 and 10.0.')

class AppointmentForm(FlaskForm):
    doctor_name = SelectField('Doctor', 
                             choices=[
                                 ('Dr. Sarah Johnson', 'Dr. Sarah Johnson - Cardiologist'),
                                 ('Dr. Michael Chen', 'Dr. Michael Chen - Internal Medicine'),
                                 ('Dr. Emily Rodriguez', 'Dr. Emily Rodriguez - Cardiothoracic Surgeon'),
                                 ('Dr. David Thompson', 'Dr. David Thompson - Preventive Medicine'),
                                 ('Dr. Lisa Wang', 'Dr. Lisa Wang - Cardiac Rehabilitation')
                             ], 
                             validators=[DataRequired()])
    
    appointment_date = DateField('Appointment Date', 
                                validators=[DataRequired()],
                                default=date.today())
    
    appointment_time = SelectField('Appointment Time',
                                  choices=[
                                      ('09:00', '9:00 AM'),
                                      ('09:30', '9:30 AM'),
                                      ('10:00', '10:00 AM'),
                                      ('10:30', '10:30 AM'),
                                      ('11:00', '11:00 AM'),
                                      ('11:30', '11:30 AM'),
                                      ('14:00', '2:00 PM'),
                                      ('14:30', '2:30 PM'),
                                      ('15:00', '3:00 PM'),
                                      ('15:30', '3:30 PM'),
                                      ('16:00', '4:00 PM'),
                                      ('16:30', '4:30 PM')
                                  ],
                                  validators=[DataRequired()])
    
    reason = TextAreaField('Reason for Appointment', 
                          validators=[DataRequired(), Length(min=10, max=500)])
    
    def validate_appointment_date(self, appointment_date):
        if appointment_date.data < date.today():
            raise ValidationError('Appointment date cannot be in the past.')
        if appointment_date.data > date.today().replace(year=date.today().year + 1):
            raise ValidationError('Appointment date cannot be more than a year in the future.')

class AIQuestionForm(FlaskForm):
    question = TextAreaField('Your Question', 
                           validators=[DataRequired(), Length(min=10, max=500)],
                           render_kw={"placeholder": "Ask me about your heart health, lifestyle recommendations, or any concerns..."})
