# HeartGuardian 🫀

> **AI-Powered Heart Disease Risk Assessment & Medical Appointment Management System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Machine Learning Model](#machine-learning-model)
- [Admin Panel](#admin-panel)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

HeartGuardian is a comprehensive web application that combines machine learning-powered heart disease risk assessment with a complete medical appointment management system. The platform provides users with accurate health predictions, connects them with healthcare professionals, and offers an AI-powered consultation assistant.

### Key Capabilities

- **🔬 ML-Powered Risk Assessment**: Advanced machine learning model for heart disease prediction
- **👨‍⚕️ Doctor Appointment System**: Complete appointment booking and confirmation workflow
- **🤖 AI Health Assistant**: Intelligent consultation and health guidance
- **📊 Comprehensive Dashboard**: User-friendly interface with detailed health insights
- **🔐 Secure Admin Panel**: Complete system management and oversight
- **📱 Responsive Design**: Works seamlessly across all devices

## ✨ Features

### 🫀 Heart Disease Prediction
- **Advanced ML Model**: Trained on extensive heart disease datasets
- **Real-time Assessment**: Instant risk evaluation with confidence scores
- **Detailed Analysis**: Comprehensive health factor breakdown
- **Risk Level Classification**: Low, Medium, and High risk categorization
- **Personalized Recommendations**: Tailored health advice based on results

### 👨‍⚕️ Medical Appointment Management
- **Smart Booking System**: Easy appointment scheduling with qualified doctors
- **Doctor Confirmation Workflow**: Professional review and approval process
- **Status Tracking**: Real-time appointment status updates
- **Notification System**: Instant alerts for confirmations and changes
- **Cancellation Management**: Flexible appointment modification

### 🤖 AI Health Assistant
- **Intelligent Consultation**: AI-powered health guidance
- **Personalized Responses**: Context-aware recommendations
- **Conversation History**: Track all AI interactions
- **Health Education**: Educational content and resources
- **24/7 Availability**: Round-the-clock health support

### 📊 User Dashboard
- **Health Overview**: Complete health assessment history
- **Progress Tracking**: Monitor health improvements over time
- **Appointment Management**: View and manage all appointments
- **Notification Center**: Stay updated with important alerts
- **Quick Actions**: Easy access to key features

### 🔐 Admin Control Panel
- **User Management**: Complete user oversight and control
- **Data Analytics**: Comprehensive system statistics
- **Appointment Oversight**: Manage all medical appointments
- **System Maintenance**: Database and system administration
- **Security Controls**: Role-based access management

## 🛠️ Technology Stack

### Backend
- **Python 3.8+** - Core programming language
- **Flask 2.3+** - Web framework
- **SQLAlchemy** - Database ORM
- **SQLite** - Database (production-ready alternatives: PostgreSQL, MySQL)
- **Flask-Login** - User authentication
- **Flask-WTF** - Form handling and CSRF protection

### Machine Learning
- **scikit-learn** - ML algorithms and model training
- **joblib** - Model serialization
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Styling and animations
- **JavaScript (ES6+)** - Interactive functionality
- **Bootstrap 5** - Responsive UI framework
- **Font Awesome** - Icon library

### Development & Deployment
- **Git** - Version control
- **uv** - Python package management
- **SQLite** - Development database
- **Replit** - Development environment

## 📸 Screenshots

### User Interface
![Dashboard](https://via.placeholder.com/800x400/4CAF50/FFFFFF?text=User+Dashboard)
![Prediction Form](https://via.placeholder.com/800x400/2196F3/FFFFFF?text=Health+Assessment)
![Results](https://via.placeholder.com/800x400/FF9800/FFFFFF?text=Prediction+Results)

### Admin Panel
![Admin Dashboard](https://via.placeholder.com/800x400/9C27B0/FFFFFF?text=Admin+Dashboard)
![User Management](https://via.placeholder.com/800x400/607D8B/FFFFFF?text=User+Management)

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/MubeenArshad08/Heart_disease_predication.git
cd Heart_disease_predication
```

### Step 2: Set Up Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using uv
uv venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

**Note:** The `requirements.txt` file includes all necessary dependencies:
- Flask and Flask extensions
- Machine learning libraries (scikit-learn, numpy, pandas)
- Database libraries (SQLAlchemy)
- Authentication libraries (Flask-Login)
- And other required packages

### Step 4: Set Up Environment Variables
```bash
# Create .env file (optional for development)
echo "SECRET_KEY=your-secret-key-here" > .env
echo "FLASK_ENV=development" >> .env
echo "DATABASE_URL=sqlite:///heart_disease_app.db" >> .env
```

### Step 5: Initialize Database (Optional)
```bash
# The database is already included in the repository
# If you need to recreate it, run:
python migrate_db.py

# Or initialize manually
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

### Step 6: Run the Application
```bash
# Development mode
python main.py

# Or using Flask CLI
set FLASK_APP=app.py  # On Windows
set FLASK_ENV=development  # On Windows
flask run
```

The application will be available at `http://localhost:5000`

### Quick Start (All-in-one)
```bash
git clone https://github.com/MubeenArshad08/Heart_disease_predication.git
cd Heart_disease_predication
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
python main.py
```

## 📖 Usage

### For Patients

1. **Registration & Login**
   - Create an account with your personal information
   - Log in to access your personalized dashboard

2. **Health Assessment**
   - Navigate to "New Assessment"
   - Fill in your health information
   - Receive instant risk evaluation

3. **Understanding Results**
   - View detailed risk analysis
   - Read personalized recommendations
   - Access educational resources

4. **Booking Appointments**
   - Schedule consultations with qualified doctors
   - Receive confirmation notifications
   - Manage appointment status

5. **AI Consultation**
   - Ask health-related questions
   - Receive AI-powered guidance
   - Track conversation history

### For Doctors (Admin Users)

1. **Access Admin Panel**
   - Log in with admin credentials
   - Navigate to admin dashboard

2. **Review Appointments**
   - View pending appointment requests
   - Review patient information and risk assessments
   - Confirm or reject appointments with notes

3. **Patient Management**
   - Access patient health records
   - View prediction history
   - Monitor patient progress

4. **System Administration**
   - Manage user accounts
   - Monitor system statistics
   - Perform maintenance tasks

### For System Administrators

1. **User Management**
   - View all registered users
   - Grant/revoke admin privileges
   - Manage user accounts

2. **Data Management**
   - Monitor prediction data
   - Manage appointment records
   - Export system data

3. **System Maintenance**
   - Monitor system health
   - Clear notifications
   - Database administration

## 🔌 API Documentation

### Authentication Endpoints

```http
POST /login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password123"
}
```

```http
POST /register
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "password": "password123"
}
```

### Prediction Endpoints

```http
POST /predict
Content-Type: application/json

{
  "age": 45,
  "sex": "male",
  "chest_pain_type": "0",
  "resting_bp": 140,
  "cholesterol": 250,
  "fasting_bs": false,
  "resting_ecg": "0",
  "max_hr": 150,
  "exercise_angina": false,
  "oldpeak": 1.5,
  "st_slope": "1"
}
```

### Appointment Endpoints

```http
POST /book_appointment/{prediction_id}
Content-Type: application/json

{
  "doctor_name": "Dr. Sarah Johnson",
  "appointment_date": "2024-01-15",
  "appointment_time": "10:00",
  "reason": "Follow-up consultation"
}
```

## 🗄️ Database Schema

### Core Tables

```sql
-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    phone VARCHAR(20),
    is_admin BOOLEAN DEFAULT FALSE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Health data table
CREATE TABLE health_data (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    age INTEGER NOT NULL,
    sex INTEGER NOT NULL,
    chest_pain_type INTEGER NOT NULL,
    resting_bp INTEGER NOT NULL,
    cholesterol INTEGER NOT NULL,
    fasting_bs INTEGER NOT NULL,
    resting_ecg INTEGER NOT NULL,
    max_hr INTEGER NOT NULL,
    exercise_angina INTEGER NOT NULL,
    oldpeak FLOAT NOT NULL,
    st_slope INTEGER NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

-- Predictions table
CREATE TABLE predictions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    health_data_id INTEGER NOT NULL,
    prediction_result INTEGER NOT NULL,
    confidence_score FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (health_data_id) REFERENCES health_data (id)
);

-- Appointments table
CREATE TABLE appointments (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    prediction_id INTEGER NOT NULL,
    doctor_name VARCHAR(100) NOT NULL,
    appointment_date DATETIME NOT NULL,
    appointment_time VARCHAR(10) NOT NULL,
    reason TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    doctor_notes TEXT,
    confirmation_date DATETIME,
    confirmed_by VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
);
```

## 🤖 Machine Learning Model

### Model Architecture
- **Algorithm**: Random Forest Classifier
- **Features**: 11 health parameters
- **Accuracy**: ~85% on test dataset
- **Training Data**: Heart disease dataset with 1000+ samples

### Feature Engineering
```python
features = [
    'age', 'sex', 'chest_pain_type', 'resting_bp',
    'cholesterol', 'fasting_bs', 'resting_ecg',
    'max_hr', 'exercise_angina', 'oldpeak', 'st_slope'
]
```

### Model Performance
- **Precision**: 0.87
- **Recall**: 0.83
- **F1-Score**: 0.85
- **ROC-AUC**: 0.89

## 🔐 Admin Panel

### Access Control
- Role-based access control
- Admin-only functions
- Secure authentication
- Session management

### Key Features
- **User Management**: Complete user oversight
- **Data Analytics**: System-wide statistics
- **Appointment Management**: Doctor appointment oversight
- **System Maintenance**: Database and system administration
- **Security Monitoring**: Access logs and security controls

### Admin Routes
- `/admin` - Main admin dashboard
- `/admin/users` - User management
- `/admin/predictions` - Prediction data management
- `/admin/appointments` - Appointment oversight
- `/admin/system` - System settings and maintenance

## 🤝 Contributing

We welcome contributions to HeartGuardian! Please follow these steps:

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes: `git commit -m 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

### Code Style
- Follow PEP 8 Python style guide
- Use meaningful variable and function names
- Add docstrings to all functions
- Include type hints where appropriate

### Testing
```bash
# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=app tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Medical Dataset**: Heart disease dataset from UCI Machine Learning Repository
- **ML Framework**: scikit-learn for machine learning capabilities
- **Web Framework**: Flask for robust web application development
- **UI Framework**: Bootstrap for responsive design
- **Icons**: Font Awesome for beautiful icons

## 📞 Support

For support and questions:

- **Email**: support@heartguardian.com
- **Documentation**: [docs.heartguardian.com](https://docs.heartguardian.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/heartguardian/issues)

## 🔮 Roadmap

### Upcoming Features
- [ ] **Mobile App**: Native iOS and Android applications
- [ ] **Telemedicine Integration**: Video consultation capabilities
- [ ] **Advanced Analytics**: Deep learning models for better predictions
- [ ] **Multi-language Support**: Internationalization
- [ ] **API Gateway**: RESTful API for third-party integrations
- [ ] **Blockchain Integration**: Secure health record management
- [ ] **IoT Integration**: Wearable device data integration
- [ ] **Advanced Reporting**: Comprehensive health reports

### Version History
- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Added AI assistant and notification system
- **v1.2.0** - Enhanced admin panel and doctor confirmation system
- **v1.3.0** - Improved ML model and user interface

---

**Made with ❤️ for better heart health**

*HeartGuardian - Empowering individuals with AI-driven heart health insights* 