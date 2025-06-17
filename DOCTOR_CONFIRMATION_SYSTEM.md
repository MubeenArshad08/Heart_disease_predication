# Doctor Appointment Confirmation System

## Overview

The HeartGuardian application now includes a comprehensive doctor appointment confirmation system that allows doctors to review, confirm, or reject patient appointment requests. This system ensures proper medical oversight and provides patients with clear status updates.

## How It Works

### 1. Appointment Request Process

1. **Patient Submits Request**: When a patient books an appointment, it's initially marked as "pending"
2. **Doctor Review**: Doctors (admin users) can view all pending appointments in their dashboard
3. **Decision**: Doctors can either confirm or reject the appointment with notes
4. **Notification**: Patients receive immediate notifications about their appointment status

### 2. Appointment Status Flow

```
Pending → Confirmed/Rejected → Completed (if confirmed)
   ↓           ↓
Patient    Patient receives
submits    notification
request
```

### 3. Status Types

- **Pending**: Initial state when patient submits request
- **Confirmed**: Doctor has approved the appointment
- **Rejected**: Doctor has declined the appointment (with reason)
- **Completed**: Appointment has been finished
- **Cancelled**: Patient or doctor cancelled the appointment

## Features

### For Patients

1. **Clear Status Tracking**: See real-time status of appointment requests
2. **Notifications**: Receive immediate updates when appointments are confirmed/rejected
3. **Cancellation**: Cancel appointments before they're confirmed
4. **History**: View complete appointment history with status

### For Doctors (Admin Users)

1. **Appointment Management Dashboard**: Review all pending appointments
2. **Confirmation/Rejection**: Approve or decline appointments with notes
3. **Patient Information**: View patient details and risk assessments
4. **Status Updates**: Mark appointments as completed
5. **Real-time Updates**: Auto-refreshing dashboard

### Notification System

- **In-app Notifications**: Users see notifications in their dashboard
- **Status Badges**: Color-coded status indicators
- **Email Notifications**: Placeholder for email integration
- **Unread Count**: Notification bell shows unread count

## Database Schema

### Updated Appointment Model

```python
class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'))
    doctor_name = db.Column(db.String(100))
    appointment_date = db.Column(db.DateTime)
    appointment_time = db.Column(db.String(10))
    reason = db.Column(db.Text)
    status = db.Column(db.String(20), default='pending')  # NEW
    doctor_notes = db.Column(db.Text)  # NEW
    confirmation_date = db.Column(db.DateTime)  # NEW
    confirmed_by = db.Column(db.String(100))  # NEW
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

### New Notification Model

```python
class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    appointment_id = db.Column(db.Integer, db.ForeignKey('appointments.id'))
    title = db.Column(db.String(200))
    message = db.Column(db.Text)
    type = db.Column(db.String(50))  # appointment_confirmed, appointment_rejected
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
```

## Routes and Endpoints

### Patient Routes
- `GET /dashboard` - View appointments with status
- `GET /notifications` - View all notifications
- `POST /appointment/<id>/cancel` - Cancel own appointment

### Doctor Routes (Admin Only)
- `GET /doctor/appointments` - Doctor appointment management dashboard
- `POST /doctor/appointment/<id>/confirm` - Confirm appointment
- `POST /doctor/appointment/<id>/reject` - Reject appointment with reason
- `POST /doctor/appointment/<id>/complete` - Mark appointment as completed

### Notification Routes
- `GET /notifications` - View all notifications
- `POST /notifications/mark-read/<id>` - Mark notification as read

## User Interface

### Patient Dashboard
- **Upcoming Appointments**: Shows pending and confirmed appointments
- **Status Badges**: Color-coded status indicators
- **Notification Bell**: Shows unread notification count
- **Recent Appointments**: Complete appointment history

### Doctor Dashboard
- **Pending Appointments**: List of appointments awaiting review
- **Confirmed Appointments**: Upcoming confirmed appointments
- **Action Buttons**: Confirm, reject, or view details
- **Patient Information**: Risk level and assessment details

### Appointment Status Display
- **Pending**: Yellow badge with "Awaiting confirmation"
- **Confirmed**: Green badge with doctor name
- **Rejected**: Red badge with rejection reason
- **Completed**: Blue badge
- **Cancelled**: Gray badge

## Setup and Migration

### 1. Run Database Migration

```bash
python migrate_db.py
```

This script will:
- Add new columns to the appointments table
- Update existing appointments to 'pending' status
- Create the notifications table

### 2. Admin Access

To access doctor features, ensure the user has admin privileges:

```python
# In the database or through admin interface
user.is_admin = True
```

### 3. Email Integration (Future)

The system includes placeholder functions for email notifications:

```python
def send_appointment_confirmation_email(appointment):
    # Implement email sending logic
    pass

def send_appointment_rejection_email(appointment):
    # Implement email sending logic
    pass
```

## Security Features

1. **Admin-Only Access**: Doctor functions require admin privileges
2. **User Isolation**: Users can only see their own appointments
3. **Status Validation**: Prevents invalid status transitions
4. **Audit Trail**: Tracks who confirmed/rejected appointments and when

## Future Enhancements

1. **Email Notifications**: Real email integration
2. **SMS Notifications**: Text message alerts
3. **Calendar Integration**: Sync with external calendars
4. **Rescheduling**: Allow doctors to suggest alternative times
5. **Video Consultations**: Integrate with video calling platforms
6. **Prescription Management**: Digital prescription system
7. **Follow-up Scheduling**: Automatic follow-up appointment suggestions

## Usage Examples

### For Patients

1. **Book Appointment**: Submit request through assessment results
2. **Check Status**: View dashboard for appointment status
3. **Receive Notifications**: Get notified when doctor responds
4. **Cancel if Needed**: Cancel before confirmation

### For Doctors

1. **Review Requests**: Check pending appointments dashboard
2. **Make Decision**: Confirm or reject with notes
3. **Track Patients**: Monitor confirmed appointments
4. **Complete Visits**: Mark appointments as completed

## Benefits

1. **Medical Oversight**: Ensures proper doctor review of appointments
2. **Patient Communication**: Clear status updates and notifications
3. **Efficiency**: Streamlined appointment management
4. **Transparency**: Patients know exactly what's happening
5. **Flexibility**: Easy to confirm, reject, or modify appointments

This system provides a professional, medical-grade appointment management experience that ensures both patients and doctors have clear communication and proper oversight throughout the appointment process. 