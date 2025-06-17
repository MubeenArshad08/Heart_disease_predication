#!/usr/bin/env python3
"""
Database migration script for HeartGuardian
Updates existing database with new appointment and notification fields
"""

from app import app, db
from models import Appointment, Notification
from sqlalchemy import text

def migrate_database():
    """Migrate the database to add new fields"""
    with app.app_context():
        try:
            # Check if new columns exist
            inspector = db.inspect(db.engine)
            existing_columns = [col['name'] for col in inspector.get_columns('appointments')]
            
            # Add new columns to appointments table if they don't exist
            if 'doctor_notes' not in existing_columns:
                with db.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN doctor_notes TEXT"))
                    conn.commit()
                print("Added doctor_notes column to appointments table")
            
            if 'confirmation_date' not in existing_columns:
                with db.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN confirmation_date DATETIME"))
                    conn.commit()
                print("Added confirmation_date column to appointments table")
            
            if 'confirmed_by' not in existing_columns:
                with db.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE appointments ADD COLUMN confirmed_by VARCHAR(100)"))
                    conn.commit()
                print("Added confirmed_by column to appointments table")
            
            # Update existing appointments to have 'pending' status instead of 'scheduled'
            with db.engine.connect() as conn:
                conn.execute(text("UPDATE appointments SET status = 'pending' WHERE status = 'scheduled'"))
                conn.commit()
            print("Updated existing appointments status from 'scheduled' to 'pending'")
            
            # Create notifications table if it doesn't exist
            if not inspector.has_table('notifications'):
                db.create_all()
                print("Created notifications table")
            
            print("Database migration completed successfully!")
            
        except Exception as e:
            print(f"Migration error: {e}")
            db.session.rollback()

if __name__ == '__main__':
    migrate_database() 