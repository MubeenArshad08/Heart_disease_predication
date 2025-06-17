#!/usr/bin/env python3
"""
Admin User Management Script for HeartGuardian
This script helps create admin users and manage admin privileges.
"""

import sys
import os
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import User

def create_admin_user():
    """Create a new admin user interactively"""
    print("=" * 60)
    print("🫀 HeartGuardian Admin User Creation")
    print("=" * 60)
    
    # Get user details
    username = input("Enter username: ").strip()
    if not username:
        print("❌ Username cannot be empty!")
        return
    
    # Check if user already exists
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        print(f"❌ User '{username}' already exists!")
        if existing_user.is_admin:
            print("✅ This user is already an admin.")
        else:
            make_admin = input("Make this user an admin? (y/n): ").lower().strip()
            if make_admin == 'y':
                existing_user.is_admin = True
                db.session.commit()
                print("✅ User is now an admin!")
        return
    
    email = input("Enter email: ").strip()
    if not email:
        print("❌ Email cannot be empty!")
        return
    
    # Check if email already exists
    if User.query.filter_by(email=email).first():
        print("❌ Email already registered!")
        return
    
    first_name = input("Enter first name: ").strip()
    last_name = input("Enter last name: ").strip()
    phone = input("Enter phone number (optional): ").strip()
    
    password = input("Enter password: ").strip()
    if len(password) < 6:
        print("❌ Password must be at least 6 characters!")
        return
    
    confirm_password = input("Confirm password: ").strip()
    if password != confirm_password:
        print("❌ Passwords do not match!")
        return
    
    # Create the user
    try:
        user = User(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            phone=phone,
            is_admin=True  # Make this user an admin
        )
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        print("✅ Admin user created successfully!")
        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"Name: {first_name} {last_name}")
        print("Role: Admin")
        print("\nYou can now log in to the admin panel at /admin")
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error creating user: {e}")

def list_admin_users():
    """List all admin users"""
    print("=" * 60)
    print("👥 Current Admin Users")
    print("=" * 60)
    
    admin_users = User.query.filter_by(is_admin=True).all()
    
    if not admin_users:
        print("❌ No admin users found!")
        return
    
    print(f"Found {len(admin_users)} admin user(s):\n")
    
    for i, user in enumerate(admin_users, 1):
        print(f"{i}. Username: {user.username}")
        print(f"   Name: {user.first_name} {user.last_name}")
        print(f"   Email: {user.email}")
        print(f"   Created: {user.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

def list_all_users():
    """List all users with their admin status"""
    print("=" * 60)
    print("👥 All Users")
    print("=" * 60)
    
    users = User.query.order_by(User.created_at.desc()).all()
    
    if not users:
        print("❌ No users found!")
        return
    
    print(f"Found {len(users)} user(s):\n")
    
    for i, user in enumerate(users, 1):
        admin_status = "✅ Admin" if user.is_admin else "👤 User"
        print(f"{i}. {user.username} ({admin_status})")
        print(f"   Name: {user.first_name} {user.last_name}")
        print(f"   Email: {user.email}")
        print(f"   Created: {user.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

def toggle_admin_status():
    """Toggle admin status for an existing user"""
    print("=" * 60)
    print("🔄 Toggle Admin Status")
    print("=" * 60)
    
    username = input("Enter username to toggle admin status: ").strip()
    if not username:
        print("❌ Username cannot be empty!")
        return
    
    user = User.query.filter_by(username=username).first()
    if not user:
        print("❌ User not found!")
        return
    
    current_status = "Admin" if user.is_admin else "Regular User"
    new_status = "Regular User" if user.is_admin else "Admin"
    
    print(f"Current status: {current_status}")
    confirm = input(f"Change to {new_status}? (y/n): ").lower().strip()
    
    if confirm == 'y':
        try:
            user.is_admin = not user.is_admin
            db.session.commit()
            print(f"✅ {user.username} is now a {new_status}!")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Error updating user: {e}")
    else:
        print("❌ Operation cancelled.")

def show_system_stats():
    """Show system statistics"""
    print("=" * 60)
    print("📊 System Statistics")
    print("=" * 60)
    
    total_users = User.query.count()
    admin_users = User.query.filter_by(is_admin=True).count()
    regular_users = total_users - admin_users
    
    from models import Prediction, Appointment, Notification
    
    total_predictions = Prediction.query.count()
    total_appointments = Appointment.query.count()
    total_notifications = Notification.query.count()
    
    print(f"👥 Total Users: {total_users}")
    print(f"   ├─ Admin Users: {admin_users}")
    print(f"   └─ Regular Users: {regular_users}")
    print()
    print(f"🫀 Total Predictions: {total_predictions}")
    print(f"📅 Total Appointments: {total_appointments}")
    print(f"🔔 Total Notifications: {total_notifications}")
    print()

def main():
    """Main menu"""
    while True:
        print("\n" + "=" * 60)
        print("🫀 HeartGuardian Admin Management")
        print("=" * 60)
        print("1. Create Admin User")
        print("2. List Admin Users")
        print("3. List All Users")
        print("4. Toggle Admin Status")
        print("5. Show System Statistics")
        print("6. Exit")
        print("=" * 60)
        
        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == '1':
            create_admin_user()
        elif choice == '2':
            list_admin_users()
        elif choice == '3':
            list_all_users()
        elif choice == '4':
            toggle_admin_status()
        elif choice == '5':
            show_system_stats()
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice! Please enter 1-6.")
        
        input("\nPress Enter to continue...")

if __name__ == '__main__':
    with app.app_context():
        main() 