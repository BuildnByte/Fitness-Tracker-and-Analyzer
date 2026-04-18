import os
import django
import random
from datetime import datetime, timedelta

# Setup django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'fitness_Tracker_and_Adviser.settings')
django.setup()

# Import the database instance from views where Pyrebase is already set up
from tracker_and_Adviser.views import db

def seed_data_for_user(uid, days=7):
    print(f"Seeding {days} days of realistic data for user: {uid}...")
    
    end_date = datetime.now()
    
    # We simulate a starting weight so there's a realistic drift over the days
    current_weight = 75.0
    
    for i in range(days):
        # We start looking from past moving towards today
        # so target date drifts forward in time
        days_ago = (days - 1) - i
        target_date = end_date - timedelta(days=days_ago)
        date_str = target_date.strftime('%Y-%m-%d')
        
        # Generate realistic, semi-random daily health logging values
        health_record = {
            'date': date_str,
            'timestamp': target_date.isoformat(),
            'sleep_hours': round(random.uniform(5.5, 9.0), 1),
            'sleep_quality': random.randint(2, 5),
            'total_calories': random.randint(1800, 2800),
            'protein': random.randint(60, 160),
            'carbs': random.randint(150, 300),
            'water_intake': round(random.uniform(1.5, 4.0), 1),
            'workout_duration': random.randint(0, 90),
            'workout_type': random.choice(['cardio', 'strength', 'mixed', 'flexibility', 'none']),
            'junk_food_level': random.randint(1, 5),
            'calorie_balance': random.randint(-400, 400)
        }
        
        # Simulate slight random weight fluctuation per day (-0.3 to 0.3 kg)
        current_weight += random.uniform(-0.3, 0.3)
        weigh_in_record = {
            'date': date_str,
            'weight': round(current_weight, 1),
            'unit': 'kg',
            'timestamp': target_date.isoformat()
        }
        
        try:
            # Commit health record mapping
            db.child("health_records").child(uid).child(date_str).set(health_record)
            # Commit weigh-in mapping
            db.child("weigh_ins").child(uid).child(date_str).set(weigh_in_record)
            print(f"[{date_str}] ✅ Health Log & Weigh-in ({round(current_weight, 1)}kg)")
        except Exception as e:
            print(f"[{date_str}] ❌ Error: {str(e)}")
            
    print("\n🎉 Seed complete! You can now load the dashboard for this user to trigger the AI plan.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python seed_user_data.py <FIREBASE_UID> [days_to_seed]")
        print("Example: python seed_user_data.py xA0v1r2gT... 7")
        sys.exit(1)
        
    uid = sys.argv[1]
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    seed_data_for_user(uid, days)
