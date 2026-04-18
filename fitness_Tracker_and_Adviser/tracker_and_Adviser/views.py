import firebase_admin
import pandas as pd
import numpy as np
from firebase_admin import credentials
from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.utils import timezone
from joblib import load
import os
import pyrebase
from django.conf import settings
import json
from datetime import datetime, timedelta
import pytz
from celery import shared_task  # If using Celery for background tasks
from django.utils.dateparse import parse_datetime
from django.utils.timezone import localtime
from .fuzzy_logic import get_fuzzy_scores_for_user
from .workout_templates import WORKOUT_TEMPLATES, GOAL_NORMALIZATION_MAP

def build_workout_schedule(goal, workout_days, fitness_state, confidence, alignment_signal, fuzzy_scores, has_easy_midweek):
    # Stage A - Template Selection
    normalized_goal = GOAL_NORMALIZATION_MAP.get(str(goal).lower().strip().replace(" ", "_"), "general")
    templates_for_goal = WORKOUT_TEMPLATES.get(normalized_goal, WORKOUT_TEMPLATES["general"])
    
    clamped_days = max(3, min(workout_days, 5))
    template_list = list(templates_for_goal.get(clamped_days, templates_for_goal[4]))
    
    # Stage B - Fitness State Modifications
    state_modifier = ""
    if fitness_state == 'Progressing':
        if confidence == 'high': state_modifier = " — progressive overload, push slightly harder than last week."
        elif confidence == 'medium': state_modifier = " — maintain current load, focus on form."
    elif fitness_state == 'Recovering':
        state_modifier = " — reduced intensity, focus on movement quality."
    elif fitness_state == 'Overtraining':
        if confidence == 'medium': state_modifier = " — significantly reduced intensity."
    elif fitness_state == 'Plateauing':
        state_modifier = " — vary exercise selection to challenge adaptation."
        
    schedule = []
    first_active_index = -1
    midweek_active_index = -1
    
    for i, (day, desc) in enumerate(template_list):
        if "Rest Day" not in desc:
            if first_active_index == -1: first_active_index = i
            if day in ["wednesday", "thursday"]: midweek_active_index = i
            
            if fitness_state == 'Overtraining' and confidence == 'high':
                modified_desc = "Active Recovery: light walking or stretching only."
            else:
                modified_desc = desc + state_modifier
            schedule.append((day, modified_desc))
        else:
            schedule.append((day, desc))
            
    # Stage C - Easy Midweek Modification
    if has_easy_midweek:
        wed_index = 2
        schedule[wed_index] = ("wednesday", "Active Recovery: mobility work, light stretching, or a short easy walk — your training load this week is high.")
        
    # Stage D - Goal Alignment Modification
    if alignment_signal == "behind" and first_active_index != -1:
        d, mod_desc = schedule[first_active_index]
        if "Rest Day" not in mod_desc and "Active Recovery" not in mod_desc:
            schedule[first_active_index] = (d, mod_desc + " — goal progress is behind target, push effort on this session.")
    elif alignment_signal == "ahead":
        # apply to midweek active or any active day
        idx = midweek_active_index if midweek_active_index != -1 else first_active_index
        if idx != -1:
            d, mod_desc = schedule[idx]
            if "Rest Day" not in mod_desc and "Active Recovery" not in mod_desc:
                schedule[idx] = (d, mod_desc + " — progress is ahead of target, maintain current effort, do not overtrain.")
                
    # Step 3 - Support Fields
    focus_map = {
        ('Progressing', 'high'): "Progressive overload week — increase load slightly on all major lifts and runs.",
        ('Progressing', 'medium'): "Steady progression week — focus on execution and consistent effort.",
        ('Progressing', 'low'): "Maintenance week — hold current workloads steady.",
        ('Recovering', 'high'): "Recovery focused week — volume reduced to allow physiological adaptation.",
        ('Recovering', 'medium'): "Recovery focused week — volume reduced to allow physiological adaptation.",
        ('Recovering', 'low'): "Recovery focused week — volume reduced to allow physiological adaptation.",
        ('Overtraining', 'high'): "Deload week — significantly reduced volume and intensity, prioritize sleep and nutrition.",
        ('Overtraining', 'medium'): "Deload week — significantly reduced volume and intensity, prioritize sleep and nutrition.",
        ('Overtraining', 'low'): "Deload week — significantly reduced volume and intensity, prioritize sleep and nutrition.",
        ('Plateauing', 'high'): "Variation week — change exercise selection and rep ranges to break adaptation plateau.",
        ('Plateauing', 'medium'): "Variation week — change exercise selection and rep ranges to break adaptation plateau.",
        ('Plateauing', 'low'): "Variation week — change exercise selection and rep ranges to break adaptation plateau."
    }
    
    weekly_focus = focus_map.get((fitness_state, confidence), f"{fitness_state} week — adjust accordingly.")
    
    rationale_parts = []
    if fuzzy_scores.get('recovery_score', 1.0) < 0.4:
        rationale_parts.append("your recovery score is low indicating insufficient sleep or rest")
    if fuzzy_scores.get('training_load', 0.0) > 0.8:
        rationale_parts.append("your training load is high indicating significant weekly volume")
    if fuzzy_scores.get('nutrition_adherence', 1.0) < 0.4:
        rationale_parts.append("your nutrition adherence needs improvement")
    if alignment_signal == "behind":
        rationale_parts.append("your progress is behind your goal expectation")
        
    if rationale_parts:
        state_rationale = f"We have structured this plan because {' and '.join(rationale_parts)}. Your classified state is {fitness_state}."
    else:
        state_rationale = f"Your metrics are well-aligned. The AI has classified your state as {fitness_state} and mapped a schedule optimized for your goal."

    active_days = []
    rest_days = []
    schedule_dict = {}
    for d, desc in schedule:
        schedule_dict[d] = desc
        if "Rest Day" in desc or "Active Recovery" in desc or "Recovery:" in desc:
            rest_days.append(d)
        else:
            active_days.append(d)
            
    days_breakdown = {
        "active_days": active_days,
        "rest_days": rest_days,
        "summary": f"{len(active_days)} active training days, {len(rest_days)} rest or recovery days"
    }
    
    return {
        "schedule_dict": schedule_dict,
        "weekly_focus": weekly_focus,
        "state_rationale": state_rationale,
        "days_breakdown": days_breakdown
    }

def get_week_start_end_dates(target_date=None):
    """Get the start (Monday) and end (Sunday) dates for a given week"""
    if target_date is None:
        target_date = timezone.now().date()
    
    # Find the Monday of the current week
    days_since_monday = target_date.weekday()
    week_start = target_date - timedelta(days=days_since_monday)
    week_end = week_start + timedelta(days=6)
    
    return week_start, week_end

def get_next_week_dates():
    """Get start and end dates for next week"""
    today = timezone.now().date()
    days_until_next_monday = 7 - today.weekday()
    next_monday = today + timedelta(days=days_until_next_monday)
    next_sunday = next_monday + timedelta(days=6)
    
    return next_monday, next_sunday

def save_weekly_plan_to_firebase(uid, week_start_date, diet_plan, workout_plan, plan_context=None):
    """Save the generated weekly plan to Firebase"""
    try:
        week_key = week_start_date.strftime('%Y-%m-%d')  # Use Monday's date as key
        
        plan_data = {
            'week_start_date': week_start_date.strftime('%Y-%m-%d'),
            'week_end_date': (week_start_date + timedelta(days=6)).strftime('%Y-%m-%d'),
            'generated_at': timezone.now().isoformat(),
            'diet_plan': diet_plan,
            'workout_plan': workout_plan,
            'context': plan_context,
            'is_current': True  # Mark as current plan
        }

        
        # Save the plan
        db.child("weekly_plans").child(uid).child(week_key).set(plan_data)
        
        # Mark previous plans as not current
        all_plans = db.child("weekly_plans").child(uid).get().val()
        if all_plans:
            for plan_key, plan in all_plans.items():
                if plan_key != week_key:
                    db.child("weekly_plans").child(uid).child(plan_key).update({'is_current': False})
        
        print(f"Successfully saved weekly plan for user {uid} starting {week_start_date}")
        return True
        
    except Exception as e:
        print(f"Error saving weekly plan for user {uid}: {str(e)}")
        return False

def get_weekly_plan_from_firebase(uid, week_start_date=None):
    """Retrieve weekly plan from Firebase"""
    try:
        if week_start_date is None:
            # Get current week's plan
            current_monday, _ = get_week_start_end_dates()
            week_key = current_monday.strftime('%Y-%m-%d')
        else:
            week_key = week_start_date.strftime('%Y-%m-%d')
        
        plan = db.child("weekly_plans").child(uid).child(week_key).get().val()
        
        if plan:
            print(f"Retrieved weekly plan for user {uid} for week {week_key}")
            return plan
        else:
            print(f"No weekly plan found for user {uid} for week {week_key}")
            return None
            
    except Exception as e:
        print(f"Error retrieving weekly plan for user {uid}: {str(e)}")
        return None

def get_current_weekly_plan(uid):
    """Get the current active weekly plan for a user"""
    try:
        all_plans = db.child("weekly_plans").child(uid).get().val()
        if not all_plans:
            return None
        
        # Find the current plan
        for plan_key, plan in all_plans.items():
            if plan.get('is_current', False):
                return plan
        
        # If no current plan found, try to get this week's plan
        current_monday, _ = get_week_start_end_dates()
        return get_weekly_plan_from_firebase(uid, current_monday)
        
    except Exception as e:
        print(f"Error getting current weekly plan for user {uid}: {str(e)}")
        return None

def generate_weekly_plan_for_user(uid):
    """Generate and save weekly plan for a specific user"""
    try:
        print(f"Generating weekly plan for user {uid}")
        
        # Get the data from the current week (Monday to Sunday)
        current_monday, current_sunday = get_week_start_end_dates()
        
        # Get health records for the current week
        records = get_health_records_from_firebase(uid, current_monday, current_sunday)
        
        if not records or len(records) < 3:
            print(f"Insufficient data for user {uid}: {len(records) if records else 0} records")
            return False
        
        # Generate the plan using existing function
        combined_plan = predict_weekly_health_and_fitness(uid)
        
        if not combined_plan:
            print(f"Failed to generate plan for user {uid}")
            return False
        
        # Get next week's dates
        next_monday, next_sunday = get_next_week_dates()
        
        # Save the plan for next week
        success = save_weekly_plan_to_firebase(
            uid, 
            next_monday, 
            combined_plan['diet_plan'], 
            combined_plan['workout_plan'],
            combined_plan.get('context')
        )
        
        if success:
            print(f"Successfully generated and saved weekly plan for user {uid}")
        
        return success
        
    except Exception as e:
        print(f"Error generating weekly plan for user {uid}: {str(e)}")
        return False

def generate_all_weekly_plans():
    """Generate weekly plans for all users - called on Sunday nights"""
    try:
        print("Starting weekly plan generation for all users...")
        
        # Get all users from Firebase
        all_users = db.child("users").get().val()
        
        if not all_users:
            print("No users found")
            return
        
        success_count = 0
        total_users = len(all_users)
        
        for uid in all_users.keys():
            try:
                if generate_weekly_plan_for_user(uid):
                    success_count += 1
                    print(f"✓ Generated plan for user {uid}")
                else:
                    print(f"✗ Failed to generate plan for user {uid}")
            except Exception as e:
                print(f"✗ Error processing user {uid}: {str(e)}")
        
        print(f"Weekly plan generation completed: {success_count}/{total_users} users")
        
        # Log the batch operation
        batch_log = {
            'timestamp': timezone.now().isoformat(),
            'total_users': total_users,
            'successful_plans': success_count,
            'failed_plans': total_users - success_count
        }
        db.child("plan_generation_logs").push(batch_log)
        
    except Exception as e:
        print(f"Error in batch weekly plan generation: {str(e)}")

# Celery task for background processing (optional)
@shared_task
def weekly_plan_generation_task():
    """Celery task to generate weekly plans"""
    generate_all_weekly_plans()

# # Django management command
# class Command(BaseCommand):
#     help = 'Generate weekly plans for all users'
    
#     def handle(self, *args, **options):
#         self.stdout.write('Starting weekly plan generation...')
#         generate_all_weekly_plans()
#         self.stdout.write(self.style.SUCCESS('Weekly plan generation completed'))

firebase_config = {
    "apiKey": settings.FIREBASE_API_KEY,
    "authDomain": "fitness-tracker-and-adviser.firebaseapp.com",
    "databaseURL": "https://fitness-tracker-and-adviser-default-rtdb.firebaseio.com",
    "projectId": "fitness-tracker-and-adviser",
    "storageBucket": "fitness-tracker-and-adviser.firebasestorage.app",
    "messagingSenderId": "553419005060",
    "appId": "1:553419005060:web:8c918ae2230559e131b6d1",
    "measurementId": "G-3D3K809NV3"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

firebase_json_str = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')

# Check if the app is already initialized (CRITICAL for serverless)
if not firebase_admin._apps:
    if firebase_json_str:
        # Parse the JSON string into a dictionary
        firebase_config = json.loads(firebase_json_str)
        
        # Initialize the app using the dictionary, NOT a file path
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
    else:
        # This part is optional: for local testing
        # It will use a local file if the env var isn't set
        try:
            cred = credentials.Certificate('firebase_config.json') # Assumes file is in root
            firebase_admin.initialize_app(cred)
        except FileNotFoundError:
            print("ERROR: Firebase credentials not found.")

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            uid = user['localId']
            user_data = db.child("users").child(uid).get().val()

            # Check if user_data exists
            if user_data:
                request.session['user'] = {
                    'email': email,
                    'name': user_data.get('name', ''),
                    'goal': user_data.get('goal', ''),
                    'uid': uid
                }
                
                messages.success(request, "Login successful!")
                return redirect('dashboard')
            else:
                messages.error(request, "User profile not found.")
        except Exception as e:
            try:
                error_detail = json.loads(e.args[1])['error']['message']
            except (IndexError, KeyError, json.JSONDecodeError):
                error_detail = str(e)
            messages.error(request, f"Login failed: {error_detail}")
    
    return render(request, 'login.html')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_week_dates(offset_weeks=0):
    """Get start and end dates for current week or offset weeks"""
    today = timezone.now().date()
    days_since_monday = today.weekday()
    week_start = today - timedelta(days=days_since_monday) + timedelta(weeks=offset_weeks)
    week_end = week_start + timedelta(days=6)
    return week_start, week_end

def safe_int(value, default=0):
    """Safely convert value to int"""
    try:
        return int(float(value)) if value not in [None, ''] else default
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        return float(value) if value not in [None, ''] else default
    except (ValueError, TypeError):
        return default

def clamp(value, min_val, max_val):
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

# ============================================================================
# FIREBASE DATA FUNCTIONS
# ============================================================================

def get_user_profile(uid):
    """Get user profile from Firebase"""
    try:
        user_data = db.child("users").child(uid).get().val()
        return user_data if user_data else {}
    except:
        return {}
    

def load_fitness_state_model():
    """Load the trained ensemble fitness state predictor pipeline"""
    try:
        model_path = os.path.join(settings.BASE_DIR, 'final_fitness_state_predictor.joblib')
        if not os.path.exists(model_path):
            print("Fitness state predictor model not found")
            return None
        
        model = load(model_path)
        print("Fitness state model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading fitness state model: {str(e)}")
        return None

def compute_goal_alignment(uid, user_goal):
    """
    Computes goal alignment signal from the past 3 to 4 weeks of weigh-in entries.
    Returns: 'ahead', 'on track', or 'behind'
    """
    try:
        weigh_ins = db.child("weigh_ins").child(uid).get().val()
        if not weigh_ins or len(weigh_ins) <= 1:
            return "on track"  # Default if less than two records exist
        
        # Parse and sort dates
        records = []
        for date_str, data in weigh_ins.items():
            records.append((datetime.strptime(date_str, '%Y-%m-%d').date(), data['weight']))
        
        records.sort(key=lambda x: x[0])
        
        recent_records = [r for r in records if (timezone.now().date() - r[0]).days <= 28] # Last 4 weeks
        if len(recent_records) <= 1:
            return "on track"
        
        first_record = recent_records[0]
        last_record = recent_records[-1]
        
        weeks_diff = max(1, (last_record[0] - first_record[0]).days / 7.0)
        weight_diff = last_record[1] - first_record[1]
        weekly_rate = weight_diff / weeks_diff
        
        if user_goal == 'weight_loss':
            # expected: -0.3 to -0.7
            if weekly_rate < -0.7: return "ahead"
            elif weekly_rate > -0.3: return "behind"
            else: return "on track"
        elif user_goal == 'muscle_gain':
            # expected: 0.15 to 0.3
            if weekly_rate > 0.3: return "ahead"
            elif weekly_rate < 0.15: return "behind"
            else: return "on track"
        else:
            # endurance or general: stable (-0.2 to +0.2)
            if weekly_rate < -0.2 or weekly_rate > 0.2: return "behind"
            else: return "on track"
    except Exception as e:
        print(f"Error calculating goal alignment: {str(e)}")
        return "on track"

def get_weekly_averages_for_workout_plan(uid):
    """
    Collect weekly averages & workout summary for workout plan model
    """
    stats = get_weekly_stats(uid)
    if not stats:
        return None
    
    # Build workout history text summary for the ML model
    workout_history = (
        f"{stats['active_days']} active days last week, "
        f"avg {stats['avg_workout_duration']} min per workout, "
        f"sleep avg {stats['avg_sleep']} h, "
        f"most popular workout: {stats['most_popular_workout']}."
    )
    
    return {
        "avg_sleep_hours": stats["avg_sleep"],
        "sleep_quality": stats["avg_sleep_quality"],
        "avg_calories": stats["avg_calories"],
        "protein": 100,  # 🔹 if you want, pull actual macros from diet records
        "carbs": 200,
        "fat": 70,
        "water_liters": stats["avg_water"],
        "goal": get_user_profile_from_firebase(uid).get("goal", "General Fitness"),
        "workout_history": workout_history
    }


def get_health_records_from_firebase(uid, start_date=None, end_date=None):
    """Helper function to get health records from Firebase with better error handling"""
    try:
        records_ref = db.child("health_records").child(uid)
        all_records = records_ref.get().val()
        
        if not all_records:
            return {}
        
        if start_date and end_date:
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            return {k: v for k, v in all_records.items() if start_str <= k <= end_str}
        
        return all_records
    except Exception as e:
        print(f"Error fetching health records: {str(e)}")
        return {}

def has_record_for_date(uid, check_date):
    """Check if user has a health record for specific date"""
    date_str = check_date.strftime('%Y-%m-%d')
    try:
        record = db.child("health_records").child(uid).child(date_str).get().val()
        return record is not None
    except:
        return False

# ============================================================================
# STATISTICS FUNCTIONS
# ============================================================================

def get_weekly_stats(uid):
    """Calculate comprehensive weekly statistics"""
    start_date, end_date = get_week_dates()
    records = get_health_records_from_firebase(uid, start_date, end_date)
    
    if not records:
        return None
    
    totals = {
        'sleep': 0, 'sleep_quality': 0, 'calories': 0,
        'water': 0, 'workout_duration': 0, 'junk_food': 0,
        'active_days': 0
    }
    workout_types = []
    
    for record in records.values():
        totals['sleep'] += safe_float(record.get('sleep_hours', 0))
        totals['sleep_quality'] += safe_int(record.get('sleep_quality', 0))
        totals['calories'] += safe_int(record.get('total_calories', 0))
        totals['water'] += safe_float(record.get('water_intake', 0))
        totals['junk_food'] += safe_int(record.get('junk_food_level', 0))
        
        workout_duration = safe_int(record.get('workout_duration', 0))
        totals['workout_duration'] += workout_duration
        if workout_duration > 0:
            totals['active_days'] += 1
        
        # Handle workout types
        types = record.get('workout_types', [])
        if isinstance(types, str):
            types = json.loads(types) if types else []
        workout_types.extend(types)
    
    count = len(records)
    most_popular = Counter(workout_types).most_common(1)[0][0] if workout_types else 'none'
    
    return {
        'avg_sleep': round(totals['sleep'] / count, 1),
        'avg_sleep_quality': round(totals['sleep_quality'] / count, 1),
        'avg_calories': round(totals['calories'] / count),
        'avg_water': round(totals['water'] / count, 1),
        'avg_workout_duration': round(totals['workout_duration'] / count),
        'avg_junk_food': round(totals['junk_food'] / count, 1),
        'total_days': count,
        'active_days': totals['active_days'],
        'most_popular_workout': most_popular
    }

def get_current_streak(uid):
    """Calculate current consecutive day streak (simplified)"""
    try:
        records = db.child("health_records").child(uid).get().val()
        if not records:
            return 0
        
        # Get sorted dates (most recent first)
        dates = sorted([
            datetime.strptime(d, '%Y-%m-%d').date() 
            for d in records.keys()
        ], reverse=True)
        
        today = timezone.now().date()
        
        # Must have record today or yesterday to have active streak
        if (today - dates[0]).days > 1:
            return 0
        
        # Count consecutive days
        streak = 1
        for i in range(1, len(dates)):
            if (dates[i-1] - dates[i]).days == 1:
                streak += 1
            else:
                break
        
        return streak
    except Exception as e:
        print(f"Error calculating streak: {str(e)}")
        return 0

# ============================================================================
# BASELINE & PROGRESS TRACKING
# ============================================================================

def get_baseline_targets(goal):
    """Get baseline targets for fitness goal"""
    baselines = {
        'weight_loss': {
            'sleep_hours': 7.5, 'sleep_quality': 4.0, 'calories': 1800,
            'water_intake': 2.5, 'workout_duration': 45, 'active_days': 5,
            'junk_food_level': 1.0, 'target_description': 'Healthy weight loss'
        },
        'muscle_gain': {
            'sleep_hours': 8.0, 'sleep_quality': 4.0, 'calories': 2400,
            'water_intake': 3.0, 'workout_duration': 60, 'active_days': 5,
            'junk_food_level': 1.5, 'target_description': 'Muscle building'
        },
        'endurance': {
            'sleep_hours': 7.5, 'sleep_quality': 4.0, 'calories': 2200,
            'water_intake': 3.5, 'workout_duration': 75, 'active_days': 6,
            'junk_food_level': 1.0, 'target_description': 'Endurance training'
        },
        'general': {
            'sleep_hours': 7.5, 'sleep_quality': 3.5, 'calories': 2000,
            'water_intake': 2.5, 'workout_duration': 30, 'active_days': 4,
            'junk_food_level': 2.0, 'target_description': 'General wellness'
        }
    }
    return baselines.get(goal.lower(), baselines['general'])

def save_user_baseline(uid, goal, custom_targets=None):
    """Save user's baseline targets"""
    try:
        targets = get_baseline_targets(goal)
        if custom_targets:
            targets.update(custom_targets)
        
        baseline_data = {
            'goal': goal,
            'targets': targets,
            'created_at': timezone.now().isoformat(),
            'is_active': True
        }
        db.child("user_baselines").child(uid).set(baseline_data)
        return True
    except Exception as e:
        print(f"Error saving baseline: {str(e)}")
        return False

def get_user_baseline(uid):
    """Retrieve user's baseline"""
    try:
        return db.child("user_baselines").child(uid).get().val()
    except:
        return None

def calculate_progress_score(current_stats, baseline_targets):
    """Calculate progress score comparing current to baseline"""
    if not current_stats or not baseline_targets:
        return None
    
    targets = baseline_targets.get('targets', {})
    
    # Calculate individual scores (0-100)
    sleep_score = 0
    if current_stats.get('avg_sleep', 0) > 0:
        sleep_target = targets.get('sleep_hours', 7.5)
        sleep_actual = current_stats['avg_sleep']
        # Score between 0-100, penalize both over and under sleeping
        if sleep_actual >= sleep_target * 0.8 and sleep_actual <= sleep_target * 1.2:
            sleep_score = min(100, (sleep_actual / sleep_target) * 100)
        else:
            sleep_score = max(0, 100 - abs(sleep_actual - sleep_target) * 20)
    
    # Sleep Quality Progress (weight: 15%)
    sleep_quality_score = 0
    if current_stats.get('avg_sleep_quality', 0) > 0:
        quality_target = targets.get('sleep_quality', 3.5)
        quality_actual = current_stats['avg_sleep_quality']
        sleep_quality_score = min(100, (quality_actual / quality_target) * 100)
    
    # Workout Consistency Progress (weight: 25%)
    workout_score = 0
    active_days_target = targets.get('active_days', 4)
    active_days_actual = current_stats.get('active_days', 0)
    workout_score = min(100, (active_days_actual / active_days_target) * 100)
    
    # Workout Duration Progress (weight: 15%)
    duration_score = 0
    if current_stats.get('avg_workout_duration', 0) > 0:
        duration_target = targets.get('workout_duration', 30)
        duration_actual = current_stats['avg_workout_duration']
        duration_score = min(100, (duration_actual / duration_target) * 100)
    
    # Hydration Progress (weight: 10%)
    water_score = 0
    if current_stats.get('avg_water', 0) > 0:
        water_target = targets.get('water_intake', 2.5)
        water_actual = current_stats['avg_water']
        water_score = min(100, (water_actual / water_target) * 100)
    
    # Diet Quality Progress (weight: 15%) - Lower junk food is better
    diet_score = 0
    if 'avg_junk_food' in current_stats:
        junk_target = targets.get('junk_food_level', 2.0)
        junk_actual = current_stats['avg_junk_food']
        # Invert score - lower junk food = higher score
        if junk_actual <= junk_target:
            diet_score = 100
        else:
            diet_score = max(0, 100 - (junk_actual - junk_target) * 25)
    
    # Calculate weighted overall score
    weights = {
        'sleep': 0.20,
        'sleep_quality': 0.15,
        'workout_consistency': 0.25,
        'workout_duration': 0.15,
        'hydration': 0.10,
        'diet_quality': 0.15
    }
    
    scores = {
        'sleep': sleep_score,
        'sleep_quality': sleep_quality_score,
        'workout_consistency': workout_score,
        'workout_duration': duration_score,
        'hydration': water_score,
        'diet_quality': diet_score
    }
    
    # Weighted overall score
    weights = {'sleep': 0.20, 'sleep_quality': 0.15, 'workout_consistency': 0.25,
               'workout_duration': 0.15, 'hydration': 0.10, 'diet_quality': 0.15}
    
    overall_score = sum(scores[key] * weights[key] for key in scores)
    
    # Progress level
    if overall_score >= 90:
        level = {'level': 'Excellent', 'color': '#10b981', 'icon': '🏆'}
    elif overall_score >= 75:
        level = {'level': 'Great', 'color': '#059669', 'icon': '⭐'}
    elif overall_score >= 60:
        level = {'level': 'Good', 'color': '#3b82f6', 'icon': '👍'}
    elif overall_score >= 40:
        level = {'level': 'Fair', 'color': '#f59e0b', 'icon': '📈'}
    else:
        level = {'level': 'Needs Work', 'color': '#ef4444', 'icon': '💪'}
    
    return {
        'overall_score': round(overall_score, 1),
        'individual_scores': scores,
        'targets': targets,
        'current_values': {
            'sleep_hours': current_stats.get('avg_sleep', 0),
            'sleep_quality': current_stats.get('avg_sleep_quality', 0),
            'active_days': current_stats.get('active_days', 0),
            'workout_duration': current_stats.get('avg_workout_duration', 0),
            'water_intake': current_stats.get('avg_water', 0),
            'junk_food_level': current_stats.get('avg_junk_food', 0)
        },
        'progress_level': level
    }

def save_progress_history(uid, progress_data):
    """Save weekly progress for historical tracking"""
    try:
        week_key = timezone.now().date().strftime('%Y-%m-%d')
        history_data = {
            'week_date': week_key,
            'overall_score': progress_data['overall_score'],
            'individual_scores': progress_data['individual_scores'],
            'progress_level': progress_data['progress_level'],
            'recorded_at': timezone.now().isoformat()
        }
        db.child("progress_history").child(uid).child(week_key).set(history_data)
        return True
    except Exception as e:
        print(f"Error saving progress history: {str(e)}")
        return False

def get_progress_trend(uid, weeks=4):
    """Get progress trend over specified weeks"""
    try:
        history = db.child("progress_history").child(uid).order_by_key().limit_to_last(weeks).get().val()
        if not history:
            return None
        
        trend_data = [{'week': k, 'score': v.get('overall_score', 0),
                      'level': v.get('progress_level', {}).get('level', 'Unknown')}
                     for k, v in history.items()]
        
        # Determine trend direction
        if len(trend_data) >= 2:
            recent_avg = sum(item['score'] for item in trend_data[-2:]) / 2
            older_avg = (sum(item['score'] for item in trend_data[:-2]) / 
                        max(1, len(trend_data) - 2) if len(trend_data) > 2 else trend_data[0]['score'])
            
            if recent_avg > older_avg + 5:
                direction = 'improving'
            elif recent_avg < older_avg - 5:
                direction = 'declining'
            else:
                direction = 'stable'
        else:
            direction = 'new'
        
        return {'trend_data': trend_data, 'trend_direction': direction, 'weeks_tracked': len(trend_data)}
    except Exception as e:
        print(f"Error getting progress trend: {str(e)}")
        return None

# ============================================================================
# WEEKLY PLAN FUNCTIONS
# ============================================================================

def get_current_weekly_plan(uid):
    """Get current active weekly plan"""
    try:
        all_plans = db.child("weekly_plans").child(uid).get().val()
        if not all_plans:
            return None
        
        # Find current plan or most recent
        for plan_key, plan in all_plans.items():
            if plan.get('is_current', False):
                return plan
        
        # Fallback to current week's plan
        current_monday, _ = get_week_dates()
        week_key = current_monday.strftime('%Y-%m-%d')
        return db.child("weekly_plans").child(uid).child(week_key).get().val()
    except Exception as e:
        print(f"Error getting current plan: {str(e)}")
        return None

def save_weekly_plan(uid, week_start_date, diet_plan, workout_plan):
    """Save generated weekly plan"""
    try:
        week_key = week_start_date.strftime('%Y-%m-%d')
        
        plan_data = {
            'week_start_date': week_start_date.strftime('%Y-%m-%d'),
            'week_end_date': (week_start_date + timedelta(days=6)).strftime('%Y-%m-%d'),
            'generated_at': timezone.now().isoformat(),
            'diet_plan': diet_plan,
            'workout_plan': workout_plan,
            'is_current': True
        }
        
        db.child("weekly_plans").child(uid).child(week_key).set(plan_data)
        
        # Mark other plans as not current
        all_plans = db.child("weekly_plans").child(uid).get().val()
        if all_plans:
            for plan_key in all_plans.keys():
                if plan_key != week_key:
                    db.child("weekly_plans").child(uid).child(plan_key).update({'is_current': False})
        
        return True
    except Exception as e:
        print(f"Error saving weekly plan: {str(e)}")
        return False

# ============================================================================
# ML MODEL FUNCTIONS
# ============================================================================



def load_workout_model():
    """Load trained workout plan model"""
    try:
        model_path = os.path.join(settings.BASE_DIR, 'workout_plan_model.joblib')
        mapping_path = os.path.join(settings.BASE_DIR, 'class_to_plan.json')
        
        if not os.path.exists(model_path) or not os.path.exists(mapping_path):
            return None, None
        
        model = load(model_path)
        with open(mapping_path, "r") as f:
            class_to_plan = json.load(f)
        return model, class_to_plan
    except Exception as e:
        print(f"Error loading workout model: {str(e)}")
        return None, None

def generate_personalized_diet(week_sleep, avg_cal, protein, carbs, water_l, goal, plan_style):
    """Generate personalized diet plan from model output"""
    base_cal = float(avg_cal)
    base_pro = float(protein)
    base_carb = float(carbs)
    base_h2o = float(water_l)
    
    sleep_target = clamp(round(max(7.0, min(9.0, (week_sleep + 7.5) / 2)), 1), 6.5, 9.0)
    
    # Plan configurations
    plans = {
        'weight_loss': {
            'cal_mult': 0.82, 'pro_add': 20, 'pro_range': (80, 180),
            'carb_mult': 0.8, 'carb_range': (120, 300), 'h2o_add': 0.4,
            'headline': 'Calorie deficit with higher protein'
        },
        'muscle_gain': {
            'cal_mult': 1.12, 'pro_min': 110, 'pro_range': (110, 200),
            'carb_min': 260, 'carb_range': (220, 420), 'h2o_add': 0.3,
            'headline': 'Slight surplus with high protein'
        },
        'endurance': {
            'cal_range': (2000, 2700), 'pro_range': (70, 130),
            'carb_min': 280, 'carb_range': (260, 480), 'h2o_add': 0.5,
            'headline': 'Carb-focused fueling for endurance'
        },
        'general': {
            'cal_mult': 1.0, 'pro_min': 90, 'pro_range': (80, 160),
            'carb_range': (180, 360), 'h2o_min': 2.4,
            'headline': 'Balanced maintenance'
        }
    }
    
    config = plans.get(plan_style, plans['general'])
    
    # Calculate targets based on plan type
    if plan_style == 'weight_loss':
        kcal = int(base_cal * config['cal_mult'])
        pro = clamp(base_pro + config['pro_add'], *config['pro_range'])
        carb = clamp(base_carb * config['carb_mult'], *config['carb_range'])
        h2o = clamp(base_h2o + config['h2o_add'], 2.2, 4.0)
    elif plan_style == 'muscle_gain':
        kcal = int(base_cal * config['cal_mult'])
        pro = clamp(max(base_pro, config['pro_min']), *config['pro_range'])
        carb = clamp(max(base_carb, config['carb_min']), *config['carb_range'])
        h2o = clamp(base_h2o + config['h2o_add'], 2.4, 4.0)
    elif plan_style == 'endurance':
        kcal = int(clamp(base_cal, *config['cal_range']))
        pro = clamp(base_pro, *config['pro_range'])
        carb = clamp(max(base_carb, config['carb_min']), *config['carb_range'])
        h2o = clamp(base_h2o + config['h2o_add'], 2.5, 4.0)
    else:  # general
        kcal = int(base_cal)
        pro = clamp(max(base_pro, config['pro_min']), *config['pro_range'])
        carb = clamp(base_carb, *config['carb_range'])
        h2o = clamp(max(base_h2o, config.get('h2o_min', 2.4)), 2.4, 4.0)
    
    bullets = [
        f"Calories: ~{kcal} kcal/day",
        f"Protein: {int(pro)} g/day",
        f"Carbs: {int(carb)} g/day",
        f"Water: {round(h2o, 1)} L/day",
        f"Sleep: {sleep_target} h/night"
    ]
    
    return {
        "plan_style": plan_style,
        "targets": {
            "calories_kcal": kcal,
            "protein_g": int(pro),
            "carbs_g": int(carb),
            "water_l": round(h2o, 1),
            "sleep_h": sleep_target
        },
        "summary": f"{config['headline']}. Focus on consistent meals and whole foods.",
        "bullets": bullets,
        "headline": headline
    }

def get_weekly_averages_for_diet_plan(uid):
    """Calculate weekly averages needed for diet plan model"""
    try:
        # Get last 7 days of data
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=6)
        
        records = get_health_records_from_firebase(uid, start_date, end_date)
        
        if not records or len(records) < 3:  # Need at least 3 days of data
            return None
        
        # Calculate averages
        total_sleep = 0
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_water = 0
        count = len(records)
        
        for record in records.values():
            total_sleep += float(record.get('sleep_hours', 0))
            total_calories += int(record.get('total_calories', 0))
            total_protein += int(record.get('protein', 0))
            total_carbs += int(record.get('carbs', 0))
            total_water += float(record.get('water_intake', 0))
        
        return {
            'week_avg_sleep': round(total_sleep / count, 1),
            'avg_calories': round(total_calories / count),
            'protein': round(total_protein / count),
            'carbs': round(total_carbs / count),
            'water_l': round(total_water / count, 1)
        }
        
    except Exception as e:
        print(f"Error calculating weekly averages: {str(e)}")
        return None

def predict_weekly_health_and_fitness(uid):
    """
    Predict fitness state, diet priorities, and workout plans based on the new ensemble model.
    """
    model = load_fitness_state_model()
    if not model:
        print("Model could not be loaded.")
        return None

    # Gather required weekly stats and data
    weekly_stats = get_weekly_stats(uid)
    diet_weekly = get_weekly_averages_for_diet_plan(uid)
    if not weekly_stats or not diet_weekly:
        print("Insufficient weekly stats or diet data.")
        return None

    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=6)
    health_records = get_health_records_from_firebase(uid, start_date, end_date)

    user_profile = get_user_profile_from_firebase(uid)
    goal = user_profile.get("goal", "general")
    
    # 1. Fuzzy scores
    fuzzy_scores = get_fuzzy_scores_for_user(weekly_stats, diet_weekly, health_records)
    
    # Calculate a simple average calorie_balance from the records
    total_bal = 0
    bal_count = 0
    for r in health_records.values():
        total_bal += float(r.get('calorie_balance', 0))
        bal_count += 1
    avg_cal_balance = total_bal / bal_count if bal_count > 0 else 0

    # 2. Build feature vector for pipeline
    df = pd.DataFrame([{
        'recovery_score': fuzzy_scores["recovery_score"],
        'nutrition_score': fuzzy_scores["nutrition_adherence"],  # name mapping adjustment
        'training_load_score': fuzzy_scores["training_load"],   # name mapping
        'hydration_score': fuzzy_scores["hydration_score"],
        'bmi': float(user_profile.get('bmi', 24.0)),
        'age': int(user_profile.get('age', 30)),
        'calorie_balance': avg_cal_balance,
        'goal_weight_loss': 1.0 if goal == 'weight_loss' else 0.0,
        'goal_muscle_gain': 1.0 if goal == 'muscle_gain' else 0.0,
        'goal_endurance': 1.0 if goal == 'endurance' else 0.0,
        'goal_general': 1.0 if goal == 'general' else 0.0
    }])

    # 3. Model Prediction
    try:
        prob_vec = model.predict_proba(df)[0]
        classes_arr = model.classes_
        
        # Get highest prob class
        max_idx = np.argmax(prob_vec)
        winning_class = str(classes_arr[max_idx])
        winning_prob = prob_vec[max_idx]
        
        # Determine confidence
        if winning_prob > 0.70:
            confidence = "high"
        elif winning_prob >= 0.50:
            confidence = "medium"
        else:
            confidence = "low"
            
        prob_dict = {str(c): float(p) for c, p in zip(classes_arr, prob_vec)}
    except Exception as e:
        print("Prediction error:", e)
        winning_class = "Progressing"
        winning_prob = 1.0
        confidence = "medium"
        prob_dict = {winning_class: 1.0}
        
    # 4. Computed goal alignment signal
    alignment_signal = compute_goal_alignment(uid, goal)
    
    # 5. Personalization Layer: Diet
    # TDEE
    try:
        age_num = int(user_profile.get('age', 30))
        weight_num = float(user_profile.get('weight', 70))
        height_num = float(user_profile.get('height', 170))
        sex = str(user_profile.get('biological_sex', 'male')).lower()
        al = user_profile.get('activity_level', 'moderately_active').lower()
        if sex == 'female':
            bmr = (10 * weight_num) + (6.25 * height_num) - (5 * age_num) - 161
        else:
            bmr = (10 * weight_num) + (6.25 * height_num) - (5 * age_num) + 5
        multipliers = {'sedentary': 1.2, 'lightly_active': 1.375, 'moderately_active': 1.55, 'very_active': 1.725, 'extra_active': 1.9}
        tdee = int(bmr * multipliers.get(al, 1.55))
    except:
        tdee = 2000

    target_cal = tdee
    prot_target = weight_num * 1.8 if weight_num else 120
    headlines = []
    
    # Goal adjustments
    if goal == 'weight_loss': target_cal -= 400
    elif goal == 'muscle_gain': target_cal += 300
    elif goal == 'endurance': target_cal += 100
    
    # Fitness state adjustment scaled by probability
    if winning_class == 'Recovering':
        target_cal += int(200 * prob_dict.get('Recovering', 0))
        headlines.append("Increased calories prioritizing recovery.")
    elif winning_class == 'Overtraining':
        target_cal += int(250 * prob_dict.get('Overtraining', 0))
        headlines.append("Extra calories added to combat overtraining fatigue.")
    elif winning_class == 'Plateauing':
        target_cal -= int(100 * prob_dict.get('Plateauing', 0))
        headlines.append("Slight reduction in calories to break the plateau.")
        
    # Goal alignment modifier
    if alignment_signal == "behind":
        if goal == 'weight_loss':
            target_cal -= 50
        elif goal == 'muscle_gain':
            target_cal += 100
    elif alignment_signal == "ahead":
        if goal == 'weight_loss':
            target_cal += 50
            
    # Fuzzy score feedback
    if fuzzy_scores["nutrition_adherence"] < 0.4:
        headlines.append("Focus on improving diet quality (more whole foods).")
    if fuzzy_scores["hydration_score"] < 0.4:
        headlines.append("Increase water consumption significantly.")
        water_l = 3.5
    else:
        water_l = max(2.5, (weight_num * 35) / 1000)

    # Macros
    # Protein ~1.8g/kg or fixed. Fat ~25%. Carbs rest.
    prot = int(prot_target)
    fat = int((target_cal * 0.25) / 9)
    carbs = int((target_cal - (prot*4) - (fat*9)) / 4)
    if carbs < 50: carbs = 50
    
    diet_plan = {
        "plan_style": goal,
        "headline": "Personalized " + goal.replace('_', ' ').title() + " Plan",
        "targets": {
            "calories_kcal": int(target_cal),
            "protein_g": prot,
            "carbs_g": carbs,
            "fat_g": fat,
            "water_l": round(water_l, 1),
            "sleep_h": 8.0,
        },
        "summary": " ".join(headlines) if headlines else "Calibrated maintenance plan based on your recent activity.",
        "bullets": headlines if headlines else ["Keep up the balanced nutrition.", f"Aim for {prot}g of protein daily."]
    }
    
    # 6. Personalization Layer: Workout
    # Base days on fitness state
    workout_days = 4
    if winning_class == 'Recovering': workout_days = min(4, 4)
    elif winning_class == 'Overtraining': workout_days = 3
    elif winning_class == 'Progressing': workout_days = 5
    elif winning_class == 'Plateauing': workout_days = 5
    
    intensity = "Moderate"
    if winning_class == 'Overtraining':
        if confidence == 'high': intensity = "Light (Deload)"
        else: intensity = "Light-Moderate"
    elif winning_class == 'Progressing' and confidence in ['high', 'medium']:
        intensity = "High"
    
    if alignment_signal == "behind":
        if intensity == "Moderate": intensity = "Moderate-High"
        
    has_easy_midweek = False
    if fuzzy_scores["training_load"] > 0.8:
        has_easy_midweek = True
        
    workout_schedule_data = build_workout_schedule(
        goal, workout_days, winning_class, confidence, alignment_signal, fuzzy_scores, has_easy_midweek
    )
    
    workout_plan = {
        "predicted_label": winning_class,
        "weekly_schedule": workout_schedule_data["schedule_dict"],
        "days_recommended": workout_days,
        "intensity_target": intensity,
        "easy_midweek_required": has_easy_midweek,
        "advice": f"Aim for {workout_days} active days at {intensity} intensity.",
        "weekly_focus": workout_schedule_data["weekly_focus"],
        "state_rationale": workout_schedule_data["state_rationale"],
        "days_breakdown": workout_schedule_data["days_breakdown"]
    }

    # Context to save
    generation_context = {
        "fitness_state": winning_class,
        "probability_vector": prob_dict,
        "fuzzy_scores": fuzzy_scores,
        "goal_alignment": alignment_signal,
        "confidence": confidence,
        "weekly_focus": workout_schedule_data["weekly_focus"],
        "state_rationale": workout_schedule_data["state_rationale"]
    }
    
    return {
        "diet_plan": diet_plan,
        "workout_plan": workout_plan,
        "context": generation_context
    }



# Views for Diet Plan Feature

# Updated diet plan view
def diet_plan_view_updated(request):
    """View to display saved weekly diet plan"""
    user = request.session.get('user')
    if not user:
        messages.warning(request, "Please login to access diet plans.")
        return redirect('login')
    
    uid = user['uid']
    
    try:
        # Get saved weekly plan
        weekly_plan = get_current_weekly_plan(uid)
        weekly_stats = get_weekly_stats(uid)
        
        if not weekly_plan:
            messages.warning(request, "No weekly plan available. Plans are generated every Sunday night.")
            return redirect('dashboard')
        
        context = {
            'user': user,
            'diet_plan': weekly_plan.get('diet_plan'),
            'generation_context': weekly_plan.get('context'),
            'weekly_stats' :weekly_stats,
            'plan_info': {
                'week_start': weekly_plan.get('week_start_date'),
                'week_end': weekly_plan.get('week_end_date'),
                'generated_at': weekly_plan.get('generated_at')
            },
            'current_streak': get_current_streak(uid),
        }

        
        
        return render(request, 'diet_plan.html', context)
        
    except Exception as e:
        print(f"Error in diet_plan_view: {str(e)}")
        messages.error(request, "Error loading diet plan. Please try again.")
        return redirect('dashboard')


# Updated workout plan view
def workout_plan_view_updated(request):
    """View to display saved weekly workout plan"""
    user = request.session.get('user')
    if not user:
        return redirect('login')

    uid = user['uid']
    
    try:
        # Get saved weekly plan
        weekly_plan = get_current_weekly_plan(uid)
        
        if not weekly_plan:
            messages.warning(request, "No weekly plan available. Plans are generated every Sunday night.")
            return redirect('dashboard')
        
        context = {
            'user': user,
            'workout_plan': weekly_plan.get('workout_plan'),
            'generation_context': weekly_plan.get('context'),
            'day_names': ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
            'plan_info': {
                'week_start': weekly_plan.get('week_start_date'),
                'week_end': weekly_plan.get('week_end_date'),
                'generated_at': weekly_plan.get('generated_at')
            }
        }
        
        return render(request, "workout_plan.html", context)
        
    except Exception as e:
        print(f"Error in workout_plan_view: {str(e)}")
        messages.error(request, "Error loading workout plan. Please try again.")
        return redirect('dashboard')


# API endpoint to manually trigger plan generation (admin use)
@csrf_exempt
def generate_plans_manually(request):
    """Manual trigger for plan generation (for testing/admin)"""
    if request.method == 'POST':
        try:
            generate_all_weekly_plans()
            return JsonResponse({'success': True, 'message': 'Weekly plans generated successfully'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

def dashboard_view_with_progress(request):
    """Main dashboard with progress tracking"""
    user = request.session.get('user')
    if not user:
        messages.warning(request, "Please login to access dashboard.")
        return redirect('login')
    
    uid = user['uid']
    
    try:
        user_profile = get_user_profile(uid)
        today = timezone.now().date()
        has_today_record = has_record_for_date(uid, today)
        weekly_stats = get_weekly_stats(uid)
        
        # Chart data
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=6)
        recent_records = get_health_records_from_firebase(uid, start_date, end_date)
        chart_data = prepare_chart_data(recent_records, start_date, end_date)

        # Weekly plan
        combined_plan = get_current_weekly_plan(uid)
        
        # If no saved plan exists, generate one (fallback for new users)
        if not combined_plan and weekly_stats:
            print(f"No saved plan found for user {uid}, generating fallback plan")
            try:
                temp_plan = predict_weekly_health_and_fitness(uid)
                if temp_plan:
                    # Save this as current plan
                    current_monday, _ = get_week_start_end_dates()
                    save_weekly_plan_to_firebase(uid, current_monday, temp_plan['diet_plan'], temp_plan['workout_plan'], temp_plan.get('context'))
                    combined_plan = get_current_weekly_plan(uid)
            except Exception as e:
                print(f"Error generating fallback plan: {str(e)}")

        baseline = get_user_baseline(uid)
        progress_data = None
        if weekly_stats and baseline:
            progress_data = calculate_progress_score(weekly_stats, baseline)
            # Log progress for history keeping
            save_progress_history(uid, progress_data)
            
        progress_trend = get_progress_trend(uid)
        
        plan_week_info = {}
        if combined_plan:
            plan_week_info = {
                'start': combined_plan.get('week_start_date'),
                'end': combined_plan.get('week_end_date')
            }

        context = {
            'user': user,
            'user_profile': user_profile,
            'has_today_record': has_today_record,
            'weekly_stats': weekly_stats,
            'chart_data': json.dumps(chart_data, default=str),
            'current_streak': get_current_streak(uid),
            'total_records': len(get_health_records_from_firebase(uid)),
            'combined_plan': combined_plan,
            'plan_week_info': plan_week_info,
            'progress_data': progress_data,
            'progress_trend': progress_trend,
            'baseline': get_user_baseline(uid)
        }

    except Exception as e:
        print(f"Error in dashboard: {str(e)}")
        context = {
            'user': user,
            'user_profile': {},
            'has_today_record': False,
            'weekly_stats': None,
            'chart_data': json.dumps({}),
            'current_streak': 0,
            'total_records': 0
        }
        messages.error(request, "Error loading dashboard data.")
    
    return render(request, 'dashboard.html', context)

def form_view(request):
    """Health data input form"""
    user = request.session.get('user')
    if not user:
        messages.warning(request, "Please login to access this page.")
        return redirect('login')
    
    uid = user['uid']
    today = timezone.now().date()
    
    if has_record_for_date(uid, today):
        messages.info(request, "You have already submitted your health data for today!")
        return redirect('dashboard')
    
    return render(request, 'weekly_health_form.html', {'user': user})

@csrf_exempt
@require_http_methods(["POST"])
def submit_health_data(request):
    """Submit daily health data"""
    user = request.session.get('user')
    if not user:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    uid = user['uid']
    today = timezone.now().date()
    date_str = today.strftime('%Y-%m-%d')
    
    if has_record_for_date(uid, today):
        return JsonResponse({
            'success': False, 
            'error': 'You have already submitted data for today'
        }, status=400)
    
    try:
        data = json.loads(request.body)
        
        # Validate required fields
        required_fields = {
            'sleepHours': 'Sleep Hours', 'sleepQuality': 'Sleep Quality',
            'totalCalories': 'Total Calories', 'waterIntake': 'Water Intake',
            'junkFood': 'Junk Food Level', 'workoutDuration': 'Workout Duration',
            'workoutIntensity': 'Workout Intensity'
        }
        
        missing = [label for field, label in required_fields.items() 
                  if field not in data or data[field] in ['', None]]
        
        if missing:
            return JsonResponse({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing)}'
            }, status=400)
        
        # Parse workout types
        workout_types = data.get('workoutTypes', [])
        if isinstance(workout_types, str):
            workout_types = [workout_types] if workout_types else []
        elif not isinstance(workout_types, list):
            workout_types = []
        
        # Helper function to safely convert to int/float
        def safe_int(value, default=0):
            try:
                return int(float(value)) if value not in [None, ''] else default
            except (ValueError, TypeError):
                return default
        
        def safe_float(value, default=0.0):
            try:
                return float(value) if value not in [None, ''] else default
            except (ValueError, TypeError):
                return default
        
        # Calculate derived TDEE and calorie balance
        try:
            user_profile = get_user_profile_from_firebase(uid)
            age = int(user_profile.get('age', 30))
            weight = float(user_profile.get('weight', 70))
            height = float(user_profile.get('height', 170))
            bio_sex = user_profile.get('biological_sex', 'male').lower()
            activity_level = user_profile.get('activity_level', 'moderately_active').lower()

            # BMR Calculation (Mifflin-St Jeor)
            if bio_sex == 'female':
                bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
            else:
                bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
            
            # Activity multipliers
            multipliers = {
                'sedentary': 1.2,
                'lightly_active': 1.375,
                'moderately_active': 1.55,
                'very_active': 1.725,
                'extra_active': 1.9
            }
            activity_multiplier = multipliers.get(activity_level, 1.55)
            
            tdee = int(bmr * activity_multiplier)
            total_cals = safe_int(data['totalCalories'])
            calorie_balance = total_cals - tdee
            
        except Exception as e:
            print(f"Error calculating TDEE: {str(e)}")
            tdee = 2000
            total_cals = safe_int(data['totalCalories'])
            calorie_balance = total_cals - 2000

        # Prepare health record data for Firebase with validation
        try:
            health_record_data = {
                'date': date_str,
                'created_at': timezone.now().isoformat(),
                'user_id': uid,  # Add user reference
                
                # Sleep data
                'sleep_hours': safe_float(data['sleepHours']),
                'sleep_quality': safe_int(data['sleepQuality']),
                'bedtime': str(data.get('bedtime', '')),
                'wake_time': str(data.get('waketime', '')),
                
                # Nutrition data
                'total_calories': safe_int(data['totalCalories']),
                'water_intake': safe_float(data['waterIntake']),
                'carbs': safe_int(data.get('carbs', 0)),
                'protein': safe_int(data.get('protein', 0)),
                'fat': safe_int(data.get('fat', 0)),
                
                # Meal breakdown
                'breakfast_calories': safe_int(data.get('breakfast', 0)),
                'lunch_calories': safe_int(data.get('lunch', 0)),
                'dinner_calories': safe_int(data.get('dinner', 0)),
                'snacks_calories': safe_int(data.get('snacks', 0)),
                
                'junk_food_level': safe_int(data['junkFood']),
                
                # Workout data
                'workout_duration': safe_int(data['workoutDuration']),
                'workout_intensity': str(data['workoutIntensity']),
                'workout_types': workout_types,
                'calories_burned': safe_int(data.get('caloriesBurned', 0)),
                
                # Derived Fields
                'estimated_tdee': tdee,
                'calorie_balance': calorie_balance,
            }
            
            # Additional validation
            if health_record_data['sleep_hours'] < 0 or health_record_data['sleep_hours'] > 24:
                return JsonResponse({
                    'success': False,
                    'error': 'Sleep hours must be between 0 and 24'
                }, status=400)
            
            if health_record_data['sleep_quality'] < 1 or health_record_data['sleep_quality'] > 5:
                return JsonResponse({
                    'success': False,
                    'error': 'Sleep quality must be between 1 and 5'
                }, status=400)
            
            if health_record_data['total_calories'] < 0:
                return JsonResponse({
                    'success': False,
                    'error': 'Total calories cannot be negative'
                }, status=400)
            
            if health_record_data['water_intake'] < 0:
                return JsonResponse({
                    'success': False,
                    'error': 'Water intake cannot be negative'
                }, status=400)
            
            if health_record_data['junk_food_level'] < 0 or health_record_data['junk_food_level'] > 4:
                return JsonResponse({
                    'success': False,
                    'error': 'Junk food level must be between 0 and 4'
                }, status=400)
            
            if health_record_data['workout_duration'] < 0:
                return JsonResponse({
                    'success': False,
                    'error': 'Workout duration cannot be negative'
                }, status=400)
            
            if health_record_data['workout_intensity'] not in ['low', 'medium', 'high']:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid workout intensity level'
                }, status=400)
            
            print(f"Prepared health record data: {health_record_data}")  # Debug log
            
        except Exception as e:
            print(f"Error preparing health record data: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Error processing health data: {str(e)}'
            }, status=400)
        
        # Save to Firebase
        db.child("health_records").child(uid).child(date_str).set(health_record_data)
        
        # Verify save
        if not db.child("health_records").child(uid).child(date_str).get().val():
            raise Exception("Data was not saved to Firebase")
        
        return JsonResponse({
            'success': True, 
            'message': 'Health data saved successfully!',
            'record_date': date_str,
            'data_summary': {
                'sleep_hours': health_record_data['sleep_hours'],
                'total_calories': health_record_data['total_calories'],
                'workout_duration': health_record_data['workout_duration']
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        print(f"Error submitting health data: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return JsonResponse({
            'success': False, 
            'error': f'An error occurred: {str(e)}'
        }, status=500)

def diet_plan_view_updated(request):
    """Display saved weekly diet plan"""
    user = request.session.get('user')
    if not user:
        messages.warning(request, "Please login to access diet plans.")
        return redirect('login')
    
    uid = user['uid']
    
    try:
        weekly_plan = get_current_weekly_plan(uid)
        weekly_stats = get_weekly_stats(uid)
        
        if not weekly_plan:
            messages.warning(request, "No weekly plan available. Plans are generated every Sunday night.")
            return redirect('dashboard')
        
        context = {
            'user': user,
            'diet_plan': weekly_plan.get('diet_plan'),
            'weekly_stats': weekly_stats,
            'plan_info': {
                'week_start': weekly_plan.get('week_start_date'),
                'week_end': weekly_plan.get('week_end_date'),
                'generated_at': weekly_plan.get('generated_at')
            },
            'current_streak': get_current_streak(uid),
        }
        
        return render(request, 'diet_plan.html', context)
        
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)

def prepare_chart_data(records_dict, start_date, end_date):
    """Prepare data for dashboard charts from Firebase records"""
    
    # Create a dictionary with all dates in range
    date_range = []
    current_date = start_date
    while current_date <= end_date:
        date_range.append(current_date)
        current_date += timedelta(days=1)
    
    # Prepare data arrays
    dates = [d.strftime('%m/%d') for d in date_range]  # MM/DD format
    sleep_hours = []
    sleep_quality = []
    calories = []
    water_intake = []
    workout_duration = []
    junk_food_levels = []
    
    print(f"Processing chart data for date range: {start_date} to {end_date}")
    print(f"Available records: {list(records_dict.keys()) if records_dict else 'None'}")
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        if records_dict and date_str in records_dict:
            record = records_dict[date_str]
            print(f"Processing record for {date_str}: {record}")
            
            sleep_hours.append(float(record.get('sleep_hours', 0)))
            sleep_quality.append(int(record.get('sleep_quality', 0)))
            calories.append(int(record.get('total_calories', 0)))
            water_intake.append(float(record.get('water_intake', 0)))
            workout_duration.append(int(record.get('workout_duration', 0)))
            junk_food_levels.append(int(record.get('junk_food_level', 0)))
        else:
            # No data for this date
            sleep_hours.append(None)  # Use null instead of 0 for better chart display
            sleep_quality.append(None)
            calories.append(None)
            water_intake.append(None)
            workout_duration.append(None)
            junk_food_levels.append(None)
    
    # Get meal distribution from latest record
    meal_data = [0, 0, 0, 0]  # Default values
    meal_labels = ['Breakfast', 'Lunch', 'Dinner', 'Snacks']
    
    if records_dict:
        # Get the most recent record
        sorted_dates = sorted(records_dict.keys(), reverse=True)
        if sorted_dates:
            latest_record = records_dict[sorted_dates[0]]
            meal_data = [
                int(latest_record.get('breakfast_calories', 0)),
                int(latest_record.get('lunch_calories', 0)),
                int(latest_record.get('dinner_calories', 0)),
                int(latest_record.get('snacks_calories', 0))
            ]
    
    # Get workout types distribution
    workout_types_count = {}
    if records_dict:
        for record in records_dict.values():
            workout_types = record.get('workout_types', [])
            if isinstance(workout_types, str):
                try:
                    workout_types = json.loads(workout_types)
                except:
                    workout_types = [workout_types] if workout_types else []
            
            for workout_type in workout_types:
                if workout_type:  # Only count non-empty workout types
                    workout_types_count[workout_type] = workout_types_count.get(workout_type, 0) + 1
    
    chart_data = {
        'dates': dates,
        'sleep_hours': sleep_hours,
        'sleep_quality': sleep_quality,
        'calories': calories,
        'water_intake': water_intake,
        'workout_duration': workout_duration,
        'junk_food_levels': junk_food_levels,
        'meal_data': meal_data,
        'meal_labels': meal_labels,
        'workout_types_count': workout_types_count
    }
    
    print(f"Final chart data: {chart_data}")
    return chart_data

def signup_view(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        goal = request.POST.get('goal')
        
        # New rich profile fields
        try:
            age = int(request.POST.get('age', 0))
            weight = float(request.POST.get('weight', 0))
            height = float(request.POST.get('height', 0))
        except ValueError:
            messages.error(request, "Invalid numerical values provided.")
            return render(request, 'signup.html')
            
        biological_sex = request.POST.get('biological_sex')
        activity_level = request.POST.get('activity_level')

        # Basic validation
        if not all([name, email, password, goal, age, weight, height, biological_sex, activity_level]):
            messages.error(request, "All fields are required.")
            return render(request, 'signup.html')

        try:
            # Register the user
            user = auth.create_user_with_email_and_password(email, password)
            uid = user['localId']

            # Calculate immediate BMI
            height_m = height / 100.0
            bmi = round(weight / (height_m * height_m), 1) if height_m > 0 else 0

            # Store user profile in Realtime DB
            data = {
                "name": name,
                "email": email,
                "goal": goal,
                "age": age,
                "biological_sex": biological_sex,
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "bmi": bmi,
                "created_at": timezone.now().isoformat()
            }
            db.child("users").child(uid).set(data)
            
            # Set up baseline targets
            custom_targets = {}
            for field, key in [('target_sleep_hours', 'sleep_hours'), 
                              ('target_calories', 'calories'),
                              ('target_weekly_workouts', 'active_days'),
                              ('target_water', 'water_intake')]:
                value = request.POST.get(field)
                if value:
                    custom_targets[key] = float(value) if '.' in value else int(value)
            
            save_user_baseline(uid, goal, custom_targets if custom_targets else None)

            messages.success(request, "Registered successfully! Your baseline targets have been set.")
            return redirect('login')
        except Exception as e:
            try:
                error_detail = json.loads(e.args[1])['error']['message']
            except:
                error_detail = str(e)
            messages.error(request, f"Registration failed: {error_detail}")
    
    context = {
        'baseline_options': {
            'weight_loss': get_baseline_targets('weight_loss'),
            'muscle_gain': get_baseline_targets('muscle_gain'),
            'endurance': get_baseline_targets('endurance'),
            'general': get_baseline_targets('general')
        }
    }
    return render(request, 'signup.html', context)

def logout_view(request):
    """Logout function to clear session"""
    request.session.flush()  # This clears all session data
    messages.success(request, "Logged out successfully!")
    return redirect('login')

def weigh_in_view(request):
    user = request.session.get('user')
    if not user:
        messages.warning(request, "Please login to access weigh-in.")
        return redirect('login')
    
    uid = user['uid']
    
    if request.method == 'POST':
        try:
            new_weight = float(request.POST.get('weight', 0))
            if new_weight <= 0:
                messages.error(request, "Please enter a valid weight.")
                return redirect('weigh_in')
                
            # Fetch current profile to get height to recalculate BMI
            user_profile = db.child("users").child(uid).get().val()
            if not user_profile:
                messages.error(request, "User profile not found. Please try again.")
                return redirect('dashboard')
            
            height = float(user_profile.get('height', 0))
            height_m = height / 100.0
            new_bmi = round(new_weight / (height_m * height_m), 1) if height_m > 0 else 0
            
            # Update user profile
            db.child("users").child(uid).update({
                "weight": new_weight,
                "bmi": new_bmi,
                "last_weigh_in": timezone.now().isoformat()
            })
            
            # Log weekly weigh-in history
            date_str = timezone.now().date().strftime('%Y-%m-%d')
            db.child("weigh_ins").child(uid).child(date_str).set({
                "weight": new_weight,
                "bmi": new_bmi,
                "logged_at": timezone.now().isoformat()
            })
            
            messages.success(request, "Weigh-in recorded successfully!")
            return redirect('dashboard')
            
        except ValueError:
            messages.error(request, "Invalid weight value.")
            return redirect('weigh_in')
        except Exception as e:
            messages.error(request, f"Error recording weigh-in: {str(e)}")
            return redirect('weigh_in')
            
    # GET request - load user current weight for display
    try:
        user_profile = db.child("users").child(uid).get().val() or {}
        current_weight = user_profile.get('weight', '')
    except Exception:
        current_weight = ''
        
    return render(request, 'weigh_in.html', {'user': user, 'current_weight': current_weight})