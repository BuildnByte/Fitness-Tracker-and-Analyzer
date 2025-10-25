import firebase_admin
import pandas as pd
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
from collections import Counter

# ============================================================================
# FIREBASE CONFIGURATION
# ============================================================================

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

def get_health_records(uid, start_date=None, end_date=None):
    """Get health records from Firebase with optional date filtering"""
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
    records = get_health_records(uid, start_date, end_date)
    
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
    def score_metric(actual, target, tolerance=0.2, invert=False):
        if actual == 0:
            return 0
        ratio = actual / target
        if invert:  # For metrics where lower is better (junk food)
            return 100 if actual <= target else max(0, 100 - (actual - target) * 25)
        # Within tolerance is 100%, penalize deviations
        if (1 - tolerance) <= ratio <= (1 + tolerance):
            return min(100, ratio * 100)
        return max(0, 100 - abs(actual - target) * 20)
    
    scores = {
        'sleep': score_metric(current_stats.get('avg_sleep', 0), targets.get('sleep_hours', 7.5)),
        'sleep_quality': score_metric(current_stats.get('avg_sleep_quality', 0), targets.get('sleep_quality', 3.5)),
        'workout_consistency': score_metric(current_stats.get('active_days', 0), targets.get('active_days', 4)),
        'workout_duration': score_metric(current_stats.get('avg_workout_duration', 0), targets.get('workout_duration', 30)),
        'hydration': score_metric(current_stats.get('avg_water', 0), targets.get('water_intake', 2.5)),
        'diet_quality': score_metric(current_stats.get('avg_junk_food', 0), targets.get('junk_food_level', 2.0), invert=True)
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

def load_diet_model():
    """Load trained diet plan model"""
    try:
        model_path = os.path.join(settings.BASE_DIR, 'diet_plan_model.joblib')
        if not os.path.exists(model_path):
            return None
        return load(model_path)
    except Exception as e:
        print(f"Error loading diet model: {str(e)}")
        return None

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
        "headline": config['headline']
    }

def predict_weekly_plans(uid):
    """Predict both diet and workout plans"""
    diet_model = load_diet_model()
    workout_model, class_to_plan = load_workout_model()
    
    if not diet_model or not workout_model:
        return None
    
    weekly_stats = get_weekly_stats(uid)
    if not weekly_stats or weekly_stats['total_days'] < 3:
        return None
    
    user_profile = get_user_profile(uid)
    goal = user_profile.get("goal", "general")
    
    # Diet plan prediction
    diet_df = pd.DataFrame([{
        "Week Avg Sleep": weekly_stats["avg_sleep"],
        "Avg Calories": weekly_stats["avg_calories"],
        "Protein": 100,  # Default if not tracked
        "Carbs": 200,
        "Water (L)": weekly_stats["avg_water"],
        "Goal": goal
    }])
    
    plan_style = diet_model.predict(diet_df)[0]
    diet_plan = generate_personalized_diet(
        weekly_stats["avg_sleep"], weekly_stats["avg_calories"],
        100, 200, weekly_stats["avg_water"], goal, plan_style
    )
    
    # Workout plan prediction
    workout_history = (f"{weekly_stats['active_days']} active days, "
                      f"avg {weekly_stats['avg_workout_duration']} min, "
                      f"most popular: {weekly_stats['most_popular_workout']}")
    
    workout_df = pd.DataFrame([{
        "avg_sleep_hours": weekly_stats["avg_sleep"],
        "sleep_quality": weekly_stats["avg_sleep_quality"],
        "avg_calories": weekly_stats["avg_calories"],
        "protein": 100,
        "carbs": 200,
        "fat": 70,
        "water_liters": weekly_stats["avg_water"],
        "goal": goal,
        "workout_history": workout_history
    }])
    
    workout_label = workout_model.predict(workout_df)[0]
    workout_plan = class_to_plan.get(workout_label, {})
    
    return {
        "diet_plan": diet_plan,
        "workout_plan": {
            "predicted_label": workout_label,
            "weekly_schedule": workout_plan
        }
    }

# ============================================================================
# CHART DATA PREPARATION
# ============================================================================

def prepare_chart_data(records_dict, start_date, end_date):
    """Prepare data for dashboard charts"""
    date_range = [start_date + timedelta(days=i) for i in range(7)]
    dates = [d.strftime('%m/%d') for d in date_range]
    
    data_arrays = {
        'sleep_hours': [], 'sleep_quality': [], 'calories': [],
        'water_intake': [], 'workout_duration': [], 'junk_food_levels': []
    }
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        if records_dict and date_str in records_dict:
            record = records_dict[date_str]
            data_arrays['sleep_hours'].append(safe_float(record.get('sleep_hours', 0)))
            data_arrays['sleep_quality'].append(safe_int(record.get('sleep_quality', 0)))
            data_arrays['calories'].append(safe_int(record.get('total_calories', 0)))
            data_arrays['water_intake'].append(safe_float(record.get('water_intake', 0)))
            data_arrays['workout_duration'].append(safe_int(record.get('workout_duration', 0)))
            data_arrays['junk_food_levels'].append(safe_int(record.get('junk_food_level', 0)))
        else:
            for key in data_arrays:
                data_arrays[key].append(None)
    
    # Meal distribution from latest record
    meal_data = [0, 0, 0, 0]
    if records_dict:
        latest = records_dict[max(records_dict.keys())]
        meal_data = [
            safe_int(latest.get('breakfast_calories', 0)),
            safe_int(latest.get('lunch_calories', 0)),
            safe_int(latest.get('dinner_calories', 0)),
            safe_int(latest.get('snacks_calories', 0))
        ]
    
    # Workout types distribution
    workout_types_count = {}
    if records_dict:
        for record in records_dict.values():
            types = record.get('workout_types', [])
            if isinstance(types, str):
                types = json.loads(types) if types else []
            for wtype in types:
                if wtype:
                    workout_types_count[wtype] = workout_types_count.get(wtype, 0) + 1
    
    return {
        'dates': dates,
        **data_arrays,
        'meal_data': meal_data,
        'meal_labels': ['Breakfast', 'Lunch', 'Dinner', 'Snacks'],
        'workout_types_count': workout_types_count
    }

# ============================================================================
# VIEW FUNCTIONS
# ============================================================================

def login_view(request):
    """User login"""
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = auth.sign_in_with_email_and_password(email, password)
            uid = user['localId']
            user_data = db.child("users").child(uid).get().val()

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
            except:
                error_detail = str(e)
            messages.error(request, f"Login failed: {error_detail}")
    
    return render(request, 'login.html')

def signup_view_with_baseline(request):
    """User signup with baseline setup"""
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        goal = request.POST.get('goal')

        if not all([name, email, password, goal]):
            messages.error(request, "All fields are required.")
            return render(request, 'signup.html')

        try:
            user = auth.create_user_with_email_and_password(email, password)
            uid = user['localId']

            # Store user profile
            data = {
                "name": name,
                "email": email,
                "goal": goal,
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
    """User logout"""
    request.session.flush()
    messages.success(request, "Logged out successfully!")
    return redirect('login')

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
        start_date, end_date = get_week_dates()
        recent_records = get_health_records(uid, start_date, end_date)
        chart_data = prepare_chart_data(recent_records, start_date, end_date)

        # Weekly plan
        combined_plan = get_current_weekly_plan(uid)
        
        print(combined_plan)
        print(weekly_stats)

        # Progress tracking
        progress_data = None
        progress_trend = None
        
        if weekly_stats:
            baseline = get_user_baseline(uid)
            if baseline:
                progress_data = calculate_progress_score(weekly_stats, baseline)
                if progress_data:
                    save_progress_history(uid, progress_data)
                    progress_trend = get_progress_trend(uid, weeks=4)
        
        # Prepare plan info
        plan_week_info = None
        if combined_plan:
            from django.utils.dateparse import parse_datetime
            from django.utils.timezone import localtime
            generated_at = combined_plan.get('generated_at')
            plan_week_info = {
                'start_date': combined_plan.get('week_start_date'),
                'end_date': combined_plan.get('week_end_date'),
                'generated_at': localtime(parse_datetime(generated_at)).strftime("%Y-%m-%d %H:%M:%S") if generated_at else None
            }
        
        context = {
            'user': user,
            'user_profile': user_profile,
            'has_today_record': has_today_record,
            'weekly_stats': weekly_stats,
            'chart_data': json.dumps(chart_data, default=str),
            'current_streak': get_current_streak(uid),
            'total_records': len(get_health_records(uid)),
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
        
        # Prepare health record
        health_record_data = {
            'date': date_str,
            'created_at': timezone.now().isoformat(),
            'user_id': uid,
            
            # Sleep
            'sleep_hours': safe_float(data['sleepHours']),
            'sleep_quality': safe_int(data['sleepQuality']),
            'bedtime': str(data.get('bedtime', '')),
            'wake_time': str(data.get('waketime', '')),
            
            # Nutrition
            'total_calories': safe_int(data['totalCalories']),
            'water_intake': safe_float(data['waterIntake']),
            'carbs': safe_int(data.get('carbs', 0)),
            'protein': safe_int(data.get('protein', 0)),
            'fat': safe_int(data.get('fat', 0)),
            
            # Meals
            'breakfast_calories': safe_int(data.get('breakfast', 0)),
            'lunch_calories': safe_int(data.get('lunch', 0)),
            'dinner_calories': safe_int(data.get('dinner', 0)),
            'snacks_calories': safe_int(data.get('snacks', 0)),
            
            'junk_food_level': safe_int(data['junkFood']),
            
            # Workout
            'workout_duration': safe_int(data['workoutDuration']),
            'workout_intensity': str(data['workoutIntensity']),
            'workout_types': workout_types,
            'calories_burned': safe_int(data.get('caloriesBurned', 0)),
        }
        
        # Validation
        validations = [
            (0 <= health_record_data['sleep_hours'] <= 24, 'Sleep hours must be 0-24'),
            (1 <= health_record_data['sleep_quality'] <= 5, 'Sleep quality must be 1-5'),
            (health_record_data['total_calories'] >= 0, 'Calories cannot be negative'),
            (health_record_data['water_intake'] >= 0, 'Water intake cannot be negative'),
            (0 <= health_record_data['junk_food_level'] <= 4, 'Junk food level must be 0-4'),
            (health_record_data['workout_duration'] >= 0, 'Workout duration cannot be negative'),
            (health_record_data['workout_intensity'] in ['low', 'medium', 'high'], 'Invalid workout intensity')
        ]
        
        for condition, error_msg in validations:
            if not condition:
                return JsonResponse({'success': False, 'error': error_msg}, status=400)
        
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
        print(f"Error in diet_plan_view: {str(e)}")
        messages.error(request, "Error loading diet plan.")
        return redirect('dashboard')

def workout_plan_view_updated(request):
    """Display saved weekly workout plan"""
    user = request.session.get('user')
    if not user:
        return redirect('login')

    uid = user['uid']
    
    try:
        weekly_plan = get_current_weekly_plan(uid)
        
        if not weekly_plan:
            messages.warning(request, "No weekly plan available. Plans are generated every Sunday night.")
            return redirect('dashboard')
        
        context = {
            'user': user,
            'workout_plan': weekly_plan.get('workout_plan'),
            'plan_info': {
                'week_start': weekly_plan.get('week_start_date'),
                'week_end': weekly_plan.get('week_end_date'),
                'generated_at': weekly_plan.get('generated_at')
            }
        }
        
        return render(request, "workout_plan.html", context)
        
    except Exception as e:
        print(f"Error in workout_plan_view: {str(e)}")
        messages.error(request, "Error loading workout plan.")
        return redirect('dashboard')