import firebase_admin
import pandas as pd
from firebase_admin import credentials
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
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
from datetime import datetime, date, timedelta






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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
firebase_file_path = os.path.join(BASE_DIR, 'firebase_config.json')

if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_file_path)
    firebase_admin.initialize_app(cred)

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

def get_user_profile_from_firebase(uid):
    """Helper function to get user profile from Firebase"""
    try:
        user_data = db.child("users").child(uid).get().val()
        return user_data if user_data else {}
    except:
        return {}
    

def load_workout_model():
    """Load the trained workout plan model"""
    try:
        model_path = os.path.join(settings.BASE_DIR, 'workout_plan_model.joblib')
        mapping_path = os.path.join(settings.BASE_DIR, 'class_to_plan.json')
        
        if not os.path.exists(model_path) or not os.path.exists(mapping_path):
            print("Workout model or mapping not found")
            return None, None
        
        model = load(model_path)
        with open(mapping_path, "r") as f:
            class_to_plan = json.load(f)
        print("Workout model loaded successfully")
        return model, class_to_plan
    except Exception as e:
        print(f"Error loading workout model: {str(e)}")
        return None, None

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
        
        if start_date and end_date:
            # Get all records first, then filter (Firebase query limitations)
            all_records = records_ref.get().val()
            if not all_records:
                return {}
            
            # Filter by date range
            filtered_records = {}
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            for date_key, record_data in all_records.items():
                if start_str <= date_key <= end_str:
                    filtered_records[date_key] = record_data
            
            print(f"Filtered {len(filtered_records)} records from {len(all_records)} total records")
            return filtered_records
        else:
            records_data = records_ref.get().val()
            return records_data if records_data else {}
            
    except Exception as e:
        print(f"Error fetching health records: {str(e)}")
        return {}

def has_record_for_date(uid, check_date):
    """Check if user has a health record for a specific date with better error handling"""
    date_str = check_date.strftime('%Y-%m-%d')
    try:
        record = db.child("health_records").child(uid).child(date_str).get().val()
        exists = record is not None
        print(f"Record exists for {date_str}: {exists}")
        return exists
    except Exception as e:
        print(f"Error checking record for date {date_str}: {str(e)}")
        return False

def get_weekly_stats(uid):
    """Calculate comprehensive weekly statistics from Firebase data"""
    end_date = timezone.now().date()
    start_date = end_date - timedelta(days=6)
    
    records = get_health_records_from_firebase(uid, start_date, end_date)
    
    if not records:
        return None
    
    # Initialize totals
    total_sleep = 0
    total_sleep_quality = 0
    total_calories = 0
    total_water = 0
    total_workout_duration = 0
    total_junk_food = 0
    active_days = 0
    workout_types = []
    
    record_count = len(records)
    
    # Process each record
    for record_data in records.values():
        # Sleep data
        sleep_hours = float(record_data.get('sleep_hours', 0))
        sleep_quality = int(record_data.get('sleep_quality', 0))
        total_sleep += sleep_hours
        total_sleep_quality += sleep_quality
        
        # Nutrition data
        calories = int(record_data.get('total_calories', 0))
        water = float(record_data.get('water_intake', 0))
        junk_food = int(record_data.get('junk_food_level', 0))
        total_calories += calories
        total_water += water
        total_junk_food += junk_food
        
        # Workout data
        workout_duration = int(record_data.get('workout_duration', 0))
        total_workout_duration += workout_duration
        
        if workout_duration > 0:
            active_days += 1
        
        # Collect workout types
        record_workout_types = record_data.get('workout_types', [])
        if isinstance(record_workout_types, str):
            try:
                record_workout_types = json.loads(record_workout_types)
            except:
                record_workout_types = [record_workout_types] if record_workout_types else []
        elif not isinstance(record_workout_types, list):
            record_workout_types = []
        
        workout_types.extend(record_workout_types)
    
    # Find most popular workout
    most_popular_workout = 'none'
    if workout_types:
        from collections import Counter
        workout_counts = Counter(workout_types)
        most_popular_workout = workout_counts.most_common(1)[0][0]
    
    return {
        'avg_sleep': round(total_sleep / record_count, 1) if record_count > 0 else 0,
        'avg_sleep_quality': round(total_sleep_quality / record_count, 1) if record_count > 0 else 0,
        'avg_calories': round(total_calories / record_count) if record_count > 0 else 0,
        'avg_water': round(total_water / record_count, 1) if record_count > 0 else 0,
        'avg_workout_duration': round(total_workout_duration / record_count) if record_count > 0 else 0,
        'avg_junk_food': round(total_junk_food / record_count, 1) if record_count > 0 else 0,
        'total_days': record_count,
        'active_days': active_days,
        'most_popular_workout': most_popular_workout
    }

def get_current_streak(uid):
    """Calculate current streak from Firebase data with proper logic"""
    try:
        # Get all health records for the user
        records = db.child("health_records").child(uid).get().val()
        if not records:
            print(f"No records found for user {uid}")
            return 0
        
        # Convert record dates to date objects and sort them
        record_dates = []
        for date_str in records.keys():
            try:
                record_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                record_dates.append(record_date)
            except ValueError:
                print(f"Invalid date format: {date_str}")
                continue
        
        if not record_dates:
            print("No valid dates found in records")
            return 0
        
        # Sort dates in descending order (most recent first)
        record_dates.sort(reverse=True)
        
        print(f"Found {len(record_dates)} valid record dates for user {uid}")
        print(f"Most recent dates: {record_dates[:5]}")  # Show first 5 dates
        
        # Get today's date
        today = timezone.now().date()
        streak = 0
        
        # Check if there's a record for today or yesterday to start the streak
        most_recent_date = record_dates[0]
        
        # Calculate days difference from today
        days_diff = (today - most_recent_date).days
        
        print(f"Today: {today}, Most recent record: {most_recent_date}, Days diff: {days_diff}")
        
        # If the most recent record is more than 1 day old, streak is 0
        if days_diff > 1:
            print(f"Most recent record is {days_diff} days old, streak broken")
            return 0
        
        # Start checking for consecutive days
        expected_date = most_recent_date
        
        for record_date in record_dates:
            if record_date == expected_date:
                streak += 1
                expected_date = record_date - timedelta(days=1)
                print(f"Streak day {streak}: {record_date}, next expected: {expected_date}")
            else:
                # Gap found, break the streak
                print(f"Gap found: expected {expected_date}, found {record_date}")
                break
        
        print(f"Final streak for user {uid}: {streak}")
        return streak
        
    except Exception as e:
        print(f"Error calculating streak for user {uid}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return 0
    

def get_streak_with_grace_period(uid):
    """
    Alternative streak calculation with a grace period
    Allows for 1 day gap without breaking the streak
    """
    try:
        records = db.child("health_records").child(uid).get().val()
        if not records:
            return 0
        
        # Get all record dates
        record_dates = []
        for date_str in records.keys():
            try:
                record_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                record_dates.append(record_date)
            except ValueError:
                continue
        
        if not record_dates:
            return 0
        
        # Sort dates in descending order
        record_dates.sort(reverse=True)
        today = timezone.now().date()
        
        # Check if we should start counting from today or yesterday
        start_date = today
        if today not in record_dates:
            if (today - timedelta(days=1)) in record_dates:
                start_date = today - timedelta(days=1)
            else:
                return 0
        
        streak = 0
        current_date = start_date
        grace_used = False
        
        while True:
            if current_date in record_dates:
                streak += 1
                current_date -= timedelta(days=1)
                grace_used = False  # Reset grace period
            else:
                if not grace_used:
                    # Use grace period (allow 1 day gap)
                    grace_used = True
                    current_date -= timedelta(days=1)
                else:
                    # No more grace, break streak
                    break
        
        return streak
        
    except Exception as e:
        print(f"Error in grace period streak calculation: {str(e)}")
        return 0
    
def get_simple_streak(uid):
    """
    Simple streak calculation - counts consecutive days from most recent record
    """
    try:
        records = db.child("health_records").child(uid).get().val()
        if not records:
            return 0
        
        # Get all dates and sort them
        dates = []
        for date_str in records.keys():
            try:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                dates.append(date_obj)
            except ValueError:
                continue
        
        if not dates:
            return 0
        
        dates.sort(reverse=True)  # Most recent first
        today = timezone.now().date()
        
        # Start from the most recent record
        streak = 0
        current_date = dates[0]
        
        # If the most recent record is today or yesterday, start counting
        if (today - current_date).days <= 1:
            for i, date in enumerate(dates):
                if i == 0:
                    streak = 1
                    continue
                
                # Check if this date is consecutive to the previous one
                prev_date = dates[i-1]
                if (prev_date - date).days == 1:
                    streak += 1
                else:
                    break
        
        return streak
        
    except Exception as e:
        print(f"Error in simple streak calculation: {str(e)}")
        return 0

def get_total_records_count(uid):
    """Get total number of health records for user"""
    try:
        records = db.child("health_records").child(uid).get().val()
        return len(records) if records else 0
    except:
        return 0

def load_diet_model():
    """Load the trained diet plan model"""
    try:
        model_path = os.path.join(settings.BASE_DIR, 'diet_plan_model.joblib')
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}")
            return None
        model = load(model_path)
        print("Diet plan model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading diet plan model: {str(e)}")
        return None

def clamp(v, vmin, vmax):
    """Helper function to clamp values between min and max"""
    return max(vmin, min(v, vmax))

def generate_personalized_plan(week_sleep, avg_cal, protein, carbs, water_l, goal, plan_style):
    """
    Convert model output (plan_style) + current weekly stats into numeric targets & readable text.
    """
    # Baseline from user's data
    base_cal = float(avg_cal)
    base_pro = float(protein)
    base_carb = float(carbs)
    base_h2o = float(water_l)

    # Sleep nudges for recovery
    sleep_target = clamp(round(max(7.0, min(9.0, (week_sleep + 7.5) / 2)), 1), 6.5, 9.0)

    # Default targets
    kcal_target = base_cal
    pro_target = base_pro
    carb_target = base_carb
    h2o_target = max(2.2, min(4.0, base_h2o))

    # Helper functions
    def g_step(x, low, high): 
        return int(clamp(x, low, high))

    g = plan_style

    if g == "weight_loss":
        kcal_target = int(round(base_cal * 0.82))
        pro_target = g_step(base_pro + 20, 80, 180)
        carb_target = g_step(base_carb * 0.8, 120, 300)
        h2o_target = clamp(base_h2o + 0.4, 2.2, 4.0)
        headline = "Calorie deficit with higher protein"
        bullets = [
            f"Calories: ~{kcal_target} kcal/day (≈18% below current average)",
            f"Protein: {pro_target} g/day to preserve lean mass",
            f"Carbs: {carb_target} g/day (favor whole grains, veggies)",
            f"Water: {round(h2o_target,1)} L/day",
            f"Sleep: {sleep_target} h/night for appetite & recovery",
        ]

    elif g == "muscle_gain":
        kcal_target = int(round(base_cal * 1.12))
        pro_target = g_step(max(base_pro, 110), 110, 200)
        carb_target = g_step(max(base_carb, 260), 220, 420)
        h2o_target = clamp(base_h2o + 0.3, 2.4, 4.0)
        headline = "Slight surplus with high protein"
        bullets = [
            f"Calories: ~{kcal_target} kcal/day (≈12% above current average)",
            f"Protein: {pro_target} g/day to support hypertrophy",
            f"Carbs: {carb_target} g/day for training fuel",
            f"Water: {round(h2o_target,1)} L/day",
            f"Sleep: {sleep_target} h/night (growth & recovery)",
        ]

    elif g == "endurance":
        kcal_target = int(round(clamp(base_cal, 2000, 2700)))
        pro_target = g_step(clamp(base_pro, 70, 120), 70, 130)
        carb_target = g_step(clamp(max(base_carb, 280), 280, 460), 260, 480)
        h2o_target = clamp(base_h2o + 0.5, 2.5, 4.0)
        headline = "Carb-focused fueling for endurance"
        bullets = [
            f"Calories: ~{kcal_target} kcal/day to support volume",
            f"Protein: {pro_target} g/day for repair",
            f"Carbs: {carb_target} g/day (prioritize pre/during/post training)",
            f"Water: {round(h2o_target,1)} L/day (+ electrolytes on long sessions)",
            f"Sleep: {sleep_target} h/night (key for aerobic adaptations)",
        ]

    else:  # "general"
        kcal_target = int(round(base_cal))
        pro_target = g_step(max(base_pro, 90), 80, 160)
        carb_target = g_step(clamp(base_carb, 180, 360), 180, 360)
        h2o_target = clamp(max(base_h2o, 2.4), 2.4, 4.0)
        headline = "Balanced maintenance"
        bullets = [
            f"Calories: ~{kcal_target} kcal/day (maintain)",
            f"Protein: {pro_target} g/day",
            f"Carbs: {carb_target} g/day (whole-food sources)",
            f"Water: {round(h2o_target,1)} L/day",
            f"Sleep: {sleep_target} h/night",
        ]

    description = f"{headline}. Focus on consistent meals, whole foods, and weekly check-ins to adjust."
    
    return {
        "plan_style": g,
        "targets": {
            "calories_kcal": int(kcal_target),
            "protein_g": int(pro_target),
            "carbs_g": int(carb_target),
            "water_l": float(round(h2o_target, 1)),
            "sleep_h": float(sleep_target),
        },
        "summary": description,
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
    Predict both weekly diet plan (macros + recommendations)
    and workout plan (7-day schedule) for the given user.
    """
    # --- Load models ---
    diet_model = load_diet_model()
    workout_model, class_to_plan = load_workout_model()

    if not diet_model or not workout_model:
        return None

    # --- Weekly stats ---
    weekly_stats = get_weekly_stats(uid)
    diet_weekly = get_weekly_averages_for_diet_plan(uid)
    if not weekly_stats or not diet_weekly:
        return None

    # --- User profile ---
    user_profile = get_user_profile_from_firebase(uid)
    goal = user_profile.get("goal", "general")

    # ------------------ DIET PLAN ------------------
    diet_df = pd.DataFrame([{
    "Week Avg Sleep": diet_weekly["week_avg_sleep"],
    "Avg Calories": diet_weekly["avg_calories"],
    "Protein": diet_weekly["protein"],
    "Carbs": diet_weekly["carbs"],
    "Water (L)": diet_weekly["water_l"],
    "Goal": goal
}])


    plan_style = diet_model.predict(diet_df)[0]
    diet_plan = generate_personalized_plan(
        week_sleep=diet_weekly["week_avg_sleep"],
        avg_cal=diet_weekly["avg_calories"],
        protein=diet_weekly["protein"],
        carbs=diet_weekly["carbs"],
        water_l=diet_weekly["water_l"],
        goal=goal,
        plan_style=plan_style
    )

    # ------------------ WORKOUT PLAN ------------------
    workout_input = get_weekly_averages_for_workout_plan(uid)
    if not workout_input:
        return None

    workout_df = pd.DataFrame([workout_input])
    workout_label = workout_model.predict(workout_df)[0]
    workout_plan = class_to_plan.get(workout_label, {})

    # ------------------ COMBINED OUTPUT ------------------
    return {
        "diet_plan": diet_plan,
        "workout_plan": {
            "predicted_label": workout_label,
            "weekly_schedule": workout_plan
        }
    }



# Views for Diet Plan Feature

def diet_plan_view(request):
    """View to display AI-generated diet plan"""
    user = request.session.get('user')
    if not user:
        messages.warning(request, "Please login to access diet plans.")
        return redirect('login')
    
    uid = user['uid']
    
    try:
        # Generate diet plan
        diet_plan = predict_weekly_health_and_fitness(uid)
        
        if not diet_plan:
            messages.warning(request, "Need at least 3 days of health data to generate a personalized diet plan.")
            return redirect('dashboard')
        
        context = {
            'user': user,
            'diet_plan': diet_plan,
            'current_streak': get_current_streak(uid),
        }
        
        return render(request, 'diet_plan.html', context)
        
    except Exception as e:
        print(f"Error in diet_plan_view: {str(e)}")
        messages.error(request, "Error generating diet plan. Please try again.")
        return redirect('dashboard')

def workout_plan_view(request):
    # Example: fetch the latest workout plan
    user = request.session.get('user')
    if not user:
        return redirect('login')

    uid = user['uid']
    plan = predict_weekly_health_and_fitness(uid)
    workout_plan = plan["workout_plan"] if plan else None

    return render(request, "workout_plan.html", {"workout_plan": workout_plan})


def dashboard_view(request):
    user = request.session.get('user')
    if not user:
        messages.warning(request, "Please login to access dashboard.")
        return redirect('login')
    
    uid = user['uid']
    
    try:
        # Get user profile from Firebase
        user_profile = get_user_profile_from_firebase(uid)
        
        # Get today's record status
        today = timezone.now().date()
        has_today_record = has_record_for_date(uid, today)
        
        # Get weekly stats
        weekly_stats = get_weekly_stats(uid)
        
        # Get recent records for charts (last 7 days)
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=6)
        
        print(f"Fetching records for user {uid} from {start_date} to {end_date}")
        recent_records = get_health_records_from_firebase(uid, start_date, end_date)
        print(f"Found {len(recent_records) if recent_records else 0} records")
        
        # Prepare chart data
        chart_data = prepare_chart_data(recent_records, start_date, end_date)

        # Get combined diet + workout plan (ML model prediction)
        combined_plan = None
        if weekly_stats:
            try:
                combined_plan = predict_weekly_health_and_fitness(uid)
            except Exception as ml_error:
                print(f"ML prediction failed: {ml_error}")
        
        print(f"Combined plan: {combined_plan}")

        context = {
            'user': user,
            'user_profile': user_profile,
            'has_today_record': has_today_record,
            'weekly_stats': weekly_stats,
            'chart_data': json.dumps(chart_data, default=str),  # Handle date serialization
            'current_streak': get_current_streak(uid),
            'total_records': get_total_records_count(uid),
            'combined_plan': combined_plan,  # NEW
        }
        
        print(f"Dashboard context prepared successfully for user {uid}")
        
    except Exception as e:
        print(f"Error in dashboard_view: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        context = {
            'user': user,
            'user_profile': {},
            'has_today_record': False,
            'weekly_stats': None,
            'chart_data': json.dumps({}),
            'current_streak': 0,
            'total_records': 0,
            'combined_plan': None  # keep safe fallback
        }
        messages.error(request, "Error loading dashboard data. Please try again.")
    
    return render(request, 'dashboard.html', context)


def form_view(request):
    user = request.session.get('user')
    if not user:
        messages.warning(request, "Please login to access this page.")
        return redirect('login')
    
    uid = user['uid']
    today = timezone.now().date()
    
    # Check if user already has a record for today
    if has_record_for_date(uid, today):
        messages.info(request, "You have already submitted your health data for today!")
        return redirect('dashboard')
    
    return render(request, 'weekly_health_form.html', {'user': user})

@csrf_exempt
@require_http_methods(["POST"])
def submit_health_data(request):
    user = request.session.get('user')
    if not user:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    uid = user['uid']
    today = timezone.now().date()
    date_str = today.strftime('%Y-%m-%d')
    
    # Check if user already has a record for today
    if has_record_for_date(uid, today):
        return JsonResponse({
            'success': False, 
            'error': 'You have already submitted data for today'
        }, status=400)
    
    try:
        # Parse JSON data
        try:
            data = json.loads(request.body)
            print(f"Received data for user {uid}: {data}")  # Debug log
        except json.JSONDecodeError as e:
            return JsonResponse({
                'success': False, 
                'error': 'Invalid JSON data received'
            }, status=400)
        
        # Validate required fields
        required_fields = {
            'sleepHours': 'Sleep Hours',
            'sleepQuality': 'Sleep Quality',
            'totalCalories': 'Total Calories',
            'waterIntake': 'Water Intake',
            'junkFood': 'Junk Food Level',
            'workoutDuration': 'Workout Duration',
            'workoutIntensity': 'Workout Intensity'
        }
        
        missing_fields = []
        for field, label in required_fields.items():
            if field not in data or data[field] == '' or data[field] is None:
                missing_fields.append(label)
        
        if missing_fields:
            return JsonResponse({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }, status=400)
        
        # Parse and validate workout types
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
        
        # Store in Firebase Realtime Database
        try:
            db.child("health_records").child(uid).child(date_str).set(health_record_data)
            print(f"Successfully saved health record for user {uid} on {date_str}")  # Debug log
            
            # Verify the data was saved
            saved_data = db.child("health_records").child(uid).child(date_str).get().val()
            if not saved_data:
                raise Exception("Data was not saved to Firebase")
                
        except Exception as e:
            print(f"Firebase error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Failed to save to database: {str(e)}'
            }, status=500)
        
        return JsonResponse({
            'success': True, 
            'message': 'Health data saved successfully!',
            'record_date': date_str,
            'data_summary': {
                'sleep_hours': health_record_data['sleep_hours'],
                'total_calories': health_record_data['total_calories'],
                'workout_duration': health_record_data['workout_duration'],
                'workout_types': len(workout_types)
            }
        })
        
    except Exception as e:
        print(f"Unexpected error in submit_health_data: {str(e)}")
        import traceback
        print(traceback.format_exc())  # Print full traceback for debugging
        
        return JsonResponse({
            'success': False, 
            'error': f'An unexpected error occurred: {str(e)}'
        }, status=500)

def get_dashboard_data(request):
    """API endpoint to get dashboard data for AJAX updates"""
    user = request.session.get('user')
    if not user:
        return JsonResponse({'success': False, 'error': 'Not authenticated'}, status=401)
    
    uid = user['uid']
    
    try:
        weekly_stats = get_weekly_stats(uid)
        
        # Get recent records for charts
        end_date = timezone.now().date()
        start_date = end_date - timedelta(days=6)
        
        recent_records = get_health_records_from_firebase(uid, start_date, end_date)
        
        chart_data = prepare_chart_data(recent_records, start_date, end_date)
        
        return JsonResponse({
            'success': True,
            'weekly_stats': weekly_stats,
            'chart_data': chart_data,
            'current_streak': get_current_streak(uid),
            'has_today_record': has_record_for_date(uid, timezone.now().date())
        })
        
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

        # Basic validation
        if not all([name, email, password, goal]):
            messages.error(request, "All fields are required.")
            return render(request, 'signup.html')

        try:
            # Register the user
            user = auth.create_user_with_email_and_password(email, password)
            uid = user['localId']

            # Store user profile in Realtime DB
            data = {
                "name": name,
                "email": email,
                "goal": goal,
                "created_at": timezone.now().isoformat()
            }
            db.child("users").child(uid).set(data)

            messages.success(request, "Registered successfully! Please login.")
            return redirect('login')
        except Exception as e:
            try:
                error_detail = json.loads(e.args[1])['error']['message']
            except (IndexError, KeyError, json.JSONDecodeError):
                error_detail = str(e)
            messages.error(request, f"Registration failed: {error_detail}")
    
    return render(request, 'signup.html')

def logout_view(request):
    """Logout function to clear session"""
    request.session.flush()  # This clears all session data
    messages.success(request, "Logged out successfully!")
    return redirect('login')