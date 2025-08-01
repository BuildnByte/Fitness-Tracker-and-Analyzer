import firebase_admin
from firebase_admin import credentials
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import os
import pyrebase
from django.conf import settings



firebase_config={
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

            request.session['user'] = {
                'email': email,
                'name': user_data['name'],
                'goal': user_data['goal']
            }
            return redirect('dashboard')
        except Exception as e:
            import json
            error = json.loads(e.args[1])['error']['message']
            messages.error(request, f"Login failed: {error}")
    
    return render(request, 'login.html')



def dashboard_view(request):
    # user = request.session.get('user')
    # if not user:
    #     return redirect('login')
    return render(request, 'dashboard.html')

def form_view(request):
    # user = request.session.get('user')
    # if not user:
    #     return redirect('login')
    return render(request, 'weekly_health_form.html')

from django.shortcuts import render, redirect
from django.contrib import messages

def signup_view(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        password = request.POST.get('password')
        goal = request.POST.get('goal')

        try:
            # Register the user
            user = auth.create_user_with_email_and_password(email, password)
            uid = user['localId']

            # Store user profile in Realtime DB
            data = {
                "name": name,
                "email": email,
                "goal": goal
            }
            db.child("users").child(uid).set(data)

            messages.success(request, "Registered successfully!")
            return redirect('login')
        except Exception as e:
            error_json = e.args[1]
            import json
            error = json.loads(error_json)['error']['message']
            messages.error(request, f"Error: {error}")
    
    return render(request, 'signup.html')


