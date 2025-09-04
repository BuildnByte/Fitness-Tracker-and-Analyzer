from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('form/', views.form_view, name='form'),  # Changed from 'formview' to 'form'
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('submit-health-data/', views.submit_health_data, name='submit_health_data'),
    path('api/dashboard-data/', views.get_dashboard_data, name='dashboard_data'),
    path("workout-plan/", views.workout_plan_view, name="workout_plan"),
    path('diet-plan/', views.diet_plan_view, name='diet_plan'),
]