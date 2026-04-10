from django.urls import path
from . import views

urlpatterns = [
    path('', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('input/', views.input_form_view, name='input_form'),
    path('result/', views.result_view, name='result'),
    path('logout/', views.logout_view, name='logout'),
]
