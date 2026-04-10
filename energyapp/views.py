from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib import messages
import pickle
import numpy as np
import os
from django.conf import settings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------- Login View ----------------
def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('input_form')
        else:
            return render(request, 'login.html', {'error': 'Invalid username or password'})
    return render(request, 'login.html')

# ---------------- Register View ----------------
def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')

        if password1 != password2:
            return render(request, 'register.html', {'error': 'Passwords do not match'})
        elif User.objects.filter(username=username).exists():
            return render(request, 'register.html', {'error': 'Username already taken'})
        else:
            user = User.objects.create_user(username=username, password=password1)
            user.save()
            messages.success(request, "Account created successfully!")
            return redirect('login')

    return render(request, 'register.html')

# ---------------- Input Form View ----------------
def input_form_view(request):
    appliances = ['AC', 'fan', 'fridge', 'washing_machine', 'TV', 'light', 'wifi', 'other']

    if request.method == 'POST':
        temp = float(request.POST.get('temperature'))
        humidity = float(request.POST.get('humidity'))
        persons = int(request.POST.get('persons'))
        target = float(request.POST.get('monthly_target'))

        total_usage = 0
        appliance_wise_usage = {}

        for app in appliances:
            if request.POST.get(app):  # if checkbox selected
                hours = float(request.POST.get(f'{app}_usage') or 0)
                days = int(request.POST.get(f'{app}_days') or 0)
                count = int(request.POST.get(f'{app}_count') or 0)

                usage = count * hours * days
                appliance_wise_usage[app] = usage
                total_usage += usage

        # Load ML model and scaler
        model_path = os.path.join(settings.BASE_DIR, 'energy_app/ml/energy_model.pkl')
        scaler_path = os.path.join(settings.BASE_DIR, 'energy_app/ml/scaler.pkl')

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        # Prepare input features for prediction
        input_data = [temp, humidity, persons, target]
        for app in appliances:
            count = int(request.POST.get(f'{app}_count') or 0)
            hours = float(request.POST.get(f'{app}_usage') or 0)
            days = int(request.POST.get(f'{app}_days') or 0)
            input_data.extend([count, hours, days])

        features = np.array(input_data).reshape(1, -1)
        scaled_features = scaler.transform(features)
        prediction = round(model.predict(scaled_features)[0], 2)

        # Store prediction and input info in session
        request.session['prediction'] = prediction
        request.session['target'] = target
        request.session['appliance_data'] = appliance_wise_usage

        return redirect('result')

    return render(request, 'input_form.html', {'appliance_list': appliances})

# ---------------- Result View ----------------
def result_view(request):
    prediction = request.session.get('prediction')
    target = request.session.get('target')
    appliance_data = request.session.get('appliance_data')

    # Generate bar chart
    fig1, ax1 = plt.subplots()
    ax1.bar(['Prediction', 'Target'], [abs(prediction), target], color=['orange', 'green'])
    ax1.set_ylabel('Energy (kWh)')
    ax1.set_title('Energy Usage Comparison')
    bar_path = os.path.join(settings.BASE_DIR, 'energy_app/static/images/bar_chart.png')
    plt.savefig(bar_path)
    plt.close()

    # Generate pie chart
    if appliance_data:
        labels = list(appliance_data.keys())
        sizes = list(appliance_data.values())
        fig2, ax2 = plt.subplots()
        ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax2.axis('equal')
        pie_path = os.path.join(settings.BASE_DIR, 'energy_app/static/images/pie_chart.png')
        plt.savefig(pie_path)
        plt.close()

    return render(request, 'result.html', {
        'prediction': abs(prediction),
        'target': target,
    })

# ---------------- Logout View ----------------
def logout_view(request):
    logout(request)
    return redirect('login')
