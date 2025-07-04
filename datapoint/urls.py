from django.urls import path
from .views import generate_data, generate_smart_data

urlpatterns = [
    path('generate/', generate_data),
    path('generate/smart/', generate_smart_data),
]
