from django.urls import include, path
from rest_framework import routers

from .views import ImageViewSet

router = routers.DefaultRouter()
router.register(r'images', ImageViewSet)

app_name = 'images'
urlpatterns = [
    path('', include(router.urls))
]