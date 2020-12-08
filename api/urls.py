from django.urls import path
from api.controllers.file import UploadFile

urlpatterns = [
    path('load-file/', UploadFile.as_view())
]
