from django.urls import path
from api.controllers.file import UploadFile, LoadTable

urlpatterns = [
    path('load-file/', UploadFile.as_view()),
    path('get-table/', LoadTable.as_view()),
]
