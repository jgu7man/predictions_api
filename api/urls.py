from django.urls import path
from api.controllers.file import UploadFile
from api.controllers.tables import LoadTable
from api.controllers.products import FilterProduct, MonthSalesDetails

urlpatterns = [
    path('load-file/', UploadFile.as_view()),
    path('table', LoadTable.as_view()),
    path('product/filter', FilterProduct.as_view()),
    path('product/month-details', MonthSalesDetails.as_view()),
]
