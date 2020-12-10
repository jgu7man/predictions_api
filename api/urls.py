from django.urls import path
from api.controllers.file import UploadFile
from api.controllers.tables import LoadTable
from api.controllers.products import FilterProduct, MonthSalesDetails
from api.controllers.predictions import EstimatedPrediction, ARIMAprediction

urlpatterns = [
    path('load-file/', UploadFile.as_view()),
    path('table', LoadTable.as_view()),
    path('product/filter', FilterProduct.as_view()),
    path('product/month-details', MonthSalesDetails.as_view()),
    path('predictions/estimated', EstimatedPrediction.as_view()),
    path('predictions/arima', ARIMAprediction.as_view()),
]
