from django.urls import path
from api.controllers.file import UploadFile
from api.controllers.tables import LoadTable
from api.controllers.products import FilterProduct
from api.controllers.predictions import MonthSalesPrediction, ReverseSalesPredictions, AnalyzeProvOffering, EstimatedPrediction, ARIMAprediction

urlpatterns = [
    path('load-file', UploadFile.as_view()),
    path('table', LoadTable.as_view()),
    path('product/filter', FilterProduct.as_view()),
    path('predictions/cant-predict', MonthSalesPrediction.as_view()),
    path('predictions/month-predict', ReverseSalesPredictions.as_view()),
    path('predictions/provider-offering', AnalyzeProvOffering.as_view()),
    path('predictions/estimated', EstimatedPrediction.as_view()),
    path('predictions/arima', ARIMAprediction.as_view()),
]
