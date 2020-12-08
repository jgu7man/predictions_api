from rest_framework.views import APIView
from rest_framework.response import Response
# from api.controllers.file import hola
# Create your views here.

class ImportFile(APIView):
    def get(self, request, format=None):
        # result = hola()
        
        return Response({
            'message': 'Archivo cargado', 
            # 'response': result, 
            })