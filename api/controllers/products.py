from api.firebase_app import FirebaseApp
from IPython.display import display
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json