from api.firebase_app import FirebaseApp
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from api.controllers.file import upload_file

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib  as mlp
import numpy as np
import locale
import datetime
import json

db = FirebaseApp.fs
st = FirebaseApp.st
mlp.style.use('seaborn')

class FilterProduct(APIView):
    def get(self, request, format=None):
        
        # VALIDATE THERE IS TABLE ID
        try:
            table_id = request.query_params['table']
            doc_ref = db.collection(u'tables').document(table_id)
        except: return Response({
                'message': 'La petición debe incluir le parámetro "table"',
                'status':500
            })
        
        # VALIDATE IS PRODUCT ID     
        try: product_id = request.query_params['product']
        except: return Response({
                'message': 'La petición debe incluir el parámetro "product"',
                'status':500
            })

        # VALIDATE IS DOCUMENT IN FIRESTORE
        try:
            doc = doc_ref.get()
            doc_URL = doc.to_dict()['fileURL']  
        except: return Response({
                'message': 'El archivo no exite o fue eliminado', 
                'status': 500
            })
            
        # READ DOCUMENT
        df = pd.read_csv(doc_URL,  decimal=".")  
        df['fecha'] = pd.to_datetime(df['fecha'])
        
        # VALIDATE IS PRODUCT IN LIST
        try:product_selected = df.loc[df['codigo'] == product_id]
        except: return Response({
            'message': 'Error al intentar elegir el producto en esta tabla',
            'status':500
        })
        
        
        # GET STATS
        product_stats = get_product_stats(product_selected)
        sales_timeline = get_sales_timeline(product_selected)
        get_sales_chart(sales_timeline)
        
        
        # DEFINE PATHS
        product_name = product_selected['descripcion'].unique()[0].strip()
        product_refname = product_name.replace(' ', '_').lower()
        product_path = 'tables/'+table_id+'/product/'+product_refname+'/'
        local_path = 'api/uploads/'
        
        # STORAGE FILES
        product_selected.to_json(local_path+'dataset.json', orient="columns")
        datasetURL = upload_file(local_path, product_path,'dataset.json')
        default_storage.delete(local_path+'dataset.json')
        
        timelineURL = upload_file(local_path, product_path, 'sales-timeline.json')
        default_storage.delete(local_path+'sales-timeline.json')
        
        databyperiodsURL = upload_file(local_path, product_path, 'sales_by_periods.json')
        default_storage.delete(local_path+'sales_by_periods.json')
        saleschartURL = upload_file(local_path, product_path, 'sales-chart.jpg')
        default_storage.delete(local_path+'sales-chart.jpg')
        
        product_stats['files'] = {
            'dataset': datasetURL,
            'timeline': timelineURL,
            'data_byperiods': databyperiodsURL,
            'chart': saleschartURL
        }
        
        product_ref = doc_ref.collection('products').document(product_id)
        product_ref.set({
            "name": product_name,
            "code": product_id,
            "stats": product_stats,
        })
        
        return Response({
            'status': 200,
            'message':'ok',
            'result': {
                "name": product_name,
                "code": product_id,
                "stats": product_stats
            }
        })
        
        


class MonthSalesDetails(APIView):
    def get(self, request, format=None):
        # VALIDATE THERE IS TABLE ID
        print('Request ok')
        try:
            table_id = request.query_params['table']
        except: return Response({
                'message': 'La petición debe incluir le parámetro "table"',
                'status':500
            })
        
        print('table ok', table_id)
        # VALIDATE IS PRODUCT ID     
        try: product_id = request.query_params['product']
        except: return Response({
                'message': 'La petición debe incluir el parámetro "product"',
                'status':500
            })
    
        print('prodid ok', product_id)
        doc_ref = db.document('tables/'+table_id+'/products/'+product_id)
        try: doc = doc_ref.get()
        except: return Response({
            'status':404,
            'message':'No se encontró el documento en la base de datos'
        })
        try: files = doc.to_dict()['stats']['files']
        except: return Response({
            'status':404,
            'message':'No se encontró la lusta de archivos en la base de datos'
        })
        
        print('firestore ok')
        try: doc_URL = files['timeline']
        except: return Response({
            'message':'El archivo "timeline.json" no exite o fue eliminado. Intenta realizar un filtrado de producto nuevamente',
            'status': 404
        })
        
        print('urls ok')
        product_name = doc.to_dict()['name']
        dataset = pd.read_json(doc_URL) 
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        
        # GROUP DATA BY SALES IN DATE 
        # display(dataset.head())
        df_dates = dataset.groupby(dataset.index, as_index=True).aggregate({ 'unidades': 'sum' })
        df_dates.to_csv('api/uploads/sales-dates.csv', header=False)
        
        # GROUP DATA BY MONTHS
        product_dataset = dataset['unidades']
        months = product_dataset.groupby(pd.Grouper(freq='M'))
        
        # GET SOME DATA
        indexes = []
        datas = []
        lengths = []
        sales = []
        for name, group in months:
            groupSales = []
            for value in group.values:
                if value != 0:
                    groupSales.append(value)
            sales.append(len(groupSales))
            datas.append(group.values)
            lengths.append(len(group.values))
            indexes.append(name.strftime('%B'))

        maxlength = max(lengths)
        transactions = np.sum(sales)
        avg_mes = transactions/len(months)
        
        # NORMALIZE DATAFRAME
        normalized_df = pd.DataFrame()
        for mes, (name, group) in zip(indexes, months):
            values = group.to_list()
            less = maxlength - len(group.values) 
            
            for zero in range(less):
                values.append(zero * 0)

            normalized_df[mes] = pd.Series(values)
            
        
        normalized_df.to_json('api/uploads/normalized_sells.json', orient="columns")
        months_box = normalized_df.replace(0,np.nan )
        months_box.plot.box(figsize=(8,5))
        plt.savefig('api/uploads/month_sales_normalized.jpg')
        
        # STORAGE FILE
        product_refname = product_name.replace(' ', '_').lower()
        cloud_path = 'tables/'+table_id+'/product/'+product_refname+'/'
        salesdatesURL = upload_file('api/uploads/', cloud_path, 'sales-dates.csv')
        normalizedsellsURL = upload_file('api/uploads/', cloud_path, 'normalized_sells.json')
        monthsaleschartURL = upload_file('api/uploads/', cloud_path, 'month_sales_normalized.jpg')
        default_storage.delete('api/uploads/month_sales_normalized.jpg')
        
        # CREATE RESULT
        result = {
            u"max_monthsales": int(maxlength),
            u"avgsales_per_month": float("{:.2f}".format( avg_mes)),
            u"month_sales_chart": monthsaleschartURL,
            u"normalized_sells": normalizedsellsURL,
            u"sales_dates": salesdatesURL    
        }
        
        # UPDATE FIRESTORE 
        doc_ref.update({ "month_details":result })
        
        return Response({
            "status":200,
            "message":"ok",
            "result":result
        })
        
    
def get_product_stats(dataset):
    
    # CALCULATE PROMEDIATES
    sales_quantity = dataset['unidades'].describe()['count']
    avg_sell_price = dataset['precio_unit_venta'].describe()['mean']
    max_sell_price = dataset['precio_unit_venta'].describe()['max']
    avg_margin =     dataset['por_margen'].describe()['mean']
    max_margin =      dataset['por_margen'].describe()['max']
    avg_purch_price =  dataset['precio_unit_costo'].describe()['mean']
    min_purch_price =   dataset['precio_unit_costo'].describe()['min']
    
    
    # CALCULATE SALES DAYS
    first = dataset.iloc[0]['fecha']
    last = dataset.iloc[-1]['fecha']
    periodo_ventas = (last-first).days

    sold_units = dataset['unidades'].sum()
    sold_units = int(sold_units)

    stats = {
        "avgs": {
            "sold_units": int(sold_units),
            "sales_quantity":int(sales_quantity),
            "avg_sell_price":int(avg_sell_price),
            "max_sell_price":int(max_sell_price),
            "avg_margin":int(avg_margin * 100),
            "max_margin":int(max_margin * 100),
            "avg_purch_price":int(avg_purch_price),
            "min_purch_price":int(min_purch_price),
        },
        "time_data":{
            "period_in_days": int(periodo_ventas),
            "first_sale_date": first,
            "last_sale_date": last,
        },  
    
    }

    # print('se vendieron', cantidad_vendida, product_name,'en', periodo_ventas, 'días')
    return stats
    
    
def get_sales_timeline(dataset):
    sales_timeline = dataset.groupby(dataset['fecha'], as_index=True).aggregate({
        'unidades': 'sum', 
        'total_venta': 'sum',
        'total_costo': 'sum',
    })
    
    sales_timeline.to_json('api/uploads/sales-timeline.json', orient="columns")
    return sales_timeline
    

def get_sales_chart(dataset):
    df_periods = dataset.groupby(pd.Grouper(freq='W')).aggregate({
        'unidades': 'sum',
        'total_venta': 'sum',
        'total_costo': 'sum',
        })
    df_periods.to_json('api/uploads/sales_by_periods.json', orient="columns")


    df_periods['total_venta'].plot(figsize = (14,6), lw=2, title='ventas vs costos')
    df_periods['total_costo'].plot(figsize = (14,6), lw=2)
    
    plt.savefig('api/uploads/sales-chart.jpg')
    

    