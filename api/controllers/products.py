# from IPython.core.display import display
from api.firebase_app import FirebaseApp
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from api.controllers.file import upload_file, upload_img

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib  as mlp
import numpy as np
import locale
import warnings
import json
import os
warnings.filterwarnings("ignore")


db = FirebaseApp.fs
st = FirebaseApp.st
mlp.style.use('seaborn')

class FilterProduct(APIView):
    def get(self, request, format=None):
        
        # VALIDATE THERE IS TABLE ID
        global table_id
        try:
            table_id = request.query_params['table']
            doc_ref = db.collection(u'tables').document(table_id)
        except: return Response({
                'message': 'La petición debe incluir le parámetro "table"',
                'status':400
            }, status=400)
        
        # VALIDATE IS PRODUCT ID     
        try: product_id = request.query_params['product']
        except: return Response({
                'message': 'La petición debe incluir el parámetro "product"',
                'status':400
            }, status=400)

        # VALIDATE IS DOCUMENT IN FIRESTORE
        try:
            doc = doc_ref.get()
            doc_URL = doc.to_dict()['fileURL']  
        except: return Response({
                'message': 'El archivo no exite o fue eliminado', 
                'status': 404
            }, status=404)
        
        
        
        
        
        
        # READ DOCUMENT
        df = pd.read_csv(doc_URL,  decimal=".", thousands=",")  
        df['Fecha'] = pd.to_datetime(df['Fecha'], format='%d/%m/%Y', errors='coerce')
        
        # VALIDATE IS PRODUCT IN LIST
        try:product_selected = df.loc[df['Codigo'] == product_id]
        except: return Response({
            'message': 'Error al intentar elegir el producto en esta tabla',
            'status':400
        }, status=400)
        
        # locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        
        print('petición realizada con éxito')
        # display(product_selected.head())
                    
        # DEFINE PATHS
        global product_name
        global product_path
        global local_path
        global current_directory
        product_name = product_selected['Descripcion'].unique()[0].strip()
        product_refname = product_name.replace(' ', '_').lower()
        product_path = 'tables/'+table_id+'/products/'+product_id+'/'
        current_directory = os.path.abspath(os.path.dirname(__file__))+'/'
        local_path = 'api/uploads/'
        

        # GET STATS
        product_stats = get_product_stats(product_selected)
        print('product_stats ok')
        sell_stats = get_sell_stats(product_selected)
        print('sell_stats ok')
        time_stats = get_sales_timeline(product_selected)
        print('time_stats ok')
        buy_stats = get_buy_stats(product_selected)
        print('buy_stats ok')
        


        # STORAGE FILES
        ps_file = product_selected.to_csv()
        datasetURL = upload_file(product_path,'product_dataset.csv', ps_file)
        print('dataset uploaded')
        
        
        product_ref = doc_ref.collection('products').document(product_id)
        product_ref.set({
            "name": product_name,
            "code": product_id,
            "dataset": datasetURL,
            "product_stats": product_stats,
            "buy_stats": buy_stats,
            "sell_stats": sell_stats,
            "time_stats": time_stats
        })
        print('firestore updated')
        
        
        return Response({
            'status': 200,
            'message':'ok',
            'result': {
                "name": product_name,
                "code": product_id,
                "dataset": datasetURL,
                "product_stats": product_stats,
                "buy_stats": buy_stats,
                "sell_stats": sell_stats,
                "time_stats": time_stats
            }
        }, status=200)
        
def get_product_stats(dataset):
    
    # CALCULATE PROMEDIATES
    # display(dataset['Unitario Venta'].describe())
    avg_margin =      dataset['PorMargen'].describe()['mean']
    avg_buy_price = dataset['Costo Unitario'].describe()['mean']
    avg_sale_price =  dataset['Unitario Venta'].describe()['mean']
    max_margin =      dataset['PorMargen'].describe()['max']
    max_sale_price =  dataset['Unitario Venta'].describe()['max']
    min_buy_price = dataset['Costo Unitario'].describe()['min']
    sales_quantity =  dataset['Unidades'].describe()['count']

    print('stats created')

    sold_units = dataset['Unidades'].sum()
    sold_units = int(sold_units)

    stats = {
        "sold_units": int(sold_units),
        "sales_quantity":int(sales_quantity),
        "avg_sale_price":int(avg_sale_price),
        "max_sale_price":int(max_sale_price),
        "avg_margin":int(avg_margin * 100),
        "max_margin":int(max_margin * 100),
        "avg_buy_price":int(avg_buy_price),
        "min_buy_price":int(min_buy_price),
    }
    return stats
    

def get_sell_stats(dataset):
    precio_venta_list = dataset.groupby(
        dataset['Unitario Venta'], 
        as_index=False).aggregate({ 
            'Unidades': 'sum', 
            'PorMargen': 'mean' 
            })
    
    # Precio de venta con mayor rendimiento
    max_venta_margen = precio_venta_list['PorMargen'].describe()['max']
    max_venta_precio_row = precio_venta_list[precio_venta_list['PorMargen'] == max_venta_margen]
    max_venta_precio_margen = max_venta_precio_row['Unitario Venta'].values[0]
    
    # Rendimiento promedio
    avg_margen = precio_venta_list['PorMargen'].describe()['mean']
    
    print('sell stats getted')
    
    
    X = precio_venta_list[['PorMargen']]
    Y = precio_venta_list['Unitario Venta']
    X = precio_venta_list[['PorMargen']]
    # saleY_test = precio_venta_list['Unitario Venta']

    sc_X = StandardScaler()

    X = sc_X.fit_transform(X.values)
    X = sc_X.transform(X)

    # Entrenación
    reg = LinearRegression().fit(X, Y)
    score_error2 = reg.score(X, Y)
    print('sell stats trained')
    

    predict = reg.predict(X)
    suggest_sale_price = reg.predict([[avg_margen]])
    print('sell predictions ok')
    
    plt.figure()
    plt.plot(predict, 'ro', suggest_sale_price, 'bo')
    suggessalepriceURL = upload_img(product_path, 'suggest_sale_price.jpg', plt)
    print('sells chart created')
    
    return {
        'max_throwput_sale_price': int(max_venta_precio_margen),
        'avg_throwput_sale': int(avg_margen),
        # 'score_error2': int(score_error2),
        'suggest_sale_price': int(suggest_sale_price),
        "suggest_sale_price_img": suggessalepriceURL,
        "avg_sale_price": precio_venta_list.describe()['Unitario Venta']['mean'],
        "max_sale_price": precio_venta_list.describe()['Unitario Venta']['max']
    }


def get_buy_stats(dataset):
    precio_compra_list = dataset.groupby(
        dataset['Costo Unitario'], 
        as_index=False).aggregate({
            'Unidades': 'sum',
            'PorMargen': 'mean',
            'Unitario Venta': 'mean'
            })

    # avg_buy_price = precio_compra_list['Costo Unitario'].describe()['mean']
    avg_buy_margen = precio_compra_list['PorMargen'].describe()['mean']
    print('buy stats getted')

    X = precio_compra_list[['PorMargen']]
    Y = precio_compra_list['Costo Unitario']

    sc_X = StandardScaler()

    X = sc_X.fit_transform(X.values)
    X = sc_X.transform(X)

    # Entrenación
    reg = LinearRegression().fit(X, Y)
    error_score2 = reg.score(X, Y)
    print('buy stats trained')

    predict = reg.predict(X)
    suggest_buy_price = reg.predict([[avg_buy_margen]])
    suggest_buy_price = suggest_buy_price[0] 
    print('buy stats predicted')
    
    plt.figure()
    plt.plot(predict, 'ro', suggest_buy_price, 'bo')
    suggestbutpriceURL = upload_img(product_path,'suggest_buy_price.jpg', plt)
    print('but stats chart created')
    
    return {
        # "error_score2":error_score2,
        "suggest_buy_price": int(suggest_buy_price),
        "suggest_buy_price_URL": suggestbutpriceURL,
        'avg_buy_price': precio_compra_list.describe()['Unitario Venta']['mean'],
        "max_buy_price": precio_compra_list.describe()['Unitario Venta']['min']
    }

            
def get_sales_timeline(dataset):
    
    sales_timeline = dataset.groupby(dataset['Fecha'], as_index=True).aggregate({
        'Unidades': 'sum', 
        'Ventas': 'sum',
        'Total Costo': 'sum',
    })
    
    first = sales_timeline.iloc[0].name
    last = sales_timeline.iloc[-1].name
    
    # STORAGE FILE
    timeline_df = sales_timeline.to_csv( )
    timelineURL = upload_file(product_path, 'sales-timeline.csv', timeline_df)
    print('timeline json created')
    
    timestats = get_timestats(dataset)
    
    
    unitsbymonth = get_salesvscosts(sales_timeline)
    

    monthsbox = get_boxmonths(sales_timeline)
    
    
    
    # BUILD RESULT
    result = {
        "max_monthsales": int(monthsbox['maxlength']),
        "avgsales_per_month": float("{:.2f}".format(monthsbox['avg_mes'])),
        "first_sale":first,
        "last_sale":last,
        "max_sales_month": timestats['sales_months'],
        "max_throwput_month": timestats['margen_months'],
        "files": {
            "salesvscosts_chart_URL": unitsbymonth['salesvscosts_chart_URL'],
            "unitsbymonths_df_URL": unitsbymonth['unitsbymonths_df_URL'],
            "timeline":timelineURL,
            "meses_list_df": timestats['meses_list_df'],
            "boxchart_URL": monthsbox['boxchart_URL'],
        },
    }
    
    return result
    

def get_salesvscosts(dataset):
    # GROUP DATA BY SALES IN DATE 
    df_dates = dataset.groupby(
        dataset.index, 
        as_index=True).aggregate({
            'Unidades': 'sum' 
        })
    
    
    df_periods = dataset.groupby(pd.Grouper(freq='W')).aggregate({
        # 'Unidades': 'sum',
        'Ventas': 'sum',
        'Total Costo': 'sum',
        })

    plt.figure()
    df_periods.plot(figsize = (14,6),  title='Ventas vs Costos', )
    # df_periods['Total Costo'].plot(figsize = (14,6), lw=2, label="Costos")
    
    # df_dates.plot( figsize=(12, 5), title="Unidades vendidas por mes");
    salesvscosts_chart_URL = upload_img( product_path, 'salesvscosts.jpg', plt)
    print('sales normalized jpg created')
    
    
    dates_df = df_dates.to_csv( header=False)
    unitsbymonths_df_URL = upload_file( product_path, 'unitsbymonths.csv', dates_df)
    print('unitsbymonths csv created')
    
    return {
        "salesvscosts_chart_URL": salesvscosts_chart_URL,
        "unitsbymonths_df_URL": unitsbymonths_df_URL,
    }
    
    
def get_boxmonths(dataset):
    
    
    product_dataset = dataset['Unidades']
    months = product_dataset.groupby(pd.Grouper(freq='M'))
    print('months sales grouped')
     
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
    print('time stats getted')
    
    
    # NORMALIZE DATAFRAME
    normalized_df = pd.DataFrame()
    for mes, (name, group) in zip(indexes, months):
        values = group.to_list()
        less = maxlength - len(group.values) 
        
        for zero in range(less):
            values.append(zero * 0)

        normalized_df[mes] = pd.Series(values)
            
    
    print('normalized dataframe created')
    # MAKE CHARTS IMAGES
    plt.figure()
    months_box = normalized_df.replace(0,np.nan )
    boxes = months_box.boxplot(figsize=(8,5))
    
    # boxes.figure.savefig(current_directory+"boxes-chart.jpg", format="jpg")
    boxchart_URL = upload_img( product_path, 'boxes-chart.jpg', boxes.figure)
    print('sales chart created')
    
    
    return {
        "maxlength": maxlength,
        "avg_mes": avg_mes,
        "boxchart_URL": boxchart_URL
    }

    
    
def get_timestats(dataset):
    # MONTHS SALES
    ps_dates = dataset.set_index('Fecha')
    meses_list = ps_dates.groupby(pd.Grouper(freq="M")).aggregate({
        'Unidades': 'sum',
        'Unitario Venta': 'mean',
        'Costo Unitario': 'mean',
        'Ventas': 'sum',
        'Total Costo': 'sum',
        'Margen': 'sum',
        'PorMargen': 'mean',
    }).dropna()
    print('months list grouped')
    
    # GET MAX SALES MONTH
    max_sales = meses_list['Unidades'].describe()['max']
    max_sales_month = meses_list[meses_list['Unidades'] == max_sales]
    sales_months = []
    for month, row in max_sales_month.iterrows():
        str_month = month.strftime('%B')
        sales_months.append(str_month)
    print('max sale months getted')


    # GET MAX THROWPUT MONTH
    max_throwput_month = meses_list['PorMargen'].describe()['max']
    max_margen_month = meses_list[meses_list['PorMargen'] == max_throwput_month]
    margen_months = []
    for month, row in max_margen_month.iterrows():
        str_month = month.strftime('%B')
        margen_months.append(str_month)
    print('max throwput months getted')
   
    monthlist_df = meses_list.to_csv()
    meseslistURL = upload_file( product_path, 'month-sales.csv', monthlist_df)
    print('month sales csv created')
    
    return {
        "sales_months": sales_months,
        "margen_months": margen_months,
        "meses_list_df": meseslistURL
    }
    