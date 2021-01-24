# from IPython.core.display import display
from api.firebase_app import FirebaseApp
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from api.controllers.file import upload_file

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
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        
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
        product_name = product_selected['Descripcion'].unique()[0].strip()
        product_refname = product_name.replace(' ', '_').lower()
        product_path = 'tables/'+table_id+'/products/'+product_id+'/'
        local_path = 'api/uploads/'
        

        # GET STATS
        product_stats = get_product_stats(product_selected)
        print('product_stats ok')
        # display(json.dumps(product_stats, indent=4))
        sell_stats = get_sell_stats(product_selected)
        print('sell_stats ok')
        # display(json.dumps(sell_stats, indent=4))
        time_stats = get_sales_timeline(product_selected)
        print('time_stats ok')
        # display(json.dumps(time_stats, indent=4))
        buy_stats = get_buy_stats(product_selected)
        print('buy_stats ok')
        # display(json.dumps(buy_stats, indent=4))
        

        # STORAGE FILES
        product_selected.to_csv(local_path+'dataset.csv', orient="columns")
        datasetURL = upload_file(local_path, product_path,'dataset.csv')
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
    sales_quantity =  dataset['Unidades'].describe()['count']
    avg_sell_price =  dataset['Unitario Venta'].describe()['mean']
    max_sell_price =  dataset['Unitario Venta'].describe()['max']
    avg_margin =      dataset['PorMargen'].describe()['mean']
    max_margin =      dataset['PorMargen'].describe()['max']
    avg_purch_price = dataset['Costo Unitario'].describe()['mean']
    min_purch_price = dataset['Costo Unitario'].describe()['min']

    print('stats created')

    sold_units = dataset['Unidades'].sum()
    sold_units = int(sold_units)

    stats = {
        "sold_units": int(sold_units),
        "sales_quantity":int(sales_quantity),
        "avg_sale_price":int(avg_sell_price),
        "max_sale_price":int(max_sell_price),
        "avg_margin":int(avg_margin * 100),
        "max_margin":int(max_margin * 100),
        "avg_buy_price":int(avg_purch_price),
        "min_buy_price":int(min_purch_price),
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
    
    
    plt.plot(predict, 'ro', suggest_sale_price, 'bo')
    plt.savefig(local_path+'suggest_sale_price.jpg')
    suggessalepriceURL = upload_file(local_path, product_path, 'suggest_sale_price.jpg')
    print('sells chart created')
    
    return {
        'max_throwput_sale_price': int(max_venta_precio_margen),
        'avg_throwput_sale': int(avg_margen),
        'score_error2': int(score_error2),
        'suggest_sale_price': int(suggest_sale_price),
        "suggest_sale_price_img": suggessalepriceURL
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
    
    plt.plot(predict, 'ro', suggest_buy_price, 'bo')
    plt.savefig(local_path+'suggest_buy_price.jpg')
    suggestbutpriceURL = upload_file(local_path, product_path,'suggest_buy_price.jpg')
    print('but stats chart created')
    
    return {
        "error_score2":int(error_score2),
        "suggest_buy_price": int(suggest_buy_price),
        "suggest_buy_price_URL": suggestbutpriceURL 
    }
            
def get_sales_timeline(dataset):
    
    sales_timeline = dataset.groupby(dataset['Fecha'], as_index=True).aggregate({
        'Unidades': 'sum', 
        'Ventas': 'sum',
        'Total Costo': 'sum',
    })
    
    # GROUP DATA BY SALES IN DATE 
    df_dates = sales_timeline.groupby(
        sales_timeline.index, 
        as_index=True).aggregate({
            'Unidades': 'sum' 
        })
    
    
    
    # GROUP DATA BY MONTHS
    product_dataset = sales_timeline['Unidades']
    months = product_dataset.groupby(pd.Grouper(freq='M'))
    print('months sales grouped')
    
    
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
    months_box = normalized_df.replace(0,np.nan )
    months_box.plot.box(figsize=(8,5))
    
    
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
   
   
    df_periods = sales_timeline.groupby(pd.Grouper(freq='W')).aggregate({
        'Unidades': 'sum',
        'Ventas': 'sum',
        'Total Costo': 'sum',
        })

    df_periods['Ventas'].plot(figsize = (14,6), lw=2, title='ventas vs costos')
    df_periods['Total Costo'].plot(figsize = (14,6), lw=2)
    
    
   
   
    # STORAGE FILE
    sales_timeline.to_csv( local_path+'sales-timeline.json')
    timelineURL = upload_file(local_path, product_path, 'sales-timeline.json')
    print('timeline json created')
    
    df_dates.to_csv(local_path+'sales-dates.csv', header=False)
    salesdatesURL = upload_file(local_path, product_path, 'sales-dates.csv')
    print('sales-dates csv created')
    
    plt.savefig(local_path+'month_sales_normalized.jpg')
    monthsaleschartURL = upload_file(local_path, product_path, 'month_sales_normalized.jpg')
    print('sales normalized jpg created')
    
    meses_list.to_csv(local_path+'month-sales.csv')
    meseslistURL = upload_file(local_path, product_path, 'month-sales.csv')
    print('month sales csv created')
    
    plt.savefig(local_path+'sales-chart.jpg')
    saleschartURL = upload_file(local_path, product_path, 'sales-chart.jpg')
    print('sales chart created')
    
    
    first = dataset.iloc[0]['Fecha']
    last = dataset.iloc[-1]['Fecha']
    
    
    # BUILD RESULT
    result = {
        "max_monthsales": int(maxlength),
        "avgsales_per_month": float("{:.2f}".format(avg_mes)),
        "first_sale":first,
        "last_sale":last,
        "max_sales_month": sales_months,
        "max_throwput_month": margen_months,
        "files": {
            "month_sales_chart": monthsaleschartURL,
            "sales_dates": salesdatesURL,
            "timeline":timelineURL,
            "month_sales": meseslistURL,
            "sales_chart": saleschartURL,
        },
    }
    
    return result
    





    

    