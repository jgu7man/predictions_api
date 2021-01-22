from api.firebase_app import FirebaseApp
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.files.storage import default_storage
from api.controllers.file import upload_file
from sklearn.svm import SVR
from pmdarima.arima import auto_arima
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# from IPython.core.display import display


import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib  as mlp
import numpy as np
import locale
import json, codecs
import urllib.request

db = FirebaseApp.fs
st = FirebaseApp.st
mlp.style.use('seaborn')


def ValidateRequest(query_params, request):

    global message
    global params
    params = {}
    for param in query_params:
        try: params[param] = request[param]
        except: message = f'La petición debe incluir el parámetro "{param}"' 
    
    return params
    


class MonthSalesPrediction(APIView):
    def get(self, request, format=None):
        
        
        
        # VALIDATE PARAMS
        try: params = ValidateRequest(['table', 'product', 'months'], request.query_params)
        except: return Response({
            'message': 'Falto un parámetro en la petición',
            'status': 400
        }, status=400)
    
        table_id = params['table']
        product_id = params['product']
        query_months = params['months']
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        
        
        
        # IMPORT FILE
        global cloud_path
        global local_path
        cloud_path = 'tables/'+table_id+'/products/'+product_id
        doc_ref = db.document(cloud_path)
        
        try:doc = doc_ref.get()
        except: return Response({
            'message':'No se encontró el documento en la base de datos',
            'status': 404
        })
        print('Petición realizada correctamente')
        
        
        
        
        # READ TABLE
        doc_URL = doc.to_dict()['time_stats']['files']['month_sales']
        dataset = pd.read_csv(doc_URL,  decimal=".")  
        local_path = 'api/uploads/'
        
        print('Archivo cargado')
        # display(dataset.head())
        
        
        
        
        # TRAIN AND PREDICT
        train_result = train_year_predictions(dataset)
        reg = train_result['reg']
        score = train_result['score']
        predictionsURL = train_result['predictionsURL']
        print('trained')
        
        
        
        
        
        
        # meses que el usuario quiere conocer en predicción
        months_query = int(query_months)
        months_required = score[1:months_query +1 ]

        predict_months = reg.predict(months_required)
        predicted_cant = math.ceil(predict_months.sum())
        print('Se venderán', predicted_cant, 'unidades en', months_query, 'meses')
        
        # plt.plot(predict_year, 'ro', predict_months, 'bo')
        # plt.savefig(local_path+'months_prediction.jpg')
        # monthpredictionURL = upload_file(local_path, cloud_path, 'months_prediction.jpg')
        
        result = {
            "predicted_cant":int(predicted_cant),
            # "months_prediction_chart": monthpredictionURL,
        }
        
        
        doc_ref.update({ "year_predictions_URL":predictionsURL})
        
        return Response({
            "result": result,
            "status":200,
            "message":"ok",
        })
        
class ReverseSalesPredictions(APIView):
    def get(self, request, format=None):
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        
        try: params = ValidateRequest(['table', 'product', 'cant'], request.query_params)
        except: return Response({
            'message': 'Falto un parámetro en la petición',
            'status': 400
        }, status=400)
        
        table_id = params['table']
        product_id = params['product']
        query_cant = params['cant']
        
        # IMPORT FILE
        global cloud_path
        cloud_path = 'tables/'+table_id+'/products/'+product_id
        doc_ref = db.document(cloud_path)
        
        try:doc = doc_ref.get()
        except: return Response({
            'message':'No se encontró el documento en la base de datos',
            'status': 404
        })
        
        print('Petición realizada correctamente')
        
        global local_path
        doc_URL = doc.to_dict()['time_stats']['files']['month_sales']
        dataset = pd.read_csv(doc_URL,  decimal=".")  
        local_path = 'api/uploads/'
        
        print('Archivo cargado')
        # display(dataset.head())
        

        
        
        # TRAIN AND PREDICT
        train_result = train_year_predictions(dataset)
        reg = train_result['reg']
        score = train_result['score']
        predictionsURL = train_result['predictionsURL']
        print('trained')
        
        
        
        remaining_cant = int(query_cant)
        months_required = score[0:12]
        predict_months = reg.predict(months_required)
        months_cant = 0
        for cant in predict_months:
            remaining_cant -= cant
            if remaining_cant > 0: months_cant = months_cant + 1
            else: break
        
        print(f"{query_cant} unidades se venderán en {months_cant} meses")
        
        result = {
            "months_cant":int(months_cant),
        }
        
        
        doc_ref.update({ "year_predictions_URL":predictionsURL})
        return Response({
            "result": result,
            "status":200,
            "message":"ok",
        })



def train_year_predictions(dataset):
    
    X = dataset[['Unitario Venta']]
    Y = dataset['Unidades']

    sc_X = StandardScaler()

    X = sc_X.fit_transform(X)
    X = sc_X.transform(X)

    # Entrenación
    reg = LinearRegression().fit(X, Y)
    print("The Linear regression score on training data is ", round(reg.score(X, Y),2))
    

    # Basado en la cantidad de meses obtenidos, se ajusta para predecir al menos 1 año
    repeats = (12 - len(X)) / int(len(X)) 
    repeats = math.ceil(repeats) + 1
    X = np.tile(X,(repeats, 1))
    
    # crea lista de predicciones
    predict_year = reg.predict(X)
    p = predict_year.tolist()
    json.dump(p, codecs.open(local_path+'year_predictions.json', 'w'))
    print(cloud_path)
    yearpredictionsURL = upload_file(local_path, cloud_path+'/', 'year_predictions.json')
    print(yearpredictionsURL)
    print('preicciones ok')
    
    return {
        'reg': reg,
        'score': X,
        'predictionsURL': yearpredictionsURL
    }




class AnalyzeProvOffering(APIView):
    def get(self, request, format=None):
        # VALIDATE THERE IS TABLE ID
        
        try: params = ValidateRequest(['table', 'product', 'provider', 'condition', 'desc', 'stock'], request.query_params)
        except: return Response({
            'message': 'Falto un parámetro en la petición',
            'status': 400
        }, status=400)
        
        table_id = params['table']
        product_id = params['product']
        provider = params['provider']
        condition = int(params['condition'])
        desc = int(params['desc']) / 100
        stock = int(params['stock'])
        


        # IMPORT FILE
        cloud_path = 'tables/'+table_id+'/products/'+product_id
        doc_ref = db.document(cloud_path)
        
        try:doc = doc_ref.get()
        except: return Response({
            'message':'No se encontró el documento en la base de datos',
            'status': 404
        })
        
        doc_URL = doc.to_dict()['year_predictions_URL']
        global dataset
        with urllib.request.urlopen(doc_URL) as url:
            dataset = json.loads(url.read()) 
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        local_path = 'api/uploads/'
        
        
        
        
        # DEF DATA
        suggest_sale_price = doc.to_dict()['sell_stats']['suggest_sale_price']
        # print('suggest_sale_price', doc.to_dict()['sell_stats']['suggest_sale_price'])
        suggest_buy_price = doc.to_dict()['buy_stats']['suggest_buy_price']
        # print('suggest_buy_price', doc.to_dict()['buy_stats']['suggest_buy_price'])
        avg_buy_price = doc.to_dict()['product_stats']['avg_buy_price']
        # print('avg_buy_price', doc.to_dict()['product_stats']['avg_buy_price'])
        
        inv_cap = stock * avg_buy_price
        # print('inv_cap', stock * avg_buy_price)
        saving = suggest_buy_price * desc
        # print('saving', suggest_buy_price * desc)
        desc_price = avg_buy_price - saving
        # print('desc_price', avg_buy_price - saving)
        total_saving = saving * condition
        # print('total_saving', saving * condition)
        # print(desc_price, condition)
        invest = desc_price * condition
        # print('invest', invest)
        total_inv = inv_cap + invest
        # print('total_inv', inv_cap + (desc_price * condition))
        remaining_inv = -(total_inv)
        # print('remaining_inv', -(total_inv))
        remaining_stock = stock + condition
        # print('remaining_stock', stock + condition)
        year_sales = dataset[0:12]
        
        global profits
        global utilities
        global porUtilities
        global viability
        global message
        
        profits = []
        for cant in year_sales:
            # print('cantidad restante', remaining_stock)
            remaining_stock = remaining_stock - cant
            possible_sales = cant * suggest_sale_price
            # print('posibles ventas',possible_sales)
            remaining_inv = remaining_inv + possible_sales
            # print('inversión restante',remaining_inv)
            if remaining_inv > 0 and remaining_stock > 0 :
                profits.append(remaining_stock * suggest_sale_price)
        
        if len(profits) > 0:
            profits = sum(profits)
            # print(profits)
            utilities = profits - total_saving
            # print(utilities)
            if utilities > 0:
                porUtilities = (total_saving * 100)/profits
                viability = True
                message = 'La oferta es conveniente'
            else: 
                viability = False
                porUtilities = 0
                message = 'Solicita más descuento'
        else:
            viability = False
            message = 'La condición de compra es alta'
            utilities = 0
            porUtilities = 0

        print(message)
        result = {
            "viability":viability,
            "message":message,
            "invest_capital": inv_cap,
            "saving":saving,
            "desc_price": desc_price,
            "total_saving": total_saving,
            "total_invest": total_inv,
            "profits": profits,
            "utilities": utilities,
            "percent_utilities": porUtilities,
        }
        
        
        doc_ref.collection('providers_offers').document(provider).set(result)
        
        return Response({
            "result": result,
            "status":200,
            "message":"ok",
        })

class EstimatedPrediction(APIView):
    def post(self, request, format=None):
        # VALIDATE THERE IS TABLE ID
        try:
            table_id = request.data['table']
        except: return Response({
                'message': 'La petición debe incluir el atributo "table" en el body',
                'status':500
            })
        
        # VALIDATE IS PRODUCT ID     
        try: product_id = request.data['product']
        except: return Response({
                'message': 'La petición debe incluir el atributo "product" en el body',
                'status':500
            })
        
        # VALIDATE IS TEST SIZE    
        try: test_size = request.data['test_size']
        except: return Response({
                'message': 'La petición debe incluir el atributo "test_size" en el body',
                'status':500
            })
        
        # VALIDATE IS TEST SIZE    
        try: window_size = request.data['window_size']
        except: return Response({
                'message': 'La petición debe incluir el atributo "window_size" en el body',
                'status':500
            })
    
        # IMPORT FILE
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        cloud_path = 'tables/'+table_id+'/products/'+product_id
        doc_ref = db.document(cloud_path)
        
        try:doc = doc_ref.get()
        except: return Response({
            'message':'No se encontró el documento en la base de datos',
            'status': 404
        })
        
        doc_URL = doc.to_dict()['stats']['files']['timeline']
        product_name = doc.to_dict()['name']
        dataset = pd.read_json(doc_URL)
        dataset = dataset.fillna(method='ffill')
        
        predict_results = make_estimate_prediction(dataset, test_size, window_size, product_name)
        
        est_pred_imgURL = upload_file('api/uploads/',cloud_path, 'estimated_predict.jpg' )
        default_storage.delete('api/uploads/estimated_predict.jpg')
        est_pred_jsonURL = upload_file('api/uploads/',cloud_path, 'estimated_predict.json' )
        default_storage.delete('api/uploads/estimated_predict.json')
        
        predict_results['imgURL'] = est_pred_imgURL
        predict_results['jsonURL'] = est_pred_jsonURL
    
        try: doc_ref.collection(u'predictions').document(u'estimated').set(predict_results)
        except: return Response({
            'message':'No se pudo guardar',
            'status':500
        })
        
        print('si se guardó')
        
        return Response({
            "result": predict_results,
            "status":200,
            "message":"ok",
        })



def make_estimate_prediction(dataset, test_size, window_size, product_name):
    df_shift = dataset['unidades'].shift(1)
    df_mean_roll = df_shift.rolling(window_size).mean()
    df_std_roll = df_shift.rolling(window_size).std()
    df_mean_roll.name = "mean_roll"
    df_std_roll.name = "std_roll"
    df_mean_roll.index = dataset.index
    df_std_roll.index = dataset.index
    
    df_w = pd.concat([dataset['unidades'],df_mean_roll,df_std_roll],axis=1)

    df_w = df_w[window_size:]
    
    test_cant = int((dataset['unidades'].describe()['count'])*(test_size*.01))

    test = df_w[-test_cant:]
    train = df_w[:-test_cant]
    X_test = test.drop("unidades",axis = 1)
    y_test = test["unidades"]
    X_train = train.drop("unidades",axis = 1)
    y_train = train["unidades"]



    clf = SVR(gamma="scale")
    clf.fit(X_train, y_train)
    y_train_hat = pd.Series(clf.predict(X_train),index=y_train.index)
    y_test_hat = pd.Series(clf.predict(X_test),index=y_test.index)
    hat_groups = y_test_hat.groupby(pd.Grouper(freq='M'))

    total_predicted = np.sum(y_test_hat)
    mean_predicted = y_test_hat.describe()['mean']
    chart_data = {
        "train":y_train,
        "prediction":y_test_hat,
        "test":y_test,
    }
    predict_df = pd.DataFrame(data=chart_data)
    predict_df.to_json('api/uploads/estimated_predict.json', orient="columns")

    

    plt.figure(figsize=(12,6));
    plt.plot(y_train ,label='Datos de entrenamiento');
    plt.plot(y_test_hat,label='Predicción');
    plt.plot(y_test , label='Datos reales');
    plt.legend(loc='best')
    plt.title('Predicción de ventas ' + product_name)
    plt.savefig('api/uploads/estimated_predict.jpg')

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, y_test_hat)
    print('Margen de error: {}'.format(mse))
    
    
    result = {
        "total_predicted": int(total_predicted),
        "months_predicted": len(hat_groups),
        "avg_for_sell": float("{:.2f}".format(mean_predicted)),
        "error_mean" : float("{:.2f}".format(mse)),
    }
    
    return result



class ARIMAprediction(APIView):
    def post(self, request, format=None):
        # VALIDATE THERE IS TABLE ID
        try:
            table_id = request.data['table']
        except: return Response({
                'message': 'La petición debe incluir el atributo "table" en el body',
                'status':500
            })
        
        # VALIDATE IS PRODUCT ID     
        try: product_id = request.data['product']
        except: return Response({
                'message': 'La petición debe incluir el atributo "product" en el body',
                'status':500
            })
        
        # VALIDATE IS TEST SIZE   
        try: test_size = request.data['test_size']
        except: return Response({
                'message': 'La petición debe incluir el atributo "test_size" en el body',
                'status':500
            })
        
        print('body_ok')
        
        locale.setlocale(locale.LC_TIME, 'es_MX.UTF-8')
        cloud_path = 'tables/'+table_id+'/products/'+product_id
        doc_ref = db.document(cloud_path)
        
        try: doc = doc_ref.get()
        except: return Response({
            'message':'No se encontró el documento en la base de datos',
            'status':404
        })
        # files = doc.to_dict()['files']
        
        
        # VALIDATE DATASET
        try: doc_URL = doc.to_dict()['month_details']['sales_dates']
        except: return Response({
            'message':'Falta dataset de datos normalizados',
            'status': 404
        })
            
            
        print('firebase ok')
        
        dataset = pd.read_csv(doc_URL, header=None, index_col=0, parse_dates=True, squeeze=True)

        # split_point = math.floor( len(month_sales) *.80)
        # dataset, validation = month_sales[0:split_point], month_sales[split_point:]
        # print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
        # dataset.to_csv('dataset.csv', header=False)
        # validation.to_csv('validation.csv', header=False)
        
        print(dataset)
        groups = dataset.groupby(pd.Grouper(freq='M'))
        sells_avg = groups.describe()['count'].sum()/len(groups)

        
        print('doc readed')
        print(dataset)
        try:
            Arima_model=auto_arima(dataset, start_p=1, start_q=1, max_p=8, max_q=8, start_P=0, start_Q=0, max_P=8, max_Q=8, m=12, seasonal=True, trace=True, d=1, D=1, error_action='warn', suppress_warnings=True, random_state = 20, n_fits=30)
        except: return Response({
            'message':'No hay suficientes datos para realizar esta predicción',
            'status':204
        })
        
        
        predict = Arima_model.predict(n_periods=test_size)
        total_value_predictions = np.sum(predict)
        months_predicted = total_value_predictions / sells_avg
        
        print('prediction ok')
        
        plt.figure(figsize=(12,6))
        plt.plot(predict);
        plt.savefig('api/uploads/arima_prediction.jpg')
        
        predict_df  = pd.DataFrame(data=predict)
        predict_df.to_json('api/uploads/arima_prediction.json', orient="index")
        
        arima_pred_imgURL = upload_file('api/uploads/', cloud_path, 'arima_prediction.jpg')
        default_storage.delete('api/uploads/arima_prediction.jpg')
        arima_pred_jsonURL = upload_file('api/uploads/', cloud_path, 'arima_prediction.json')
        default_storage.delete('api/uploads/arima_prediction.json')
        
        print('files ok')
        
        predict_result = {
            "avg_for_sell": float("{:.2f}".format(sells_avg)),
            "total_predicted": int(total_value_predictions),
            "months_predicted":float("{:.2f}".format(months_predicted)),
            "imgURL": arima_pred_imgURL,
            "jsonURL": arima_pred_jsonURL,
        }
        
        doc_ref.collection(u'predictions').document(u'arima').set(predict_result)
        
        print('saved ok')
        return Response({
            'result': predict_result,
            'status':200,
            'message': 'ok',
        })
        
        