from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('modelo.pkl','rb'))
phone=pd.read_csv('novo_dados.csv')

@app.route('/',methods=['GET','POST'])
def index():
    marcas=sorted(phone['Brand'].unique())
    modelos=sorted(phone['Model'].unique())
    memoria_ram=sorted(phone['Memory'].unique(),reverse=True)
    memoria_interna=phone['Storage'].unique()

    marcas.insert(0,'Escolha uma Marca')
    return render_template('index.html',marcas=marcas, modelos=modelos, memoria_ram=memoria_ram,memoria_interna=memoria_interna)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    marca=request.form.get('marca_phone')
    modelo=request.form.get('model_a')
    memoria_ram=request.form.get('memoria')
    memoria_interna=request.form.get('memoria_inter')
    avaliacao=request.form.get('avalia')

    prediction=model.predict(pd.DataFrame(columns=['Brand', 'Model', 'Memory', 'Storage', 'Rating'],
                              data=np.array([marca,modelo,memoria_ram,memoria_interna,avaliacao]).reshape(1, 5)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run()
