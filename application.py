from unicodedata import name
from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('RandomForestModel.pkl','rb'))
cars=pd.read_csv('cleaned_cars.csv')

@app.route('/',methods=['GET','POST'])
def index():
    # companies=sorted(car['company'].unique())
    # car_models=sorted(car['name'].unique())
    # year=sorted(car['year'].unique(),reverse=True)
    # fuel_type=car['fuel_type'].unique()

    # companies.insert(0,'Select Company')
    # return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)
    name=sorted(cars['name'].unique())
    year=sorted(cars['year'].unique())
    fuel=sorted(cars['fuel'].unique())
    seller_type=sorted(cars['seller_type'].unique())
    transmission=sorted(cars['transmission'].unique())
    owner=sorted(cars['owner'].unique())
    mileage=sorted(cars['mileage'].unique())
    engine=sorted(cars['engine'].unique())
    max_power=sorted(cars['max_power'].unique())
    seats=sorted(cars['seats'].unique())
    return render_template('index.html', name=name, year=year, fuel=fuel, seller_type=seller_type, 
    transmission=transmission, owner=owner, mileage=mileage, engine=engine, max_power=max_power, seats=seats)




@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    name=request.form.get('name')
    year=request.form.get('year')
    fuel=request.form.get('fuel')
    seller_type=request.form.get('seller_type')
    transmission=request.form.get('transmission')
    owner=request.form.get('owner')
    mileage=request.form.get('mileage')
    engine=request.form.get('engine')
    max_power=request.form.get('max_power')
    seats=request.form.get('seats')

    km_driven=request.form.get('km_driven')

    prediction=model.predict(pd.DataFrame([[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]], 
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats']))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__=='__main__':
    app.run()