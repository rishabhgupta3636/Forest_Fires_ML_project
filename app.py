import pickle
from flask import Flask,render_template,jsonify,request

app=Flask(__name__)

scaler = pickle.load(open("models/scaler_forest_fires.pkl",'rb'))
ridge = pickle.load(open("models/ridge_regressor_forest_fires.pkl",'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge.predict(new_data_scaled)

        return render_template('home.html',result=result[0])        
    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0',debug=True)