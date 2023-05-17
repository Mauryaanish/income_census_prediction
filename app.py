from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app = application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Age=int(request.form.get('Age')),
            Work_class=str(request.form.get('Work_class')),
            Final_weight=int(request.form.get('Final_weight')),
            Education=request.form.get('Education'),
            Education_num=int(request.form.get('Education_num')),
            Marital_status= request.form.get('Marital_status'),
            Occupation= request.form.get('Occupation'),
            Relationship= request.form.get('Relationship'),
            Race =request.form.get('Race'),
            Sex =request.form.get('Sex'),
            Capital_gain=int(request.form.get('Capital_gain')),
            Capital_loss=int(request.form.get('Capital_loss')),
            Hours_per_week=int(request.form.get('Hours_per_week'))
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True ,port = 5500)
