from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipelines.predict_pipeline import CustomData, PredictOutput

application = Flask(__name__)
app=application

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/home", methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template("home.html")

    else:
        input_data = CustomData(
            request.form.get('gender'),
            request.form.get('ethnicity'),
            request.form.get('parental_level_of_education'),
            request.form.get('lunch'),
            request.form.get('test_preparation_course'),
            float(request.form.get('reading_score')),
            float(request.form.get('writing_score'))
        )

        df = input_data.input_data_to_dataframe()

        pred = PredictOutput()
        results = pred.predict(df)

        return render_template("home.html", result=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
