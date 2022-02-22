from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
from diagnostics import model_predictions
from diagnostics import dataframe_summary
from diagnostics import missing_data
from diagnostics import execution_time
from diagnostics import outdated_packages_list
from scoring import score_model
#import predict_exited_from_saved_model
import json
import os



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():
    file_pth = request.form.get("path")
    print("THe path is" + file_pth)
    preds,actual = model_predictions(file_pth)
    print("preds are")
    print(preds)
    #call the prediction function you created in Step 3
    preds_s = ""
    for p in preds:
        preds_s += str(p)
    return preds_s

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    return(score_model())

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summary_stats():
    #check means, medians, and modes for each column
    return (dataframe_summary())

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    output = {}
    output['missing_data'] = missing_data()
    output['execution_time'] = execution_time()
    output['outdated_packages_list'] = outdated_packages_list()
    return (output)


if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
