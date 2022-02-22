import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 




##############Function for reporting
def score_model():
    cur_path = os.getcwd()
    preds,actual = model_predictions()
    clf = SVC(random_state=0)
    clf.fit(preds.reshape(-1, 1), actual.reshape(-1, 1))
    plot_confusion_matrix(clf, preds.reshape(-1, 1), actual.reshape(-1, 1))
    plt.savefig(cur_path + "/confusionmatrix.png")

    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace





if __name__ == '__main__':
    score_model()
