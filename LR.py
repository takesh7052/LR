import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import creme
from creme import compose
from creme import linear_model
from creme import metrics
from creme import preprocessing
from creme import datasets

metric = metrics.Accuracy()
data = datasets.Phishing()

model = compose.Pipeline(
preprocessing.StandardScaler(),
linear_model.LogisticRegression()
)

for A, b in data:
     pred = model.predict_one(A) 
     metric = metric.update(b, pred)
     print(metric)
     model = model.fit_one(A, b)
print(metric)
