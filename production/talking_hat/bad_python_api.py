from __future__ import print_function # In python 2.7
import os
import subprocess
import json
import re
from flask import Flask, request, jsonify
from inspect import getmembers, ismethod
import numpy as np
import pandas as pd
import math
import os
import pickle
import xgboost as xgb
import sys
from letter import Letter
from talking_hat import *
from sklearn.ensemble import RandomForestClassifier
import warnings



def add(params):
    return params['a'] + params['b']


def prod_predict_classes_for_name (full_name):
    featurized_person = parse_line_to_hogwarts_df(full_name)

    person_df = pd.DataFrame(featurized_person,
        columns=[
         'name', 
         'surname', 
         'is_english',
         'name_starts_with_vowel', 
         'name_starts_with_consonant',
         'name_ends_with_vowel', 
         'name_ends_with_consonant',
         'name_length', 
         'name_vowels_count',
         'name_double_vowels_count',
         'name_consonant_count',
         'name_double_consonant_count',
         'name_paired_count',
         'name_deaf_count',
         'name_sonorus_count',
         'surname_starts_with_vowel', 
         'surname_starts_with_consonant',
         'surname_ends_with_vowel', 
         'surname_ends_with_consonant',
         'surname_length', 
         'surname_vowels_count',
         'surname_double_vowels_count',
         'surname_consonant_count',
         'surname_double_consonant_count',
         'surname_paired_count',
         'surname_deaf_count',
         'surname_sonorus_count',
        ],
                             index=[0]
    )

    slitherin_model =  pickle.load(open("models/slitherin.xgbm", "rb"))
    griffindor_model = pickle.load(open("models/griffindor.xgbm", "rb"))
    ravenclaw_model = pickle.load(open("models/ravenclaw.xgbm", "rb"))
    hufflpuff_model = pickle.load(open("models/hufflpuff.xgbm", "rb"))

    predictions =  get_predctions_vector([
                        slitherin_model,
                        griffindor_model,
                        ravenclaw_model,
                        hufflpuff_model
                        ], 
                      person_df.drop(['name', 'surname'], axis=1))
    
    return {
        'slitherin': float(predictions[0][1]),
        'griffindor': float(predictions[1][1]),
        'ravenclaw': float(predictions[2][1]),
        'hufflpuff': float(predictions[3][1])
    }


def predict(params):
    fullname = params['fullname']
    print(params)
    return prod_predict_classes_for_name(fullname)


def create_app():
    app = Flask(__name__)

    functions_list = [add, predict]

    @app.route('/<func_name>', methods=['POST'])
    def api_root(func_name):
        for function in functions_list:
            if function.__name__ == func_name:
                try:
                    json_req_data = request.get_json()
                    if json_req_data:
                        res = function(json_req_data)
                    else:
                        return jsonify({"error": "error in receiving the json input"})
                except Exception as e:
                    data = {
                        "error": "error while running the function"
                    }
                    if hasattr(e, 'message'):
                        data['message'] = e.message
                    elif len(e.args) >= 1:
                        data['message'] = e.args[0]
                    return jsonify(data)
                return jsonify({"success": True, "result": res})
        output_string = 'function: %s not found' % func_name
        return jsonify({"error": output_string})


    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0')

