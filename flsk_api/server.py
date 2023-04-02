import numpy as np
from flask import Flask, request, jsonify, make_response
from flask_restful import Resource,Api
import pandas as pd
from itertools import combinations
from nltk.corpus import wordnet
import requests
from bs4 import BeautifulSoup
from collections import Counter

import pickle


app = Flask(__name__)

api = Api(app)

# Load the model
model = pickle.load(open('flsk_api\model.pkl','rb'))

#synonyms
def synonyms(term):
    synonyms = []
    response = requests.get('https://www.thesaurus.com/browse/{}'.format(term))
    soup = BeautifulSoup(response.content,  "html.parser")
    try:
        container=soup.find('section', {'class': 'MainContentContainer'}) 
        row=container.find('div',{'class':'css-191l5o0-ClassicContentCard'})
        row = row.find_all('li')
        for x in row:
            synonyms.append(x.get_text())
    except:
        None
    for syn in wordnet.synsets(term):
        synonyms+=syn.lemma_names()
    return set(synonyms)

#For POST request to http://localhost:5000/pred
class prediction(Resource):
    def post(self):
        if request.is_json:
            # Get the data from the POST request.
            data = request.get_json(force=True)
            # Make prediction using model loaded from disk as per the data.
            prediction = model.predict([np.array(data)])
            # Take the first value of prediction
            output = prediction[0]
            return jsonify(output)
        else:
            return {'error' : 'Request must be in JSON'}, 400
        
        
#For POST request to http://localhost:5000/symptom
class process_symptom(Resource):
    def post(self):
        if request.is_json:
            df_norm = pd.read_csv("Dataset/dis_sym_dataset_norm.csv")
            X = df_norm.iloc[:, 1:]
            dataset_symptoms = list(X.columns)
            
            processed_user_symptoms= list(request.get_json(force=True))
            
            user_symptoms = []
            for user_sym in processed_user_symptoms:
                user_sym = user_sym.split()
                str_sym = set()
                for comb in range(1, len(user_sym)+1):
                    for subset in combinations(user_sym, comb):
                        subset=' '.join(subset)
                        subset = synonyms(subset) 
                        str_sym.update(subset)
            str_sym.add(' '.join(user_sym))
            user_symptoms.append(' '.join(str_sym).replace('_',' '))
        
            found_symptoms = set()
            for idx, data_sym in enumerate(dataset_symptoms):
                data_sym_split=data_sym.split()
                for user_sym in user_symptoms:
                    count=0
                    for symp in data_sym_split:
                        if symp in user_sym.split():
                            count+=1
                if count/len(data_sym_split)>0.5:
                    found_symptoms.add(data_sym)
            found_symptoms = list(found_symptoms)
            
        
            
            return jsonify(found_symptoms)
        
        else:
            return {'error' : 'Request must be in JSON'}, 400

api.add_resource(prediction, '/pred')
api.add_resource(process_symptom, '/symptom')


            


if __name__ == '__main__':
    app.run(debug=True)