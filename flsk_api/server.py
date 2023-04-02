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
import operator


app = Flask(__name__)

api = Api(app)

# Load the model
model = pickle.load(open('flsk_api\model.pkl','rb'))

# Global vars and initializations

global df_norm, found_symptoms, dataset_symptoms
df_norm = pd.read_csv("Dataset/dis_sym_dataset_norm.csv")
X = df_norm.iloc[:, 1:]
dataset_symptoms = list(X.columns)

# Method to find synonyms
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

   
# For POST request to http://localhost:5000/symptom to get the top combination of the symptoms.
class process_symptom(Resource): 
    def post(self):
        if request.is_json:
            
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
        
# For POST request to http://localhost:5000/topindices to send selected indices by the user.
class process_indices(Resource):
    def post(self):
        if request.is_json:
            dis_list = set()
            final_symp = [] 
            counter_list = []
            select_list = list(request.get_json(force=True))
            for idx in select_list:
                symp=found_symptoms[int(idx)]
                final_symp.append(symp)
                dis_list.update(set(df_norm[df_norm[symp]==1]['label_dis']))
            
            for dis in dis_list:
                row = df_norm.loc[df_norm['label_dis'] == dis].values.tolist()
                row[0].pop(0)
                for idx,val in enumerate(row[0]):
                    if val!=0 and dataset_symptoms[idx] not in final_symp:
                        counter_list.append(dataset_symptoms[idx])
            
            dict_symp = dict(Counter(counter_list))
            dict_symp_tup = sorted(dict_symp.items(), key=operator.itemgetter(1),reverse=True)  
        
            return jsonify(dict_symp_tup)
        
        else:
            return {'error' : 'Request must be in JSON'}, 400

# For POST request to http://localhost:5000/pred for predicting the disease with the final list sent.
class prediction(Resource):
    def post(self):
        if request.is_json:
            # Get the data from the POST request.
            data = list(request.get_json(force=True))
            
            # boolean sample generation
            sample = [0 for x in range(0,len(dataset_symptoms))]
            for val in data:
                sample[dataset_symptoms.index(val)]=1
                
            # Make prediction using model loaded from disk as per the data.
            prediction = model.predict([np.array(sample)])
            
            # Take the first value of prediction
            output = prediction[0]
            
            return jsonify(output)
        
        else:
            return {'error' : 'Request must be in JSON'}, 400
        

# api initializations

api.add_resource(prediction, '/pred')
api.add_resource(process_symptom, '/symptom')
api.add_resource(process_indices,'/topindices')



if __name__ == '__main__':
    app.run(debug=True)