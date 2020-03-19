import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from Freq_Analysis import Freq
from sklearn.preprocessing import StandardScaler 
import pandas as pd
app = Flask(__name__)
model = pickle.load(open('Finalmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
def create_scalers():
    df = pd.read_csv('finaldf.csv')
    df = df.drop('Unnamed: 0', axis =1)

    fdf = pd.read_csv('feature.csv')
    fdf = fdf.drop('Unnamed: 0', axis =1)

    target_pd = pd.DataFrame(df['Target'])
    df = df.drop('Target', axis=1)

    scaler = StandardScaler() 
    target_scaler = StandardScaler()
    # To scale data 
    
    scaler.fit(fdf) 
    target_scaler.fit(target_pd)
    return scaler, target_scaler
    

@app.route('/predict',methods=['POST'])
def predict():
    features =[]
    for x in request.form.values():
        try :
            feature = int(x)
            features.append(feature)
        except:
            description = x
    #int_features = [int(x) for x in request.form.values()]
    #description = int_features.pop()
    tokenized_description = Freq(description)
    tokenized_description = sorted(tokenized_description)
    rare_words,common_words, all_words =0,0,0
    for x in tokenized_description[0:5]:
        rare_words += x
    for x in tokenized_description[len(tokenized_description)-5 : len(tokenized_description)]:
        common_words+= x
    rare_words = rare_words/5
    common_words = common_words/5
    all_words = sum(tokenized_description) /len(tokenized_description)
    features.append(all_words)
    features.append(rare_words)
    features.append(common_words)
    print(common_words, 'This is the common english score')
    final_features = [np.array(features)]
    
    # And now to load..
    scaler, target_scaler = create_scalers()

    
    final_features = scaler.transform(final_features)

    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    output = target_scaler.inverse_transform([output,.7])
    output = output[0]
    output = round(output,2)
    output = '{:,}'.format(output) 
    
    return render_template('index.html', prediction_text='Overseas Revenue for this film should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)