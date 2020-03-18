import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from Freq_Analysis import Freq

app = Flask(__name__)
model = pickle.load(open('Finalmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

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
    all_words = sum(tokenized_description)/len(tokenized_description)
    features.append(all_words)
    features.append(rare_words)
    features.append(common_words)
    final_features = [np.array(features)]

    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Overseas Revenue for this film should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)