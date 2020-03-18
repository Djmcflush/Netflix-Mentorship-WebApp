import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[4, 300, 500,4,'dog killer gets murdered']]))