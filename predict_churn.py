
import pandas as pd

from pycaret.classification import predict_model, load_model


df = pd.read_csv("new_churn_data.csv")
    

model = load_model('cbc')

predictions = predict_model(model, data=df)

    
print('predictions:')

print(predictions)