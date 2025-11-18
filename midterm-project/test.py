import requests

potability_prediction_service_url = 'http://localhost:9696/predict'

def print_prediction(predictions):
    if predictions['is_potable']:
        print('water is potable')
    else:
        print('water is not potable')

# example water sample 1 for prediction
water_sample_1 = {
    'ph': 8.596391179552715,
    'Hardness': 189.52316077036443,
    'Solids': 14518.974500689232,
    'Chloramines': 5.124129421964379,
    'Sulfate': 422.99041301909364,
    'Conductivity': 348.0414888599463,
    'Organic_carbon': 17.358071468102764,
    'Trihalomethanes': None,
    'Turbidity': 3.519884366217547
}
response_1 = requests.post(potability_prediction_service_url, json=water_sample_1)
print("Water Sample 1 Prediction:")
print_prediction(response_1.json())

# water_sample_2 with values 5.418503763025523,187.7768087099991,35902.715683707924,4.357087861232644,,454.52029561547636,6.374070260129599,103.37300526956227,4.506539591260472
water_sample_2 = {
    'ph': 5.418503763025523,
    'Hardness': 187.7768087099991,
    'Solids': 35902.715683707924,
    'Chloramines': 4.357087861232644,
    'Sulfate': None,
    'Conductivity': 454.52029561547636,
    'Organic_carbon': 6.374070260129599,
    'Trihalomethanes': 103.37300526956227,
    'Turbidity': 4.506539591260472
}
response_2 = requests.post(potability_prediction_service_url, json=water_sample_2)
print("\nWater Sample 2 Prediction:")
print_prediction(response_2.json())
