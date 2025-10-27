## Week 5 Notes : Deployment
Practice jupyter notebooks: 
   1. [Deployment](week5-deployment.ipynb)
   2. [Load Model](week5-load-model.ipynb)


### Video 1 :  Intro / Session Overview 

----

[![Intro / Session Overview ](https://img.youtube.com/vi/agIFak9A3m8/0.jpg)](https://www.youtube.com/watch?v=agIFak9A3m8&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=51)

1. Model is trained using jupyter notebook
2. Model is saved to a file e.g., model.bin
3. Webservice 'churn-service' loads the model from model.bin and exposes an API endpoint
4. Client application will invoke the churn service API endpoint to get predictions
   - e.g., Marketing service will call the churn-service API to get predictions and decides whether the customer should be sent an offer or not

- Docker will be used for containerization of the webservice
- pipenv will be used for the churn service to add dependencies

### Video 2 :  Model Serialization (Save and Load)

----

[![Model Serialization (Save and Load)](https://img.youtube.com/vi/EJpqZ7OlwFU/0.jpg)](https://www.youtube.com/watch?v=EJpqZ7OlwFU&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=52)

- Use pickle library to save and load the model
- Saving the model
```
import pickle
output_file = f'model_C={C}.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
    f_out.close()
```

- Loading the model
```
import pickle
input_file = 'model_C=1.0.bin'
with open(input_file, 'rb') as f_in: 
    dv, model = pickle.load(f_in)
```
- The loaded model can be tested with a sample customer
- 
