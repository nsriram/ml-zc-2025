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

### Video 3 : Web Services: Introduction to Flask

----

[![Web Services: Introduction to Flask](https://img.youtube.com/vi/W7ubna1Rfv8/0.jpg)](https://www.youtube.com/watch?v=W7ubna1Rfv8&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=53)

- simple ping api using flask

```
from flask import Flask

app = Flask('ping')

@app.route('/ping', methods=['GET'])
def ping():
    return 'pong', 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Video 4 : Serving the Churn Model with Flask

----

[![Serving the Churn Model with Flask](https://img.youtube.com/vi/Q7ZWPgPnRz8/0.jpg)](https://www.youtube.com/watch?v=Q7ZWPgPnRz8&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=54)

- Create a web service that loads the chrun model and exposes an endpoint for predictions
- Webservice code [predict.py](predict.py)
- Code to test the webservice [predict-test.py](predict-test.py)
- gunicorn allows running the webservice in production mode
- Run the webservice using gunicorn e.g., `gunicorn --bind 0.0.0.0:8080 predict:app`


### Video 5 :  Python Virtual Environment: Pipenv

----

[![Python Virtual Environment: Pipenv](https://img.youtube.com/vi/BMXh8JGROHM/0.jpg)](https://www.youtube.com/watch?v=BMXh8JGROHM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=55)

- pipenv is used to create a virtual environment and manage dependencies
- Different services can have different dependencies and hence virtual environments are useful
- Tools in python ecosystem for virtual environments
   - venv (built-in) / virtualenv (out of the box from python)
   - conda
   - pipenv
   - poetry

### Video 6 :  Environment management with Docker

----

[![Environment management with Docker](https://img.youtube.com/vi/wAtyYZ6zvAs/0.jpg)](https://www.youtube.com/watch?v=wAtyYZ6zvAs&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=56)


### Video 7 : Deploying ML Models with FastAPI, UV

----

[![Deploying ML Models with FastAPI, UV](https://img.youtube.com/vi/jzGzw98Eikk/0.jpg)](https://www.youtube.com/watch?v=jzGzw98Eikk)
