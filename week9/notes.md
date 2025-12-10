## Week 9 Notes : Serverless and Model Deployment
Practice jupyter notebooks:
- [Week 9 Homework Jupyter Notebook](week9-serveless.ipynb)

### Video 1 :  Introduction to Serverless

----
[![Introduction to Serverless](https://img.youtube.com/vi/JLIVwIsU6RA/0.jpg)](https://www.youtube.com/watch?v=JLIVwIsU6RA&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=82&pp=iAQB)

- We will use AWS lambda for deploying the models. The model classifying the images of clothing items will be deployed

### Video 2 : AWS Lambda

----
[![AWS Lambda](https://img.youtube.com/vi/_UX8-2WhHZo/0.jpg)](https://www.youtube.com/watch?v=_UX8-2WhHZo&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=84)

- Create a basic python lambda function that returns "PONG" when invoked
- Example code

```python
import json 
def lambda_handler(event, context):
    print("parameters:", event)
    return "PONG"
```

- In the above function, 
  - `event` contains the input parameters passed to the function when invoked 
  - `context` contains runtime information about the function
  - The function returns "PONG" when invoked


### Video 3 : Workshop - Deploying a Model using AWS Lambda

----
[![Workshop - Deploying a Model using AWS Lambda](https://img.youtube.com/vi/sHQaeVm5hT8/0.jpg)](https://www.youtube.com/watch?v=sHQaeVm5hT8&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=84)

This module will cover the following :
1. Deploying scikit learn model using AWS Lambda
2. Using ONNX for keras and tensorflow models
3. ONNX for PyTorch models

#### Deploying scikit learn model using AWS Lambda
- [train.py](serverless-workshop/train/train.py) for a logistic regression model from week5 will be used
- Model binary (model.bin) is created by running uv as below
    - `uv run python week9/serverless-workshop/lambda-sklearn/train/train.py`
- The lambda function code is in [lambda_function.py](serverless-workshop/lambda-sklearn/lambda_function.py) will 
load the model binary and use it for inference
- It can be invoked from CLI as below
```
aws lambda invoke \
  --function-name churn-prediction \
  --cli-binary-format raw-in-base64-out \
  --payload file://week9/serverless-workshop/lambda-sklearn/customer.json \
  output.json
```
- [invoke.py](serverless-workshop/lambda-sklearn/invoke.py) script can be used to invoke the lambda function programmatically
by using the boto3 library
  - ` uv run python week9/serverless-workshop/lambda-sklearn/invoke.py`

#### Docker for lambda function
- Running the lambda function will require its dependencies. Docker image will be created with all the dependencies and deployed
- Dockerfile is in [Dockerfile](serverless-workshop/lambda-sklearn/Dockerfile) has the installation steps
  - lambda function code, model binary are the important files
  - uv is used for installing dependencies using the requirements.txt created from [pyproject.toml](serverless-workshop/lambda-sklearn/pyproject.toml)
- Build and Run the docker image
- Test the local docker image using [test.py](serverless-workshop/lambda-sklearn/test.py)
  - `uv run python week9/serverless-workshop/lambda-sklearn/test.py`