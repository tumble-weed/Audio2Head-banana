Deployment of [Audio2Head](https://github.com/wangsuzhen/Audio2Head) on banana.dev

Structure is as suggested by https://github.com/bananaml/serverless-template/

- __server.py__ is to be left unchanged 
- __app.py__ does the request handling and response return. It contains a wrapper to call a function from the original repo. 
- __Dockerfile__ handles the deployment, downloading and moving relevant resources.
- __videos__ will contain all the driving videos that may be referred to in calls to the API  