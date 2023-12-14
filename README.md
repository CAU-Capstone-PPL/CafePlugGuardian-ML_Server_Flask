## CafePlugGuardian Project
### Introduction
CafePlugGuardian is a capstone design project at the Chung-Ang University's Department of Software Engineering.

Please note that we will not be accepting contributions for CafePlugGuardian, as it is a Capstone Design Project.

#### The Goal of CafePlugGuardian Project
1. The pin number allows only cafe customers to use the plug, preventing unauthorized use of the plug.
2. Limit the amount of electricity to restrict customers who use excessive power or stay for long periods of time.
3. By analyzing the current patterns of devices in use, devices not permitted in the cafe, such as smartphones and laptop chargers, are automatically blocked through machine learning.

### Structure of CafePlugGuardian
<img width="80%" src="https://github.com/CAU-Capstone-PPL/CafePlugGuardian-Server/assets/55429793/74940115-831a-49f7-ab9a-3d5dc402089a"/>

### Sub Projects of CafePlugGuardian
* [CafePlugGuardian-Client](https://github.com/CAU-Capstone-PPL/CafePlugGuardian-Client)
    * Cafe Manager App - flutter app
* [CafePlugGuardian-WebClient](https://github.com/CAU-Capstone-PPL/CafePlugGuardian-WebClient)
    * Cafe Customer Web - flutter web
* [CafePlugGuardian-Server](https://github.com/CAU-Capstone-PPL/CafePlugGuardian-Server)
    * Backend server - express.js
* [CafePlugGuardian-Hardware](https://github.com/CAU-Capstone-PPL/CafePlugGuardian-Hardware)
    * SmartPlug embedded system - arduino(tasmota open source)
* [CafePlugGuardian-ML](https://github.com/CAU-Capstone-PPL/CafePlugGuardian-ML)
    * AI model - pytorch, GRU model
* [CafePlugGuardian-ML_Server_Flask](https://github.com/CAU-Capstone-PPL/CafePlugGuardian-ML_Server_Flask)
    * AI server - flask


## Introduction
CafePlugGuardian-ML_Server_FLask is a Server for active Machine Learning Model. Our Machine Learning Model is wrote by Pytorch. So we have to make Python Server for active our model. 
It was wrote by python version 3.10.6. And use Flask 3.0.0.

### Requirements to run or modifiy server
* Python version more than 3.10.6
* Flask 3.0.0
* Ide which you can edit python code.
* Torch version 2.1.1
* Open your port 5000 or change port num which you will use

### How to run a flask server
Grab a git clone from the desired location and run it by typing python app.py in the terminal.

### How to categorize the data you receive.
When the data is sent in JSON from the backend server, it fetches the training model stored in the same location as the server for classification.

### How to modify criteria
![image](https://github.com/CAU-Capstone-PPL/CafePlugGuardian-ML_Server_Flask/assets/106421292/1ff4d904-d10d-44f1-b280-8c7f4017ade7)<br>
You can arbitrarily modify the disallow criteria in the above code. The range can be modified to any value between 468 and 0.

### Prediction result
The prediction result is in the form of [x,468-x], and one data feature is moved to the class with the higher similarity by comparing the similarity between the disallowed data and the allowed data. A total of 468 data features are moved and output in the form above. After classification, it returns the results to the backend server.

#### Example
![image](https://github.com/CAU-Capstone-PPL/CafePlugGuardian-ML_Server_Flask/assets/106421292/6f0c1738-3698-4dd4-adac-934500f1ed08)

### How to chage your own end point
Line 48 initially end point is '/predict2'. You can change your own end point such as '/predict', '/AI' and others.

## License
This program is licensed under MIT
