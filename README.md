# Heart Condition Checker
According to the CDC, there are several key factors that overwhelminghly influence the likelihood of heart disease. They write: About half of all Americans (47%) have at least 1 of 3 key risk factors for heart disease: high blood pressure, high cholesterol, and smoking.. Majore health factors include the following:

high blood pressure,
high blood cholesterol levels,
diabetes mellitus,
obesity.
Heart disease also depends on habits and behaviors. Here, the CDC lists the following:

eating a diet high in saturated fats, trans fat, and cholesterol,
not getting enough physical activity,
drinking too much alcohol,
tobacco use,
Also, the higher the age, the risk of the disease increases. It predominates in most ethnic groups (African Americans, American Indians and Alaska Natives), while in others it gives way to cancer (Asian Americans and Pacific Islanders and Hispanics).

According to the aforementioned information, variables were isolated from the dataset first, whose scientific confirmation attests to a high impact on heart disease. After these were extracted and converted, other variables that do not have a leading effect on heart disease but may indirectly lead to it were included in the final dataset.

The app created with Python to predict person's heart health condition based on well-trained machine learning model (logistic regression).



## Table of Contents
1. [General info](#general-info)
2. [Technologies](#technologies)
3. [Installation](#installation)


## General info
In this project, logistic regression was used to predict person's heart health condition expressed as a dichotomous variable (heart disease: yes/no). The model was trained on approximately 70,000 data from an annual telephone survey of the health of U.S. residents from the year 2020. The dataset is publicly available at the following link: https://www.cdc.gov/brfss/annual_data/annual_2020.html. The data is originally stored in SAS format. The original dataset contains approx. 400,000 rows and over 200 variables.This project contains:
* the app - the application construct is located in the `app.py` file. This file uses data from the `data` folder and saved (previously trained) ML models from the `model` folder.

The logistic regression model was found to be satisfactorily accurate (accuracy approx. 80%).

## Technologies
The app is fully written in Python 3.9.9. `streamlit 1.8.0` was used to create the user interface, and the machine learning itself was designed using the module `scikit-learn 1.0.2`. `pandas 1.41.`, `numpy 1.22.2` and `polars 0.13.0` were used to perform data converting operations.

## Installation
The project was uploaded to the web using heroku. You can use it online at the following link: https://heartdisease-check.herokuapp.com/. If you want to use this app on your local machine, make sure that you have installed the necessary modules in a version no smaller than the one specified in the `requirements.txt` file. You can either install them globally on your machine or create a virtual environment (`pipenv`), which is highly recommended.
1.  Install the packages according to the configuration file `requirements.txt`.
```
pip install -r requirements.txt
```

2.  Ensure that the `streamlit` package was installed successfully. To test it, run the following command:
```
streamlit hello
```
If the example application was launched in the browser tab, everything went well. You can also specify a port if the default doesn't respond:
```
streamlit hello --server.port port_number
```
Where `port_number` is a port number (8889, for example).

3.  To start the app, type:
```
streamlit run app.py
```

And that's it! Now you can predict your heart health condition expressed as a binary variable based on a dozen factors that best describe you.
