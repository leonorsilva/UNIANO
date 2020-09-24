# UNIANO

## Requirements
Requires python 3.8 to be installed. All dependencies defined in requirements.txt. You can install them by:
```
 pip install -r requirements.txt
```
## Instalation
1-Create a folder for the project (e.g. proj)

2-Download or clone the UNIANO code under the project directory
```
proj
 -anomaly_detection
 --anomalies.pdf
 --anomalies.py
 --gui_utils.py
 --holtwinters.py
 --LSTM.py
 --predictors.py
 --sarima.py
 --series.csv
 --series.pdf
 --timer.csv
 ```
 3-Import the project into a Python IDE (such as PyCharm and Eclipse/PyDev)
 
 4-Run anomalies.py file to get the results
 
 ## Explanation
 
 The presented code takes information from series.csv and timer.csv files to detect anomalies from this files.
 It generates the:
 -series.pdf, which containes the series and the fitted sarima, holtwinters and lstm models.
 -anomalies.pdf, which containes the series and the detected anomalies.
 
