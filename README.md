# 🎓 Student Dropout Risk Predictor

This app helps university administrators identify students at risk of dropping out using machine learning.

## Features
- Upload student data via CSV(your file shuld be saved as 'CSV (comma delimited)')
- Predict dropout risk using a trained model
- View and download high-risk student reports

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`

## Deployment
This app is ready for Streamlit Cloud. Just connect this repo and select `app.py` as the entry point.
## CSV column requirements:
Your CSV file's columns must EXACTLY MATCH the following columns:
1.                                       Student_ID
2.                                       Full_Name
3.                                Application order
4.                                           Course
5.                                  Admission grade
7.                          Tuition fees up to date
8.                               Scholarship holder
9.           Curricular units 1st sem (evaluations)
10.              Curricular units 1st sem (approved)
11.                Curricular units 1st sem (grade)
12.  Curricular units 1st sem (without evaluations)
13.          Curricular units 2nd sem (evaluations)
14.             Curricular units 2nd sem (approved)
15.                Curricular units 2nd sem (grade)
16. Curricular units 2nd sem (without evaluations)
17.                               Unemployment rate
18.                                  Inflation rate
19.                                             GDP
20.                                         Target
❗️**Important:** NOTE THAT YOUR FILE SHOULD NOT CONTAIN MORE OR LESS COLUMNS THAN WHAT WAS GIVEN ABOVE!
## Additional notes:
GDP, Unemployment rate, Inflation rate means you have to enter the values of mentioned fetaures that YOUR country has.
