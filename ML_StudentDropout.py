import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, r2_score, confusion_matrix
df = pd.read_csv('Dropout_Student.csv', sep=';')

scaler = StandardScaler()
X = df.drop(["Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification", 'Student_ID', 'Full_Name', 'Target', 'Marital status',
            'Daytime/evening attendance\t', 'Age at enrollment', 'Displaced', 'Curricular units 2nd sem (enrolled)', 'Curricular units 1st sem (enrolled)', 'Previous qualification', 'Previous qualification (grade)', 'Debtor', 'Curricular units 2nd sem (credited)', 'Curricular units 1st sem (credited)', 'Gender', 'Nacionality', 'International', 'Educational special needs'], axis=1)
y = (df['Target'] == 'Dropout').astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42)
smote = SMOTE(random_state=42)
X_train_oversample, y_train_oversample = smote.fit_resample(X_train, y_train)
X_train_scaled = scaler.fit_transform(
    X_train_oversample)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=79, max_depth=10,
                               class_weight='balanced', min_samples_split=4, max_leaf_nodes=15, random_state=42)
# model = KNeighborsClassifier(n_neighbors=5)
# model = BernoulliNB()
# model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train_scaled, y_train_oversample)
# columns = pd.DataFrame({'Feature': X.columns})
prediction = model.predict(X_test_scaled)
prediction2 = model.predict(X_train_scaled)
# print('Test Accuracy: ', accuracy_score(y_test, prediction))
# print('Train Accuracy: ', accuracy_score(y_train_oversample, prediction2))
print(classification_report(y_test, prediction))
print(classification_report(y_train_oversample, prediction2))
# print(columns)
# Model is ready for deployment
joblib.dump(model, "dropout_model.pkl")
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "model_features.pkl")

