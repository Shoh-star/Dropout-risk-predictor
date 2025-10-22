import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, r2_score, confusion_matrix
df = pd.read_csv('Dropout_Student.csv', sep=';')

# scaler = StandardScaler()
# List of columns to drop if they exist
optional_cols = ['Application mode',
                 "Mother's occupation", "Father's occupation", "Mother's qualification", "Father's qualification",
                 'Student_ID', 'Full_Name', 'Target', 'Marital status', 'Daytime/evening attendance\t',
                 'Age at enrollment', 'Displaced', 'Curricular units 2nd sem (enrolled)',
                 'Curricular units 1st sem (enrolled)', 'Previous qualification', 'Previous qualification (grade)',
                 'Debtor', 'Curricular units 2nd sem (credited)', 'Curricular units 1st sem (credited)',
                 'Gender', 'Nacionality', 'International', 'Educational special needs'
                 ]

# Filter only the columns that exist in the DataFrame
cols_to_drop = [col for col in optional_cols if col in df.columns]
smote = BorderlineSMOTE(random_state=42, k_neighbors=7)
# Drop safely
X = df.drop(cols_to_drop, axis=1)
y = (df['Target'] == 'Dropout').astype(int)
X_sampled, y_sampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_sampled, y_sampled, test_size=0.3, random_state=42)
# X_train_scaled = scaler.fit_transform(X_train_oversample)
# X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(
    n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
# model = KNeighborsClassifier(n_neighbors=5)
# model = BernoulliNB()
# model = LogisticRegression(class_weight='balanced', penalty='l2', solver='liblinear', max_iter=1000, random_state=42)
model.fit(X_train, y_train)
# columns = pd.DataFrame({'Feature': X.columns})
prediction = model.predict(X_test)
prediction2 = model.predict(X_train)
# print('Test Accuracy: ', accuracy_score(y_test, prediction))
# print('Train Accuracy: ', accuracy_score(y_train_oversample, prediction2))
print(classification_report(y_test, prediction))
print(classification_report(y_train, prediction2))
# print(columns)
# Model is ready for deployment
joblib.dump(model, "dropout_model.pkl")
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "model_features.pkl")
