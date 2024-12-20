import pandas as pd
import pickle


from sklearn.preprocessing import LabelEncoder, StandardScaler

# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data_url = "train_dataset.csv"
data = pd.read_csv(data_url)

# # Inspect the data
# print("Data Head:\n", data.head())
# print("\nData Info:\n")
# data.info()
# print("\nMissing Values:\n", data.isnull().sum())

# Creating a dictionary of label encoder instance of each column
col_to_encode = ["Gender", "Married", "Dependents", "Education", "Self_Employed"]
label_encoder = {}
for col in col_to_encode:
    label_encoder[col] = LabelEncoder()
    label_encoder[col].fit(data[col])


# Data preprocessing
def preprocess_data(data, encoder=label_encoder):
    # Dropping irrelevant columns
    data = data.drop(["Loan_ID", "Property_Area"], axis=1)

    # dropping missing values
    data = data.dropna()

    # Encoding categorical variables
    for col, encoder in label_encoder.items():
        data[col] = encoder.transform(data[col])
    # le = LabelEncoder()
    # data["Gender"] = le.fit_transform(data["Gender"])
    # data["Married"] = le.fit_transform(data["Married"])
    # data["Dependents"] = le.fit_transform(data["Dependents"])
    # data["Education"] = le.fit_transform(data["Education"])
    # data["Self_Employed"] = le.fit_transform(data["Self_Employed"])

    data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

    return data


# Apply preprocessing
processed_data = preprocess_data(data)

# Define features and target
X = processed_data.drop("Loan_Status", axis=1)
y = processed_data["Loan_Status"]

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# Standardizing the features
scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
X_scaled = scaler.fit_transform(X)

# Build the model
rf_model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_model.fit(X_scaled, y)

# Predictions
# y_pred = rf_model.predict(X_test)

# # Evaluate the model
# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# saving the model as .pkl file
with open("loan_clf_with_encoders.pkl", "wb") as f:
    pickle.dump((rf_model, label_encoder, scaler), f)
