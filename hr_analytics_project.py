import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report

# Step 1: Load the data
df = pd.read_csv("hr_data.CSV.csv")

# Step 2: Encode categorical columns
le = LabelEncoder()
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    if col != 'Attrition':  # We'll handle Attrition separately
        df_encoded[col] = le.fit_transform(df_encoded[col])

# Step 3: Prepare features (X) and target (y)
X = df_encoded.drop("Attrition", axis=1)
y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert Yes/No to 1/0

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Logistic Regression
model = LogisticRegression(solver='saga',max_iter=2000)  # Increased iterations
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Step 7: Output results
print(classification_report(y_test, y_pred))

import pandas as pd

# Use absolute value of coefficients (because negative just means inverse effect)
feature_importance = pd.Series(abs(model.coef_[0]), index=X.columns)

# Sort in descending order
feature_importance = feature_importance.sort_values(ascending=False)

# Save to a CSV file in your project folder
feature_importance.to_csv("feature_importance.csv")

print("Top 3 Attrition Drivers:")
print(feature_importance.head(3))
print(feature_importance.tail(3))
print(df.isnull().sum())


