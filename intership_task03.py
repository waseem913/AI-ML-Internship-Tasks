# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load Dataset
data = pd.read_csv("heart_disease_uci.csv")  # Load the Heart Disease dataset
df = pd.DataFrame(data)  # Convert to DataFrame
uncleaned_data = df.copy()  # Keep a copy of original data
print(uncleaned_data.isnull().sum())  # Check for missing values


# Handle Missing Values

# Fill missing numerical columns with mean
df['chol'].fillna(df['chol'].mean(), inplace=True)
df['trestbps'].fillna(df['trestbps'].mean(), inplace=True)
df['thalch'].fillna(df['thalch'].mean(), inplace=True)
df['oldpeak'].fillna(df['oldpeak'].mean(), inplace=True)
df['ca'].fillna(df['ca'].mean(), inplace=True)

# Fill missing categorical columns with mode
df['fbs'].fillna(df['fbs'].mode()[0], inplace=True)
df['restecg'].fillna(df['restecg'].mode()[0], inplace=True)
df['exang'].fillna(df['exang'].mode()[0], inplace=True)
df['slope'].fillna(df['slope'].mode()[0], inplace=True)
df['thal'].fillna(df['thal'].mode()[0], inplace=True)


# 3. Encode Categorical Variables

le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['thal'] = le.fit_transform(df['thal'])
df['cp'] = le.fit_transform(df['cp'])
df['restecg'] = le.fit_transform(df['restecg'])
df['slope'] = le.fit_transform(df['slope'])

# Keep a cleaned copy
cleaned_data = df.copy()
print(cleaned_data)


# Exploratory Data Analysis (EDA)

# Histogram for age distribution
plt.hist(df["age"], bins=5, color='purple', edgecolor='black')
plt.title('Age Distribution')
plt.show()

# Boxplot for resting blood pressure
sns.boxplot(y=df["trestbps"])
plt.title("Box Plot (Resting Blood Pressure)")
plt.show()

# Heatmap of correlations between numerical features
numerical_cols = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation (Numerical Columns Only)")
plt.show()

# Pairplot of selected features vs target
sns.pairplot(df[['age', 'chol', 'trestbps', 'num']], hue='num')
plt.title("Pairplot (ST Depression Induced by Exercise)")
plt.show()

# Violin plot of cholesterol vs target
sns.violinplot(x='num', y='chol', data=df)
plt.title("Violin Plot (Cholesterol vs Target)")
plt.show()

# Convert Target to Binary

# Heart disease presence: 0 = No, 1 = Yes (any value >0)
cleaned_data['num'] = cleaned_data['num'].apply(lambda x: 1 if x > 0 else 0)


# Define Features and Target

X = cleaned_data[['age','sex','cp','trestbps','chol','fbs','restecg',
                  'thal','exang','oldpeak','slope','ca','thalch']] 
y = cleaned_data['num']


# 7. Feature Scaling

scaler = StandardScaler()
X = scaler.fit_transform(X)
# Convert back to DataFrame for easier plotting later
X = pd.DataFrame(X, columns=['age','sex','cp','trestbps','chol','fbs','restecg',
                             'thal','exang','oldpeak','slope','ca','thalch'])

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ----------------------------
# Train Logistic Regression Model
# ----------------------------
model = LogisticRegression(max_iter=2000, solver='saga')  # Logistic Regression
model.fit(X_train, y_train)  # Train model


#  Make Predictions

predictions = model.predict(X_test)

#  Model Evaluation

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, predictions)
print("Confusion Matrix:\n", cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve and AUC
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC Curve (AUC = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


#  Feature Importance

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(feature_importance)

plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
plt.title("Feature Importance (Logistic Regression Coefficients)")

plt.show()
