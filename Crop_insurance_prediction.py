import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

# Load your data
df = pd.read_csv('APY.csv')

# Clean column names by stripping leading/trailing spaces
df.columns = df.columns.str.strip()

# Assuming 'df' is your DataFrame and it's already loaded
# Preprocessing steps
X = df.drop(['Production', 'Yield'], axis=1)  # Features
y = df['Production']  # Target variable

# Handling missing values in the target variable y
y.fillna(y.mean(), inplace=True)

categorical_features = ['State', 'District', 'Crop', 'Season']
numerical_features = ['Crop_Year', 'Area']

# Creating a ColumnTransformer to apply transformations
# Note: Now using SimpleImputer with 'most_frequent' for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), categorical_features)
    ])

# Creating a pipeline that first transforms the data and then applies Gradient Boosting Regressor
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', GradientBoostingRegressor())])

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fitting the model
model.fit(X_train, y_train)

# Predicting on test data
y_pred = model.predict(X_test)

# Evaluating the model
rs = r2_score(y_test, y_pred)
print("r2_score:", rs)


def predict_production(model, input_data):
    return model.predict(input_data)

# Example of user input
user_input = {
    'State': input("Enter the State:\n"),
    'District': input("Enter the District:\n"),
    'Crop': input('Enter the Crop Type:\n'),
    'Crop_Year': int(input('Enter the Year:\n')),
    'Season': input('Enter the Season:\n'),
    'Area': float(input('Enter the area of cultivation:\n'))
}

# Predicting from user input
user_input_df = pd.DataFrame([user_input])
predicted_production = predict_production(model, user_input_df)

# Calculating production per unit area
production_per_area = predicted_production[0] / user_input['Area']
print("Yield:",production_per_area)
