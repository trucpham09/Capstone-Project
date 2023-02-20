import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.model_selection import train_test_split
import time
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor



# read data
train_data = pd.read_csv("Train.csv")

# Create function to drop columns with high missing rate ()
def drop_missing_columns (data, missing_threshold = 0.98):
    data_copy = data.copy()
    missing_columns = []
    number_of_row = data_copy.shape[0]
    for col in data_copy.columns:
        # calculate percentage of missing row in each column
        missing_row = data_copy[col].isnull().sum()
        missing_frac = missing_row / float(number_of_row)
        if missing_frac > missing_threshold:
            missing_columns.append(col)
    keep_columns = [col for col in data_copy.columns if col not in missing_columns]   
    keep_data = data_copy[keep_columns].copy()
    return keep_data, missing_columns

keep_data, missing_columns = drop_missing_columns(train_data,missing_threshold = 0)

# Assign missing value as property not having the feature
values = {
    "LotFrontage": 0,
    "Alley": "No alley access",
    "MasVnrType": "No Veneer",
    "MasVnrArea": 0,
    "BsmtQual": "No Basement",
    "BsmtCond": "No Basement",
    "BsmtExposure": "No Basement",
    "BsmtFinType1": "Not Applicable",
    "BsmtFinType2": "Not Applicable",
    "FireplaceQu": "No Fireplace",
    "GarageType": "No Garage",
    "GarageYrBlt": "None built",
    "GarageFinish": "No Garage",
    "GarageQual": "No Garage",
    "GarageCond": "No Garage",
    "PoolQC": "No Pool",
    "Fence": "No Fence",
    "MiscFeature": "No Feature",    
    "GarageArea": 0,
}

# fill missing data with assigned values

train_data.fillna(value=values, inplace=True)

# fill missing data for object type columns
features = train_data.select_dtypes(include=["object"]).columns
for features in features:
    train_data[features].fillna(train_data[features].mode()[0], inplace=True)

# change basement values from NANs to 0
Basementmetrics = [
    "BsmtHalfBath",
    "BsmtFullBath",
    "BsmtFinSF1",
    "GarageCars",
    "TotalBsmtSF",
    "BsmtUnfSF",
    "BsmtFinSF2",
]

for Basementmetrics in Basementmetrics:
    train_data.loc[(train_data[Basementmetrics].isnull()), Basementmetrics] = 0


# Change variable types

# Convert YearBuilt, YrSold, GarageYrBlt, and YearRemodAdd to category variables

train_data.YearBuilt = train_data.YearBuilt.astype(str)

train_data.YrSold = train_data.YrSold.astype(str)

train_data.GarageYrBlt = train_data.GarageYrBlt.astype(str)

train_data.YearRemodAdd = train_data.YearRemodAdd.astype(str)


# Decode MSSUbCLass, Overallcond, and OverallQual to string values
MSSUbCLass = {
    20: "1-STORY 1946 & NEWER ALL STYLES",
    30: "1-STORY 1945 & OLDER",
    40: "1-STORY W/FINISHED ATTIC ALL AGES",
    45: "1-1/2 STORY - UNFINISHED ALL AGES",
    50: "1-1/2 STORY FINISHED ALL AGES",
    60: "2-STORY 1946 & NEWER",
    70: "2-STORY 1945 & OLDER",
    75: "2-1/2 STORY ALL AGES",
    80: "SPLIT OR MULTI-LEVEL",
    85: "SPLIT FOYER",
    90: "DUPLEX - ALL STYLES AND AGES",
    120: "1-STORY PUD (Planned Unit Development) - 1946 & NEWER",
    150: "1-1/2 STORY PUD - ALL AGES",
    160: "2-STORY PUD - 1946 & NEWER",
    180: "PUD - MULTILEVEL - INCL SPLIT LEV/FOYER",
    190: "2 FAMILY CONVERSION - ALL STYLES AND AGES",
}


OverallQualCond = {
    10: "Very Excellent",
    9: "Excellent",
    8: "Very Good",
    7: "Good",
    6: "Above Average",
    5: "Average",
    4: "Below Average",
    3: "Fair",
    2: "Poor",
    1: "Very Poor",
}


train_data.replace(
    {
        "OverallQual": OverallQualCond,
        "OverallCond": OverallQualCond,
        "MSSubClass": MSSUbCLass,
    },
    inplace=True,
)

# Visualization
# Plot distributions of sale price 
sns.set_style("whitegrid")
sns.histplot(train_data.SalePrice, kde = True)
plt.show()

# Plot log of sale price
train_data["SalePrice_log"] = np.log(train_data.SalePrice)
sns.histplot(train_data.SalePrice_log, kde=True)
plt.show()

# Plot the heatmap: the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(train_data.corr())

# Omit columns that have correlation with SalePrice < 0.15
omit_columns = []
corr_mat = train_data.corr()
numeric_features = train_data.select_dtypes(include=np.number).columns
for col in numeric_features:
    if abs(corr_mat[col]["SalePrice"]) < 0.15:
        omit_columns.append(col)
        
# Add SalePrice and SalePrice_log to omit_columns
omit_columns.extend(["SalePrice", "SalePrice_log"])

# Encode categorical variables
class OneHotEncoder(SklearnOneHotEncoder):
    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)
        self.fit_flag = False

    def fit(self, X, **kwargs):
        out = super().fit(X)
        self.fit_flag = True
        return out

    def transform(self, X, **kwargs):
        sparse_matrix = super(OneHotEncoder, self).transform(X)
        new_columns = self.get_new_columns(X=X)
        d_out = pd.DataFrame(
            sparse_matrix.toarray(), columns=new_columns, index=X.index
        )
        return d_out

    def fit_transform(self, X, **kwargs):
        self.fit(X)
        return self.transform(X)

    def get_new_columns(self, X):
        new_columns = []
        for i, column in enumerate(X.columns):
            j = 0
            while j < len(self.categories_[i]):
                new_columns.append(f"{column}_<{self.categories_[i][j]}>")
                j += 1
        return new_columns




# create function to encode categorical columns and concatinate with numeric columns
def transform(data, df):

    # select categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    cat_df = df[cat_cols]

    # select numeric columns
    num_df = df.select_dtypes(include=np.number)

    # create one hot encoder 
    onehot = OneHotEncoder(categories="auto", handle_unknown="ignore")
    # fit the endcoder to data
    onehot.fit(data[cat_cols])

    # transform categorical columns
    df_encoded = onehot.transform(cat_df)

    # concatinate numeric columns with categorical columns
    df_encoded_full = pd.concat([df_encoded, num_df], axis=1)

    return df_encoded_full

# Transform data to numeric form
train_encoded = transform(train_data, train_data)

# add log sale price column
train_encoded["SalePrice_log"] = np.log(train_encoded.SalePrice)

# split train.csv into training and test set
X = train_encoded.drop(columns=omit_columns)
y = train_encoded.SalePrice_log

X_train, X_test, y_train , y_test = train_test_split(X, y, test_size=0.2)


# Create function to tune the model with GridSearchCV and cv = 5
def tune_pipeline(model, param_grid, scoring, model_name):
    tuned_model = GridSearchCV(
        estimator=model, param_grid=param_grid, scoring=scoring, cv=5
    )

    # Fit model and calculate the time of the tunning process 
    print("Tunning the" , model_name, "model")
    start_time = time.process_time()

    tuned_model.fit(X_train, y_train)

    # Calculate the tunning time
    print(time.process_time() - start_time, "seconds")
    print("Finished tunning",model_name,"model")
    return tuned_model



# Model 1: Decision tree
# create parameter grid for Grid Search CV
param_grid = {
    "max_depth": np.arange(5,35,5), 
    "max_features": np.arange(0.1, 0.8,0.1)
}
tree_model = DecisionTreeRegressor(criterion="squared_error", random_state=0)

tree_tuned_model = tune_pipeline(tree_model, param_grid, "neg_mean_absolute_error", "Decision Tree")
print("Best parameters of Decision Tree model:",tree_tuned_model.best_params_)
print("Best score of Decision Tree model:", tree_tuned_model.best_score_)
# Model 2: Random Forest
# create parameter grid for Grid Search CV
param_grid = {
    "n_estimators": np.arange(100,700,100),
    "max_features": np.arange(0.1, 0.8,0.1),
}
# Initialise the random forest model
rand_forest_reg = RandomForestRegressor(n_jobs=-1, random_state=0, bootstrap=True)

rand_forest_tuned_model = tune_pipeline(rand_forest_reg, param_grid, "neg_mean_absolute_error", "Random Forest")

print("Best parameters of Random Forest model:",rand_forest_tuned_model.best_params_)
print("Best score of Random Forest model:", rand_forest_tuned_model.best_score_)
# Model 3: Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor

# create parameter grid for Grid Search CV
param_grid = {
    "n_estimators": np.arange(100,700,100),
    "max_depth": np.arange(5,35,5),
    "learning_rate": [0.01, 0.05, 0.1],
}
gra_boost = GradientBoostingRegressor(random_state=0, max_features="sqrt")

gra_boost_tuned_model = tune_pipeline(gra_boost, param_grid, "neg_mean_absolute_error","Gradient Boosting")

print("Best parameters of Gradient Boosting model:",gra_boost_tuned_model.best_params_)
print("Best score of Gradient Boosting model:", gra_boost_tuned_model.best_score_)

