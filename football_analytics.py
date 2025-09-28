import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import xgboost
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df_players = pd.read_csv('D://New folder//New folder (7)//final_processed_data.csv', index_col=0)

# Display basic information
print(df_players.head())
print(df_players.describe())
print(df_players.info())


# Drop rows with missing values in critical columns
df_players = df_players.dropna(subset=['contract_expires', 'foot', 'height', 'price', 'max_price'])
print(df_players.shape)

#How much is the highest valued soccer player in EPL priced?
sns.barplot(x='league', y='price', data= df_players)


# Replace NaN values with 'unknown'
df_players.fillna('unknown', inplace=True)

# Prepare features and target
df_target = df_players[['price']]
df_features = df_players[['age', 'height', 'league', 'foot', 'position', 'club',
                          'contract_expires', 'joined_club', 'player_agent', 'outfitter', 'nationality']]

# One Hot Encoding
columns_to_encode = ['league', 'foot', 'position', 'club', 'contract_expires',
                     'joined_club', 'player_agent', 'outfitter', 'nationality']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(handle_unknown='ignore'), columns_to_encode)],
                       remainder='passthrough')
df_features_encoded = ct.fit_transform(df_features)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(df_features_encoded, df_target, test_size=0.3, random_state=22)

# Convert DataFrame to NumPy array and flatten
y_train, y_test = y_train.values.flatten(), y_test.values.flatten()

print(f'x_train: {x_train.shape}, x_test: {x_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}')


# Hyperparameter tuning for XGBoost
param_grid = {
    'nthread': [4],
    'objective': ['reg:squarederror'],
    'learning_rate': [0.03, 0.05],
    'max_depth': [4, 7],
    'min_child_weight': [2, 3, 4],
    'subsample': [0.3, 0.5],
    'colsample_bytree': [0.7],
    'n_estimators': [300]
}

xgb = xgboost.XGBRegressor(objective='reg:squarederror')
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=4, verbose=1)
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Visualizing parameter tuning results
plt.figure(figsize=(10, 6))
plt.plot(range(len(grid_search.cv_results_['mean_test_score'])), grid_search.cv_results_['mean_test_score'], marker='o')
plt.xlabel('Parameter Combination')
plt.ylabel('Negative RMSE')
plt.title('Grid Search Results')
plt.xticks(range(len(grid_search.cv_results_['params'])), grid_search.cv_results_['params'], rotation=90)
plt.show()

# Train final model
best_xgb = xgboost.XGBRegressor(**best_params)
best_xgb.fit(x_train, y_train)

# Predictions and evaluation
pred = best_xgb.predict(x_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

#we are going to create a model that determines how good the metrics can be in predicting a player's market value.
Top_five_leagues_players = df_players.bfill(axis=0).fillna(0)

target = Top_five_leagues_players[['price']]
features = Top_five_leagues_players[['age', 'height', 'league','foot', 'position', 'club',
                        'contract_expires', 'joined_club', 'player_agent', 'outfitter', 'nationality']]

columns_to_encode = ['league' ,'foot', 'position', 'club', 'contract_expires', 'joined_club', 'player_agent', 'outfitter', 'nationality']

features.loc[:, columns_to_encode] = features.loc[:, columns_to_encode].astype(str)

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), columns_to_encode)], remainder='passthrough')
features_encoded = ct.fit_transform(features)

x_train, x_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

model = LinearRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions, squared=False)

print('Mean Absolute Error:', mae)
print('Mean Squared Error:', mse)

print('Root Mean Squared Error:', rmse)


# Train Decision Tree Regression
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(x_train, y_train)
dt_pred = dt_regressor.predict(x_test)

# Evaluate Decision Tree Regressor
dt_mae = mean_absolute_error(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = mean_squared_error(y_test, dt_pred, squared=False)

print('\n--- Decision Tree Regressor ---')
print(f'Mean Absolute Error: {dt_mae}')
print(f'Mean Squared Error: {dt_mse}')
print(f'Root Mean Squared Error: {dt_rmse}')


# Train Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(x_train, y_train)
rf_pred = rf_regressor.predict(x_test)

# Evaluate Random Forest Regressor
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)

print('\n--- Random Forest Regressor ---')
print(f'Mean Absolute Error: {rf_mae}')
print(f'Mean Squared Error: {rf_mse}')
print(f'Root Mean Squared Error: {rf_rmse}')


# --- Logistic Regression (Converting Price into Categories) ---

# Convert price into 3 bins: Low, Medium, High
bin_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
y_binned = bin_discretizer.fit_transform(target).ravel()

# Train-test split with classified target
x_train_log, x_test_log, y_train_log, y_test_log = train_test_split(features_encoded, y_binned, test_size=0.2, random_state=42)

# Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(x_train_log, y_train_log)
log_pred = log_reg.predict(x_test_log)

# Evaluate Logistic Regression
log_accuracy = log_reg.score(x_test_log, y_test_log)
log_precision = precision_score(y_test_log, log_pred, average='weighted')
log_recall = recall_score(y_test_log, log_pred, average='weighted')
log_f1 = f1_score(y_test_log, log_pred, average='weighted')
log_conf_matrix = confusion_matrix(y_test_log, log_pred)

print('\n--- Logistic Regression (Categorical Price Prediction) ---')
print(f'Accuracy: {log_accuracy:.4f}')
print(f'Precision: {log_precision:.4f}')
print(f'Recall: {log_recall:.4f}')
print(f'F1 Score: {log_f1:.4f}')

# Display confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(log_conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Compute ROC curve and AUC
y_prob = log_reg.predict_proba(x_test_log)[:, 1]  # Get probability estimates for positive class
fpr, tpr, _ = roc_curve(y_test_log, y_prob, pos_label=1)
roc_auc = roc_auc_score(y_test_log, log_reg.predict_proba(x_test_log), multi_class='ovr')

# Plot ROC Curve
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc="lower right")
plt.show()

from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

# Convert continuous price predictions into categories (Low, Medium, High)
bin_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
y_train_class = bin_discretizer.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_class = bin_discretizer.transform(y_test.reshape(-1, 1)).ravel()

# --- Train Classification Models (Decision Tree, Random Forest, XGBoost, Logistic Regression) ---
dt_clf = DecisionTreeClassifier(random_state=42)
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', random_state=42)

# Fit models
dt_clf.fit(x_train, y_train_class)
rf_clf.fit(x_train, y_train_class)
xgb_clf.fit(x_train, y_train_class)

# Make Predictions
dt_pred = dt_clf.predict(x_test)
rf_pred = rf_clf.predict(x_test)
xgb_pred = xgb_clf.predict(x_test)

# Store models & predictions
models = {
    "Decision Tree": (dt_clf, dt_pred),
    "Random Forest": (rf_clf, rf_pred),
    "XGBoost": (xgb_clf, xgb_pred),
}


# Function to compute & print evaluation metrics
def evaluate_model(name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"\n--- {name} ---")
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Confusion Matrix Visualization
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

    return accuracy, precision, recall, f1, conf_matrix


# Compute and display metrics for all models
results = {}
for model_name, (model, y_pred) in models.items():
    results[model_name] = evaluate_model(model_name, y_test_class, y_pred)

# --- ROC Curves ---
plt.figure(figsize=(8, 6))

for model_name, (model, y_pred) in models.items():
    y_prob = model.predict_proba(x_test)  # Get probability estimates
    fpr, tpr, _ = roc_curve(y_test_class, y_prob[:, 1], pos_label=1)
    roc_auc = roc_auc_score(y_test_class, y_prob, multi_class='ovr')
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random guess line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Classification Models')
plt.legend(loc="lower right")
plt.show()