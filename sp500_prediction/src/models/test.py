from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load and split the data
X, y = load_iris(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    eval_metric='mlogloss',  # ✅ goes in the constructor
    importance_type='gain'
)

# Print class info
print("XGBClassifier type:", type(model))
print("XGBClassifier source:", model.__module__)

# Fit the model
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

