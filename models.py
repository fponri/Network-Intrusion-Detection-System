from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def initialize_random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42)

def initialize_logistic_regression():
    return LogisticRegression(max_iter=1000)