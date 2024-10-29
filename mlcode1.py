'''import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Traffic flow model training
def train_traffic_model():
    data = pd.read_csv('traffic_flow_data.csv')
    X = data[['source', 'target', 'speed', 'congestion_factor']]
    y = data['vehicles']
    clf_traffic = DecisionTreeClassifier(random_state=100)
    clf_traffic.fit(X, y)
    joblib.dump(clf_traffic, 'traffic_flow_model.pkl')

# Transport mode model training
def train_transport_mode_model():
    data = pd.read_csv('finrecord.csv')
    X = data[['duration', 'vehicles']]
    y = data['mode_of_transport']
    clf_transport_mode = DecisionTreeClassifier(random_state=100)
    clf_transport_mode.fit(X, y)
    joblib.dump(clf_transport_mode, 'transport_mode_model.pkl')

# Call these once to train and save models
train_traffic_model()
train_transport_mode_model()

# Inspect features used in the saved models'''
import joblib
import pandas as pd
# Load models
clf_traffic = joblib.load('traffic_flow_model.pkl')
clf_transport_mode = joblib.load('transport_mode_model.pkl')

import joblib

def mypredict(s, t, sp, c, ti):
    # Load trained models
    clf_traffic = joblib.load('traffic_flow_model.pkl')
    clf_transport_mode = joblib.load('transport_mode_model.pkl')
    
    # Create input data for the traffic model with correct column names
    traffic_input = pd.DataFrame([[s, t, sp, c]], columns=clf_traffic.feature_names_in_)
    veh = int(clf_traffic.predict(traffic_input)[0])

    # Create input data for the transport mode model with exact column order and names
    transport_input = pd.DataFrame([[ti, veh]], columns=clf_transport_mode.feature_names_in_)
    nv = clf_transport_mode.predict(transport_input)

    return nv[0]

# Test the function
d = mypredict(20, 30, 6, 2, 45)
print(d)
