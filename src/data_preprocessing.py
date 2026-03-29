import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df=pd.read_csv(path)
    x=df.drop("condition", axis=1)
    y=df["condition"]
    return x,y
def preprocess_data(x,y):
    scaler=StandardScaler()
    x_scaled=scaler.fit_transform(x)
    x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    return train_test_split(x_scaled,y,test_size=0.2,random_state=42)
