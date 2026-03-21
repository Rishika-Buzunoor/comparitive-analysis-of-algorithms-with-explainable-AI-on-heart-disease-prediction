from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_all_models(x_train, y_train):
    models={
        "Logistic":LogisticRegression(),
        "DecisionTree":DecisionTreeClassifier(),
        "RandomFirest":RandomForestClassifier(),
        "SVM":SVC(probability=True),
        "KNN":KNeighborsClassifier()
    }   
    for model in models.values():
        model.fit(x_train,y_train)
    return models