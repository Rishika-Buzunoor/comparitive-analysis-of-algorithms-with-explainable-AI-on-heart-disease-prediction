from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
def evaluate(models,x_test,y_test):
    results={}
    for name,model in models.items():
        y_pred=model.predict(x_test)
        results[name]={
            "accuracy":accuracy_score(y_test,y_pred),
            "precision":precision_score(y_test,y_pred),
            "recall":recall_score(y_test,y_pred),
            "f1":f1_score(y_test,y_pred),
            "y_pred":y_pred
        }
        print(f"\n{name} Results:")
        print(results[name])
    return results