from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import os

def lime_to_sentence(lime_list):
    positive = []
    negative = []

    for condition, weight in lime_list:
        if weight > 0:
            positive.append(condition)
        else:
            negative.append(condition)

    sentence = ""

    if positive:
        sentence += "The model predicts heart disease mainly because "
        sentence += ", ".join(positive) + ". "

    if negative:
        sentence += "However, factors like "
        sentence += ", ".join(negative) + " reduce the risk."

    return sentence

def explain_lime(model, X_train, X_test, name):
    os.makedirs("outputs/explanations", exist_ok=True)

    
    explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["No Disease", "Disease"],
    mode="classification"
)
    
    i = 0  # you can change index

    # Step 3: Generate explanation
    exp = explainer.explain_instance(
        X_test.iloc[i].values,
        model.predict_proba
    )    
    lime_list = exp.as_list()
    sentence = lime_to_sentence(lime_list)

    print(sentence)

    with open(f"outputs/explanations/{name}_lime.txt", "w") as f:
        f.write(sentence)