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

import os
from lime.lime_tabular import LimeTabularExplainer

def explain_lime(models, X_train, X_test):
    os.makedirs("outputs/explanations", exist_ok=True)

    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["No Disease", "Disease"],
        mode="classification"
    )

    i = 0  # same test instance for all models

    # Open one file for all models
    with open("outputs/explanations/lime_explanations.txt", "w") as f:

        for name, model in models.items():

            exp = explainer.explain_instance(
                X_test.iloc[i].values,
                model.predict_proba
            )

            lime_list = exp.as_list()
            sentence = lime_to_sentence(lime_list)

            #print(f"\n{name} Explanation:\n")
            #print(sentence)

            # Write into file
            f.write(f"Model: {name}\n")
            f.write("-" * 40 + "\n")
            f.write(sentence + "\n\n")