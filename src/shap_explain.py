import shap
import matplotlib.pyplot as plt
import os

def explain_shap(model,x_train,x_test,name):
    os.makedirs("outputs/explanations",exist_ok=True)

    if name in ["RandomForest","DecisionTree"]:
        explainer=shap.TreeExplainer(model)
    elif name == "Logistic":
        explainer=shap.LinearExplainer(model,x_train)
    else:
        explainer=shap.KernelExplainer(model.predict, x_train[:50])
    shap_values = explainer.shap_values(x_test[:10])

    shap.summary_plot(shap_values,x_test[:10],show =False)
    plt.savefig(f"outputs/explanations/{name}_shap.png")
    plt.close()