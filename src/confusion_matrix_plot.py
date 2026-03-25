from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

def plot_confusion_matrices(models,results,x_test,y_test):
    os.makedirs("outputs/graphs",exist_ok=True)

    for name,model in models.items():
        cm=confusion_matrix(y_test,results[name]["y_pred"])
        disp=ConfusionMatrixDisplay(cm)

        disp.plot()
        plt.title(f"{name} Confusion Matrix")
        plt.savefig(f"outputs/graphs/{name}_cm.png")
        plt.close()