import matplotlib.pyplot as plt
import os

def plot_metrics(results):
    os.makedirs("outputs/graphs",exist_ok=True)
    models=list(results.keys())

    accuracy=[results[m]["accuracy"] for m in models]
    precision=[results[m]["precision"] for m in models]
    recall=[results[m]["recall"] for m in models]
    f1_score=[results[m]["f1_score"] for m in models]

    x=range(len(models))

    plt.figure()
    plt.plot(x, accuracy, marker='o', label="Accuracy")
    plt.plot(x, precision, marker='o', label="Precision")
    plt.plot(x, recall, marker='o', label="Recall")
    plt.plot(x, f1_score, marker='o', label="F1 Score")

    

    plt.savefig("outputs/graphs/metrics.png")
    plt.close()