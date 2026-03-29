def plot_metrics(results):
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    os.makedirs("outputs/graphs", exist_ok=True)

    models = list(results.keys())

    accuracy = [results[m]["accuracy"] for m in models]
    precision = [results[m]["precision"] for m in models]
    recall = [results[m]["recall"] for m in models]
    f1 = [results[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.2

    plt.figure(figsize=(12,6))

    bars1 = plt.bar(x - 1.5*width, accuracy, width, label="Accuracy")
    bars2 = plt.bar(x - 0.5*width, precision, width, label="Precision")
    bars3 = plt.bar(x + 0.5*width, recall, width, label="Recall")
    bars4 = plt.bar(x + 1.5*width, f1, width, label="F1 Score")

    plt.xticks(x, models)
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Model Comparison Metrics")

    plt.ylim(0.65, 0.85)  # 🔥 zoom for clarity
    plt.legend()
    plt.grid(axis='y')

    # ✅ Add values on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.2f}",
                ha='center',
                va='bottom',
                fontsize=8
            )

    plt.savefig("outputs/graphs/all_metrics.png")
    plt.close()