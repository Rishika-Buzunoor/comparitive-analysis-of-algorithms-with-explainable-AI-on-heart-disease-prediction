import pandas as pd
import os

def show_predictions(models, X_test, y_test):
    os.makedirs("outputs", exist_ok=True)

    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Create DataFrame
        df = pd.DataFrame(X_test, columns=X_test.columns)
        df["Actual"] = y_test.values
        df["Predicted"] = y_pred

        print(f"\n{name} Predictions:\n")
        print(df.head(10))

        # Save to CSV
        df.to_csv(f"outputs/{name}_predictions.csv", index=False)