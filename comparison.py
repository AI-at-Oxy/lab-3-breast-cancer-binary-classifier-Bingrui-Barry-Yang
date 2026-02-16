"""
Model Comparison with Random Forest Classifier

I compared my binary classifier with RandomForest model from sklearn. 
RandomForest is an ensemble method that combines many decision trees together.
Each tree makes a prediction and the model decides the final result by voting
to improve accuracy and reduce overfitting. 
I chose this model because it can capture nonlinear relationships in the data 
and learn complex patterns in the data. 

"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from binary_classification import load_data, train, predict, accuracy
import torch

if __name__ == "__main__":

    # Load data
    X_train, X_test, y_train, y_test, _ = load_data()

    # Load our from-scratch model
    checkpoint = torch.load('trained_model.pth')
    w = checkpoint['w']
    b = checkpoint['b']

    scratch_test_pred = predict(X_test, w, b)
    scratch_test_acc = accuracy(y_test, scratch_test_pred)

    # Train sklearn model
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train.numpy(), y_train.numpy())

    sklearn_test_pred = rf.predict(X_test.numpy())
    sklearn_test_acc = accuracy_score(
        y_test.numpy(), 
        sklearn_test_pred
    )

    # Print comparison
    print(f"From-scratch model test accuracy: {scratch_test_acc:.4f}")
    print(f"Random Forest test accuracy: {sklearn_test_acc:.4f}")

    if sklearn_test_acc > scratch_test_acc:
        print("Random Forest performed better.")
    else:
        print("From-scratch model performed better.")

"""
The from-scratch linear model performed slightly better than the Random Forest model. 
This suggests that the breast cancer dataset is mostly linearly separable, 
which means the two classes can be separated by a straight line or a plane in higher dimensions. 

"""