#  Copyright (c) Prior Labs GmbH 2025.
"""Example of using TabPFN for regression.

This example demonstrates how to use TabPFNRegressor on a regression task
using the diabetes dataset from scikit-learn.
"""

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)



# ---- debug start 
# bug description:
# Classifier was loading on gpu even by passing device="cpu" to load_fitted_tabpfn_model.
# it loads on cpu by passing arg device=torch.device("cpu")
# but Regressor seems to ignore the argument and attempts to load on cuda

from tabpfn.model.loading import save_fitted_tabpfn_model, load_fitted_tabpfn_model
import torch
from tabpfn import TabPFNClassifier
from torch import from_numpy


# Initialize a regressor
reg = TabPFNRegressor()
cls = TabPFNClassifier()

reg.fit(X_train, y_train)


X_train = torch.randn_like(from_numpy(X_train)).cpu()
y_train = torch.randint(low=0, high=2, size=list(y_train.shape)).cpu()

cls.fit(X_train, y_train)

save_fitted_tabpfn_model(estimator= reg, path="reg_model_path.tabpfn_fit")
save_fitted_tabpfn_model(estimator= cls, path="cls_model_path.tabpfn_fit")

for device in ["cpu", torch.device("cpu")]:
    loaded_reg = load_fitted_tabpfn_model(path="reg_model_path.tabpfn_fit", device=device)
    loaded_cls = load_fitted_tabpfn_model(path="cls_model_path.tabpfn_fit", device=device)

print("debug checkpoint")
# ----- debug end 

# Predict a point estimate (using the mean)
predictions = reg.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# Predict quantiles
quantiles = [0.25, 0.5, 0.75]
quantile_predictions = reg.predict(
    X_test,
    output_type="quantiles",
    quantiles=quantiles,
)
for q, q_pred in zip(quantiles, quantile_predictions):
    print(f"Quantile {q} MAE:", mean_absolute_error(y_test, q_pred))
# Predict with mode
mode_predictions = reg.predict(X_test, output_type="mode")
print("Mode MAE:", mean_absolute_error(y_test, mode_predictions))
