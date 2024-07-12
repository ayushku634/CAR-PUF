import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn.linear_model import LogisticRegression


# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y_train):
    ################################
    #  Non Editable Region Ending  #
    ################################

    # Use this method to train your model using training CRPs
    # X_train has 32 columns containing the challenge bits
    # y_train contains the responses

    feat = my_map(X_train)
    #model = LinearSVC(C=100, loss='squared_hinge',dual=False ,intercept_scaling=1, max_iter=10000)
    model = LogisticRegression(C=90, tol = 0.1)

    model.fit(feat, y_train) 

    # wTfeat + b
    w = model.coef_[0]
    b = model.intercept_

    # THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
    # If you do not wish to use a bias term, set it to 0
    return w, b

def my_map(X):
    # Transform input features
    transformed_X = 1 - 2 * X

    # Calculate cumulative product along the flipped matrix
    cumulative_product = np.flip(transformed_X, axis=1).cumprod(axis=1)

    # Get the dimension of a single observation
    n_dim = cumulative_product.shape[1]

    # Generate indices for selecting columns from the Khatri-Rao product
    upper_triangular_indices = np.triu_indices(n=n_dim, m=n_dim, k=1)
    col_indices = upper_triangular_indices[0] * n_dim + upper_triangular_indices[1]

    # Compute the Khatri-Rao product
    kr_product = khatri_rao(cumulative_product.T, cumulative_product.T).T

    # Select columns using the computed indices
    feature = kr_product[:, col_indices]

    # Append the original features x0, x1, ..., x31
    feature = np.hstack((feature, cumulative_product))

    return feature