def coefficient_linear_kernel(database, weights):
    """Assuming a linear kernel, compute the coefficients of the data corresponding to the database.

    Args:
        database (_type_): _description_
        weights (_type_): _description_
    """

    # database is of shape [n_samples, n_features] too.
    # database.T.dot(weights) is [n_features, n_samples] @ [n_samples, 1] = [n_features, 1]
    coefs = database.T.dot(weights)
    # data.dot(coefs) \approx f(t)
    return coefs