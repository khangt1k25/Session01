import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def get_data(path):
    table = np.loadtxt(path)
    m, n = table.shape
    X = table[:, 1 : n - 1]
    y = table[:, n - 1]
    return X, y


def normalize_and_addbias(X):
    X_copy = np.array(X)
    m, n = X_copy.shape
    for column_id in range(n):
        max_values = np.max(X_copy[:, column_id])
        min_values = np.min(X_copy[:, column_id])
        X_copy[:, column_id] = (X_copy[:, column_id] - min_values) / (
            max_values - min_values
        )

    X_copy = np.column_stack((np.ones((m, 1)), X_copy))
    return np.array(X_copy)


class RIDGE_REGRESSION:
    def __init__(self):
        return

    def compute_RSS(self, y_new, y_predicted):
        error = 1.0 / 2 * np.sum((y_new - y_predicted) ** 2)
        return error

    def compute_ridge(self, y_new, y_predicted, w_learned):
        error = np.sum((y_new - y_predicted) ** 2) + np.sum(w_learned ** 2)
        return error / 2

    def fit(self, X_train, y_train, LAMBDA):
        iden = np.identity(X_train.shape[1])
        iden[0][0] = 0
        w = (
            np.linalg.pinv((X_train.transpose().dot(X_train) + LAMBDA * iden))
            .dot(X_train.transpose())
            .dot(y_train)
        )
        return w

    def predict(self, X, w):
        return np.array(X.dot(w))

    def fit_GD(
        self,
        X_train,
        y_train,
        LAMBDA,
        learning_rate=0.015,
        max_epoch=100,
        batch_size=15,
    ):
        iden = np.identity(X_train.shape[1])
        iden[0][0] = 0
        w = np.random.randn(X_train.shape[1])
        last_loss = 10e8
        for epoch in range(max_epoch):
            arr = np.array(range(X_train.shape[0]))
            np.random.shuffle(arr)
            total_minibatch = int(np.ceil(X_train.shape[0] / batch_size))
            X_train, y_train = X_train[arr], y_train[arr]

            for i in range(total_minibatch):
                index = i * batch_size
                X_train_sub = X_train[index : index + batch_size]
                y_train_sub = y_train[index : index + batch_size]

                grad = X_train_sub.transpose().dot(
                    X_train_sub.dot(w) - y_train_sub
                ) + iden.dot(w)

                w = w - learning_rate * grad

            new_loss = self.compute_ridge(self.predict(X_train, w), y_train, w)
            if np.abs(new_loss - last_loss) <= 1e-5:
                break
            else:
                last_loss = new_loss
        return w

    def get_best_LAMBDA(self, X_train, y_train):
        def cross_validation(number_folds, LAMBDA):

            row_ids = np.array(range(X_train.shape[0]))
            valid_ids = np.split(
                row_ids[: len(row_ids) - len(row_ids) % number_folds], number_folds
            )
            valid_ids[-1] = np.append(
                valid_ids[-1], row_ids[len(row_ids) - len(row_ids) % number_folds :]
            )
            train_ids = [
                [k for k in row_ids if k not in valid_ids[i]]
                for i in range(number_folds)
            ]
            aver_RSS = 0

            for i in range(number_folds):
                valid_part = {"X": X_train[valid_ids[i]], "y": y_train[valid_ids[i]]}
                train_part = {"X": X_train[train_ids[i]], "y": y_train[train_ids[i]]}
                w = self.fit(train_part["X"], train_part["y"], LAMBDA)
                y_predicted = self.predict(valid_part["X"], w)
                aver_RSS = self.compute_RSS(valid_part["y"], y_predicted)

            return aver_RSS / number_folds

        def range_scan(best_LAMBDA, minimum_RSS, LAMBDA_values):
            for current_LAMBDA in LAMBDA_values:
                aver_RSS = cross_validation(number_folds=5, LAMBDA=current_LAMBDA)
                if aver_RSS < minimum_RSS:
                    best_LAMBDA = current_LAMBDA
                    minimum_RSS = aver_RSS
            return best_LAMBDA, minimum_RSS

        best_LAMBDA, minimum_RSS = range_scan(
            best_LAMBDA=0, minimum_RSS=1000 ** 2, LAMBDA_values=range(50)
        )

        LAMBDA_values = [
            k * 1.0 / 1000
            for k in range(max(0, (best_LAMBDA - 1) * 100, (best_LAMBDA + 1) * 100, 1))
        ]
        best_LAMBDA, minimum_RSS = range_scan(
            best_LAMBDA=best_LAMBDA,
            minimum_RSS=minimum_RSS,
            LAMBDA_values=LAMBDA_values,
        )

        return best_LAMBDA


if __name__ == "__main__":

    X, y = get_data("data/death_rate_data.txt")
    X = normalize_and_addbias(X)
    X_train, y_train = X[:50], y[:50]
    X_test, y_test = X[50:], y[50:]

    model = RIDGE_REGRESSION()
    best_LAMBDA = model.get_best_LAMBDA(X_train, y_train)
    print("Best Lambda: ", best_LAMBDA)

    print("...........................")
    w_learned = model.fit(X_train, y_train, LAMBDA=best_LAMBDA)
    y_predicted = model.predict(X_test, w_learned)
    testRSS = mean_squared_error(y_test, y_predicted)
    print(
        " Train error fit: ",
        mean_squared_error(y_train, model.predict(X_train, w_learned)),
    )
    print(" Test error fit: ", testRSS)
    print(w_learned)

    print("...........................")
    w_learned1 = model.fit_GD(X_train, y_train, LAMBDA=best_LAMBDA)
    y_predicted1 = model.predict(X_test, w_learned1)
    testRSS1 = mean_squared_error(y_predicted1, y_test)
    print(
        " Train error fitGD: ",
        mean_squared_error(y_train, model.predict(X_train, w_learned1)),
    )
    print(" Test error fitGD: ", testRSS1)
    print(w_learned1)

    print("...........................")
    model_sklearn = Ridge(alpha=best_LAMBDA, solver="cholesky", fit_intercept=True)
    model_sklearn.fit(X_train, y_train)
    y_predicted2 = model_sklearn.predict(X_test)
    testRSS2 = mean_squared_error(y_test, y_predicted2)
    print(
        " Train error bysklearn: ",
        mean_squared_error(y_train, model_sklearn.predict(X_train)),
    )
    print(" Test error by sklearn: ", testRSS2)
    print(model_sklearn.intercept_, model_sklearn.coef_)
