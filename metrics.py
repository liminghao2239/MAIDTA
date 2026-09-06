import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr


def get_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def get_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def get_ci(y_true, y_pred):
    n = len(y_true)
    concordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
    return concordant / (n * (n - 1) / 2) if n > 1 else 0.5


def get_pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def get_spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


def get_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def get_rm2(y_true, y_pred):
    r2 = get_r2(y_true, y_pred)
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r2_0 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return r2 * (1 - np.sqrt(abs(r2 - r2_0)))