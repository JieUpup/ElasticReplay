import numpy as np

def average_accuracy(acc_matrix, upto_task):
    vals = [acc_matrix[upto_task][j] for j in range(upto_task + 1)]
    return float(np.mean(vals))

def final_average_accuracy(acc_matrix):
    final_task = len(acc_matrix) - 1
    vals = [acc_matrix[final_task][j] for j in range(final_task + 1)]
    return float(np.mean(vals))

def forgetting(acc_matrix):
    K = len(acc_matrix)
    if K <= 1:
        return 0.0

    vals = []
    final_task = K - 1
    for j in range(final_task):
        best_prev = max(acc_matrix[i][j] for i in range(j, final_task))
        vals.append(best_prev - acc_matrix[final_task][j])

    return float(np.mean(vals)) if vals else 0.0
