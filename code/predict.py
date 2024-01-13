from joblib import load
import numpy as np

def predict(X, y , subjectId):
    PREDICT_MODEL = f"models/model-subject-{subjectId}.joblib"
    try:
        clf = load(PREDICT_MODEL)
    except FileNotFoundError as e:
        raise Exception(f"File not found: {PREDICT_MODEL}")

    scores = []
    for n in range(X.shape[0]):
        pred = clf.predict(X[n:n + 1, :, :])
        # print(f"epoch_{n} =", "[{:<2}]".format(pred[0]), "truth =", "[{:<2}]".format(y[n:n + 1][0]))
        scores.append(1 - np.abs(pred[0] - y[n:n + 1][0]))
    # print("Mean acc= ", str(np.mean(scores).round(2)*100) + "%")
    return np.mean(scores).round(3)

