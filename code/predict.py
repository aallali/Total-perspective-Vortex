from joblib import load
import numpy as np

def predict(X, y , subjectId, experiment_name, log=False):
    PREDICT_MODEL = f"models/model_subject_{subjectId}_{experiment_name}.joblib"
    try:
        clf = load(PREDICT_MODEL)
    except FileNotFoundError as e:
        raise Exception(f"File not found: {PREDICT_MODEL}")

    scores = []
    print("epoch_nb =  [prediction]    [truth]    equal?")
    print("---------------------------------------------")
    for n in range(X.shape[0]):
        pred = clf.predict(X[n:n + 1, :, :])[0]
        truth = y[n:n + 1][0]
        if log:
            print(f"epoch_{n:2} =      [{pred}]           [{truth}]      {'' if pred == truth else False}")
        scores.append(1 - np.abs(pred - y[n:n + 1][0]))
    # print("Mean acc= ", str(np.mean(scores).round(2)*100) + "%")
    return np.mean(scores).round(3)

