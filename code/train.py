# coding: utf-8

import matplotlib

from myCSP import CSP  # use my own CSP

# from mne.decoding import CSP
from mne.decoding import SPoC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from joblib import dump
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

matplotlib.use('TkAgg')


def pipeline_creation(X, y, transformer1, transformer2, transformer3):
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    lda = LDA(solver='svd')
    log_reg = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto')
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    scores1 = []
    scores2 = []
    scores3 = []

    pipeline1 = make_pipeline(transformer1, lda)
    pipeline2 = make_pipeline(transformer2, log_reg)
    pipeline3 = make_pipeline(transformer3, rfc)

    scores1 = cross_val_score(pipeline1, X, y, cv=cv, n_jobs=1)
    scores2 = cross_val_score(pipeline2, X, y, cv=cv, n_jobs=1)
    scores3 = cross_val_score(pipeline3, X, y, cv=cv, n_jobs=1)

    # print(f"LinearDiscriminantAnalysis : accuracy {scores1.mean().round(2)}, std: {scores1.std().round(2)}")
    # print(f"LogisticRegression         : accuracy {scores2.mean().round(2)}, std: {scores2.std().round(2)}")
    # print(f"RandomForestClassifier     : accuracy {scores3.mean().round(2)}, std: {scores3.std().round(2)}")

    return [('LDA ', pipeline1, scores1), ('LOGR', pipeline1, scores2), ('RFC ', pipeline3, scores3)]


def save_pipeline(pipe, epochs_data_train, labels, subjectID):
    pipe = pipe.fit(epochs_data_train, labels)
    fileName = f"models/model-subject-{subjectID}.joblib"
    dump(pipe, fileName)
    # print(f"-> model saved to {fileName}")
    return


def train_data(X, y, subjectID, transformer="CSP"):
    if transformer == "CSP":
        # using CSP transformers
        csp1 = CSP()
        csp2 = CSP()
        csp3 = CSP()
        return pipeline_creation(X, y, csp1, csp2, csp3)
    else:
        # using Spoc transformers
        Spoc1 = SPoC(n_components=15, reg='oas', log=True, rank='full')
        Spoc2 = SPoC(n_components=15, reg='oas', log=True, rank='full')
        Spoc3 = SPoC(n_components=15, reg='oas', log=True, rank='full')
        return pipeline_creation(X, y, Spoc1, Spoc2, Spoc3)
