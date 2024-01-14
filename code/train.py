# coding: utf-8

import matplotlib

from CSP import CSP  # use my own CSP

# from mne.decoding import CSP
from mne.decoding import SPoC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from joblib import dump
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

matplotlib.use('TkAgg')


def pipeline_creation(X, y, transformer1, transformer2=None, transformer3=None):
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    lda = LDA(solver='lsqr', shrinkage='auto')
    log_reg = LogisticRegression(penalty='l1', solver='liblinear', multi_class='auto')
    rfc = RandomForestClassifier(n_estimators=100, random_state=42)

    final_result = []

    pipeline1 = make_pipeline(transformer1, lda)
    scores1 = cross_val_score(pipeline1, X, y, cv=cv, n_jobs=1)
    final_result.append(('LDA ', pipeline1, scores1))
    if transformer2:
        pipeline2 = make_pipeline(transformer2, log_reg)
        scores2 = cross_val_score(pipeline2, X, y, cv=cv, n_jobs=1)
        final_result.append(('LOGR', pipeline2, scores2))
    if transformer3:
        pipeline3 = make_pipeline(transformer3, rfc)
        scores3 = cross_val_score(pipeline3, X, y, cv=cv, n_jobs=1)
        final_result.append(('RFC', pipeline3, scores3))

    # print(f"LinearDiscriminantAnalysis : accuracy {scores1.mean().round(2)}, std: {scores1.std().round(2)}")
    # print(f"LogisticRegression         : accuracy {scores2.mean().round(2)}, std: {scores2.std().round(2)}")
    # print(f"RandomForestClassifier     : accuracy {scores3.mean().round(2)}, std: {scores3.std().round(2)}")

    return final_result


def save_pipeline(pipe, epochs_data_train, labels, subjectID):
    pipe = pipe.fit(epochs_data_train, labels)
    fileName = f"models/model-subject-{subjectID}.joblib"
    dump(pipe, fileName)
    # print(f"-> model saved to {fileName}")
    return


def train_data(X, y, transformer="CSP", run_all_pipelines=False):
    if transformer == "CSP":
        from mne.decoding import CSP
        # using CSP transformers
        if run_all_pipelines:
            csp1 = CSP(n_components=10)
            csp2 = CSP(n_components=10)
            csp3 = CSP(n_components=10)
            return pipeline_creation(X, y, csp1, csp2, csp3)
        return pipeline_creation(X, y, CSP())
    elif transformer == "FAST_CSP":
        from CSP import CSP
        # using custom CSP transformers
        if run_all_pipelines:
            csp1 = CSP(n_components=10)
            csp2 = CSP(n_components=10)
            csp3 = CSP(n_components=10)
            return pipeline_creation(X, y, csp1, csp2, csp3)
        return pipeline_creation(X, y, CSP())

    elif transformer == "SPoC":
        # using Spoc transformers
        if run_all_pipelines:
            Spoc1 = SPoC(n_components=15, reg='oas', log=True, rank='full')
            Spoc2 = SPoC(n_components=15, reg='oas', log=True, rank='full')
            Spoc3 = SPoC(n_components=15, reg='oas', log=True, rank='full')
            return pipeline_creation(X, y, Spoc1, Spoc2, Spoc3)
        else:
            return pipeline_creation(X, y, SPoC(n_components=15, reg='oas', log=True, rank='full'))
    else:
        raise ValueError(f"Unknown transformer, please enter valid one.")