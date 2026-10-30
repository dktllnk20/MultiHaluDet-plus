import time

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from scipy.special import logit
import xgboost as xgb
import lightgbm as lgb

# Classifiers that use one core per fit; their 6 fits run in separate processes.
SINGLE_CORE = {'svm', 'lr'}

def _log_odds(clf, X):
    """Meta-learner input: the SVC margin as it is, the logit of the probability otherwise."""
    if hasattr(clf, 'predict_proba'):
        return logit(np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6))
    return clf.decision_function(X)

def _fit_predict(clf, X_train, y_train, X_val):
    clf.fit(X_train, y_train)
    return clf, (None if X_val is None else _log_odds(clf, X_val))

class LogOddsStacker:
    def __init__(self, seed, names=None):
        self.seed = seed
        self.base_classifiers = [c for c in self._init_base_classifiers() if names is None or c[0] in names]
        self.meta_learner = LogisticRegression(penalty='l2', C=1.0, class_weight='balanced')
        
    def _init_base_classifiers(self):
        clfs = [
            ('rf', RandomForestClassifier(n_estimators=450, max_depth=17, class_weight='balanced', random_state=self.seed, n_jobs=-1)),
            # Histogram version of the former GradientBoostingClassifier: same trees, depth and rate, multi-core.
            ('gbc', HistGradientBoostingClassifier(max_iter=250, learning_rate=0.02, max_depth=6, max_leaf_nodes=None,
                                                   early_stopping=False, random_state=self.seed)),
            ('lr', LogisticRegression(C=0.25, class_weight='balanced', random_state=self.seed, max_iter=3000)),
            # No probability=True: it costs 5 extra fits, and the meta-learner can use the margin directly.
            ('svm', SVC(C=2.5, kernel='rbf', class_weight='balanced', cache_size=2500, random_state=self.seed)),
            ('xgb', xgb.XGBClassifier(n_estimators=400, learning_rate=0.008, random_state=self.seed, eval_metric='auc',
                                      tree_method='hist', device='cuda' if torch.cuda.is_available() else 'cpu')),
            ('lgb', lgb.LGBMClassifier(n_estimators=400, learning_rate=0.008, class_weight='balanced', random_state=self.seed, verbose=-1))
        ]
        return clfs

    def fit(self, X, y, groups=None):
        # The seed must differ from the one of the stage 2+3 folds. X holds out-of-fold features
        # from 5 separately trained networks whose feature spaces are unrelated. With the same
        # seed these folds are identical to those, so every held-out fold comes from a network
        # the classifier never saw: its out-of-fold scores are noise, the meta-learner weights
        # get a random sign, and the final AUC can come out inverted (0.02 instead of 0.98).
        # With groups (question ids) the two answers of a question always share a fold.
        if groups is None:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed + 1)
        else:
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=self.seed + 1)
        folds = list(skf.split(X, y, groups))
        oof_log_odds = np.zeros((X.shape[0], len(self.base_classifiers)))

        for i, (name, clf) in enumerate(self.base_classifiers):
            start = time.perf_counter()
            # 5 fold fits for the out-of-fold column, then one fit on all rows for prediction.
            jobs = [(X[train_idx], y[train_idx], X[val_idx]) for train_idx, val_idx in folds] + [(X, y, None)]
            n_jobs = len(jobs) if name in SINGLE_CORE else 1
            results = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(clone(clf), *job) for job in jobs)

            for (_, val_idx), (_, log_odds) in zip(folds, results[:-1]):
                oof_log_odds[val_idx, i] = log_odds
            self.base_classifiers[i] = (name, results[-1][0])
            print(f"[ensemble] {name}: {time.perf_counter() - start:.1f}s", flush=True)

        self.meta_learner.fit(oof_log_odds, y)
        return self

    def predict_proba(self, X):
        base_log_odds = np.column_stack([_log_odds(clf, X) for _, clf in self.base_classifiers])
        return self.meta_learner.predict_proba(base_log_odds)

LITE_STACK = ('lr', 'lgb')

def get_ensemble(seed, kind='full'):
    """kind='lite': only logistic regression and LightGBM, which is usually enough once the inputs
    are probe logits (a handful of strong, nearly linear features)."""
    return LogOddsStacker(seed, names=LITE_STACK if kind == 'lite' else None)

def calibrate_ensemble(ensemble, X_train, y_train, groups=None):
    ensemble.fit(X_train, y_train, groups)
    return ensemble, 1.0

def temperature_scale(probs, temperature):
    return probs
