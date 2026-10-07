"""Open-set evaluation metrics (numpy/sklearn only).
Convention: higher score = more likely KNOWN. Known labels are 0..C-1; unknown = -1."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

_trapz = getattr(np, "trapezoid", None) or np.trapz


def threshold_at_tpr(val_scores, tpr=0.95):
    """tau such that a fraction `tpr` of KNOWN VALIDATION samples has score >= tau."""
    return float(np.quantile(val_scores, 1.0 - tpr))


def oscr(score_k, pred_k, label_k, score_u):
    """Area under the open-set classification rate curve (correct classification rate vs FPR)."""
    correct = pred_k == label_k
    s = np.concatenate([score_k, score_u])
    order = np.argsort(-s, kind="stable")
    is_u = np.r_[np.zeros(len(score_k)), np.ones(len(score_u))][order]
    corr = np.r_[correct, np.zeros(len(score_u), bool)][order]
    ccr = np.r_[0.0, np.cumsum(corr) / len(score_k)]
    fpr = np.r_[0.0, np.cumsum(is_u) / len(score_u)]
    return float(_trapz(ccr, fpr))


def evaluate(score_k, pred_k, label_k, score_u, pred_u, tau, n_classes):
    m = {}
    # closed-set (known test only, no rejection)
    m["closed_acc"] = float(accuracy_score(label_k, pred_k))
    m["closed_macro_f1"] = float(f1_score(label_k, pred_k, average="macro",
                                          labels=list(range(n_classes)), zero_division=0))
    # threshold-free open-set (unknown = positive class)
    y = np.r_[np.zeros(len(score_k)), np.ones(len(score_u))]
    s = -np.r_[score_k, score_u]
    m["auroc"] = float(roc_auc_score(y, s))
    m["aupr_unknown"] = float(average_precision_score(y, s))
    t95 = np.quantile(score_k, 0.05)
    m["fpr_at_95tpr"] = float(np.mean(score_u >= t95))
    m["oscr"] = oscr(score_k, pred_k, label_k, score_u)
    # threshold-based, tau fixed on known validation data
    y_true = np.r_[label_k, np.full(len(score_u), -1)]
    y_pred = np.r_[np.where(score_k >= tau, pred_k, -1), np.where(score_u >= tau, pred_u, -1)]
    labels = list(range(n_classes))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    m["open_known_macro_f1"] = float(np.mean(f))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[-1], zero_division=0)
    m["unknown_precision"], m["unknown_recall"], m["unknown_f1"] = float(p[0]), float(r[0]), float(f[0])
    m["tau"] = float(tau)
    m["known_accept_rate_test"] = float(np.mean(score_k >= tau))
    return m


def rejection_rate_by(score_u, tau, groups):
    """Unknown detection rate per group label (e.g. site_type, unknown_group, model_code)."""
    groups = np.asarray(groups)
    return {str(g): {"n": int((groups == g).sum()),
                "detection_rate": float(np.mean(score_u[groups == g] < tau))}
            for g in sorted(set(groups))}
