import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, \
    roc_auc_score, precision_score, matthews_corrcoef, cohen_kappa_score, average_precision_score
from imblearn.metrics import sensitivity_score, specificity_score
from sksurv.metrics import concordance_index_censored


def compute_cls_metrics(ground_truth, activations, avg='micro', demical_places=4):

    ground_truth = ground_truth.cpu().detach().numpy()
    activations = activations.cpu().detach().numpy()
    predictions = np.argmax(activations, -1)

    multi_class = 'ovr'
    ill_avg = avg
    # For binary classification
    if activations.shape[1] == 2:
        activations = activations[:, 1]
        multi_class = 'raise'
        # binary average is illegal for auc
        ill_avg = None
        avg = 'binary'

    mean_acc = accuracy_score(y_true=ground_truth, y_pred=predictions)
    f1 = f1_score(y_true=ground_truth, y_pred=predictions, average=avg)

    try:
        auc = roc_auc_score(y_true=ground_truth, y_score=activations, multi_class=multi_class, average=ill_avg)
    except ValueError as error:
        print('Error in computing AUC. Error msg:{}'.format(error))
        auc = 0
    try:
        ap = average_precision_score(y_true=ground_truth, y_score=activations, average=ill_avg)
    except Exception as error:
        print('Error in computing AP. Error msg:{}'.format(error))
        ap = 0
    bac = balanced_accuracy_score(y_true=ground_truth, y_pred=predictions)
    sens = sensitivity_score(y_true=ground_truth, y_pred=predictions, average=avg)
    spec = specificity_score(y_true=ground_truth, y_pred=predictions, average=avg)
    prec = precision_score(y_true=ground_truth, y_pred=predictions, average=avg)
    mcc = matthews_corrcoef(y_true=ground_truth, y_pred=predictions)
    kappa = cohen_kappa_score(y1=ground_truth, y2=predictions)

    metrics = {'Accuracy': mean_acc, 'F1': f1, 'AUC': auc, 'AP': ap, 'BAC': bac,
               'Sensitivity': sens, 'Specificity': spec, 'Precision': prec, 'MCC': mcc, 'Kappa': kappa}

    metrics = {k: round(v, demical_places) for k, v in metrics.items()}
    return metrics


def compute_surv_metrics(event_indicator, event_time, estimate, demical_places=4):
    event_indicator = event_indicator.cpu().detach().numpy()
    event_time = event_time.cpu().detach().numpy()
    estimate = estimate.cpu().detach().numpy()

    # event_indicator must be boolen
    event_indicator = event_indicator.astype(bool)

    cindex, *_ = concordance_index_censored(event_indicator, event_time, estimate, tied_tol=1e-08)
    metrics = {'C-index': cindex}

    metrics = {k: round(v, demical_places) for k, v in metrics.items()}
    return metrics
    