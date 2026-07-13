import torch
import torch.nn as nn
import torch.nn.functional as F
from functional import pcc, jsd, nss, auc_judd

METRIC_REGISTRY = {}

def register_metric(name):
    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn
    return decorator

# Mean Squared Error
@register_metric("MSE")
def build_mse_metric(config):
    def mse(pred, gt):
        return F.mse_loss(pred, gt).item()
    return mse

# Pearson Correlation Coefficient
@register_metric("PCC")
def build_pcc_metric(config):
    def pcc_metric(pred, gt):
        cc = pcc(pred, gt)
        return torch.mean(cc).item()
    return pcc_metric

# Jensen-Shannon Similarity
@register_metric("JSS")
def build_jss_metric(config):
    def jss_metric(pred, gt):
        return (1.0 - jsd(pred, gt)).item()
    return jss_metric

# Normalized Scanpath Saliency
@register_metric("NSS")
def build_nss_metric(config):
    def nss_metric(pred, gt):
        nss_val = nss(pred, gt)
        return torch.mean(nss_val).item()
    return nss_metric

# AUC-Judd
@register_metric("AUC_Judd")
def build_auc_judd_metric(config):
    def auc_judd_metric(pred, gt):
        auc_val = auc_judd(pred, gt)
        return torch.mean(auc_val).item()
    return auc_judd_metric

def build_metrics(config):
    """Factory function called by train.py"""
    metrics_dict = {}
    if not config:
        return metrics_dict
    for name in config:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Metric {name} not found in registry.")
        metrics_dict[name] = METRIC_REGISTRY[name](config.get(name, {}))
    return metrics_dict
