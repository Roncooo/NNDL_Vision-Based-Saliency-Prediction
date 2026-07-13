"""
Mathematical implementations for saliency metrics and losses.

These functions are implemented natively in PyTorch (rather than using SciPy) 
so that operations can run efficiently on the GPU and their gradients can be 
tracked by autograd, allowing them to be used directly as loss functions.
"""

import torch
import torch.nn.functional as F

def kl(pred, gt):
    """Computes the Kullback-Leibler Divergence."""
    p = pred / (torch.sum(pred, dim=[-1, -2], keepdim=True) + 1e-7)
    q = gt / (torch.sum(gt, dim=[-1, -2], keepdim=True) + 1e-7)
    
    log_p = torch.log(p + 1e-7)
    
    # F.kl_div expects the input (prediction) to be log-probabilities
    # and the target (ground truth) to be probabilities.
    return F.kl_div(log_p, q, reduction='batchmean')

def pcc(pred, gt):
    """Computes the Pearson Correlation Coefficient."""
    pred_mean = pred - torch.mean(pred, dim=[1, 2, 3], keepdim=True)
    gt_mean = gt - torch.mean(gt, dim=[1, 2, 3], keepdim=True)
    
    cov = torch.sum(pred_mean * gt_mean, dim=[1, 2, 3])
    std_pred = torch.sqrt(torch.sum(pred_mean ** 2, dim=[1, 2, 3]) + 1e-7)
    std_gt = torch.sqrt(torch.sum(gt_mean ** 2, dim=[1, 2, 3]) + 1e-7)
    
    return cov / (std_pred * std_gt)

def jsd(pred, gt):
    """Computes the Jensen-Shannon Divergence."""
    p = pred / (torch.sum(pred, dim=[-1, -2], keepdim=True) + 1e-7)
    q = gt / (torch.sum(gt, dim=[-1, -2], keepdim=True) + 1e-7)
    
    m = 0.5 * (p + q)
    log_m = torch.log(m + 1e-7)
    
    kl_pm = F.kl_div(log_m, p, reduction='batchmean')
    kl_qm = F.kl_div(log_m, q, reduction='batchmean')
    
    return 0.5 * kl_pm + 0.5 * kl_qm

def nss(pred, gt):
    """Computes the Normalized Scanpath Saliency."""
    pred_mean = torch.mean(pred, dim=[1, 2, 3], keepdim=True)
    pred_std = torch.std(pred, dim=[1, 2, 3], keepdim=True)
    pred_norm = (pred - pred_mean) / (pred_std + 1e-7)
    
    gt_mask = (gt > 0).float()
    nss_val = torch.sum(pred_norm * gt_mask, dim=[1, 2, 3]) / (torch.sum(gt_mask, dim=[1, 2, 3]) + 1e-7)
    return nss_val

def auc_judd(pred, gt):
    """Computes AUC-Judd natively in PyTorch."""
    batch_size = pred.size(0)
    auc_vals = torch.zeros(batch_size, device=pred.device)
    for b in range(batch_size):
        p = pred[b].flatten()
        g = (gt[b].flatten() > 0).float()
        
        if torch.sum(g) == 0:
            auc_vals[b] = 0.5
            continue
            
        sorted_p, indices = torch.sort(p, descending=True)
        sorted_g = g[indices]
        
        tpr = torch.cumsum(sorted_g, dim=0) / (torch.sum(g) + 1e-7)
        fpr = torch.cumsum(1 - sorted_g, dim=0) / (torch.sum(1 - g) + 1e-7)
        
        tpr = torch.cat([torch.tensor([0.0], device=p.device), tpr])
        fpr = torch.cat([torch.tensor([0.0], device=p.device), fpr])
        
        auc = torch.sum((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1])) / 2.0
        auc_vals[b] = auc
        
    return auc_vals
