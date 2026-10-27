import torch
from torchmetrics import Metric
import numpy as np
from torchmetrics.utilities import dim_zero_cat
from scipy.stats import norm

class ExcessAtShortage(Metric):
    # Per torchmetrics docs: "setting this to False indicates that all batch states are independent" 
    full_state_update = False
    
    def __init__(self, thresholds, uncertainty_type, **kwargs):
        """
        Computes the amount of excess reserves allocated at a given shortage frequency threshold. This metric was inspired by the paper
        `How Can Probabilistic Solar Power Forecasts  Be Used to Lower Costs and Improve Reliability in Power Spot Markets? A Review and Application to Flexiramp Requirements`
        by Hobbs et.al.

        For a desired threshold `p` < 0.5, recalibrate the model's predicted ditribution such that q_p (respectively q_(1-p)) represents the `p'th` quantile.
        Then, obtain the amount of `up` excess (respectively `down`) through E_up = max(y_true-q_p, 0) (respectively E_down = max(q_(1-p)-y_true, 0)).
        Finally, obtain the mean normalized excess by E_up.mean() / (y_true-y_pred).mean()

        Args:
            -thresholds: list of floats. For every value in the list, use the indicated threshold to recalibrate the model.
            -uncertainty_type: string indicating the type of uncertainty. Can be one of: `gaussian` or `histogram`.
            -**kwargs: other arguments passed to the torchmetrics.Metric class.
        """
        super().__init__(**kwargs)
        
        self.thresholds = thresholds
        self.uncertainty_type = uncertainty_type

        valid_uncertainty = ["gaussian", "histogram"]
        assert self.uncertainty_type in valid_uncertainty, f"Uncertainty needs to be one of {valid_uncertainty}. Got {uncertainty_type}"

        # Define metric states
        self.add_state("y_true", default=[], dist_reduce_fx="cat")
        self.add_state("y_pred", default=[], dist_reduce_fx="cat")

        # Define state gaussian and histogram specific metric states 
        if self.uncertainty_type == "gaussian":
            self.add_state("y_var", default=[], dist_reduce_fx="cat")
        elif self.uncertainty_type == "histogram":
            self.add_state("bin_borders", default=[], dist_reduce_fx="cat")
            self.add_state("bin_cumulative_probs", default=[], dist_reduce_fx="cat")

    def update(self, trues, preds_point, preds_uncertainty=None):
        if preds_uncertainty is None:
            preds_uncertainty = preds_point
        self.y_true.append(trues["ghi"])
        self.y_pred.append(preds_point["ghi"])

        if self.uncertainty_type == "gaussian":
            self.y_var.append(preds_uncertainty["ghi_var"])
        elif self.uncertainty_type == "histogram":
            self.bin_borders.append(preds_uncertainty["bin_borders_ghi"])
            self.bin_cumulative_probs.append(preds_uncertainty["cumulative_probs"])
            
    def compute(self):
        # Standardize list states
        y_true = dim_zero_cat(self.y_true)
        y_pred = dim_zero_cat(self.y_pred)
        if self.uncertainty_type == "gaussian":
            y_var = dim_zero_cat(self.y_var)
        elif self.uncertainty_type == "histogram":
            bin_borders = dim_zero_cat(self.bin_borders)
            bin_cumulative_probs = dim_zero_cat(self.bin_cumulative_probs) 

        # Compute residual error across dataset
        errors = y_true - y_pred

        # When error is negative, the prediction is higher then the realized value, which means we need to increase generation elsewhere (upward ramp reserve) and vice-versa.
        up_samples = errors < 0
        down_samples = ~up_samples

        # Recalibrate the model's prediction using the desired shortage frequency thresholds
        if self.uncertainty_type == "gaussian":
            # Compute the cumulative prob at y_true and sort probs to obtain the desired threshold
            cdf_up = norm.cdf(y_true[up_samples], loc=y_pred[up_samples], scale=y_var[up_samples].sqrt())
            cdf_down = norm.cdf(y_true[down_samples], loc=y_pred[down_samples], scale=y_var[down_samples].sqrt())
            k_up, k_down = [int(np.floor(len(cdf_up) * th)) for th in self.thresholds], [int(np.ceil(len(cdf_down) * (1-th))) for th in self.thresholds]
            p_up, p_down = [np.sort(cdf_up)[k-1] for k in k_up], [np.sort(cdf_down)[k-1] for k in k_down]

            # Obtain the threshold quantiles using the calibrated probabilities
            quantiles_up = torch.tensor(norm.ppf(p_up, loc=y_pred[up_samples][:, None], scale=y_var[up_samples][:, None].sqrt()))
            quantiles_down = torch.tensor(norm.ppf(p_down, loc=y_pred[down_samples][:, None], scale=y_var[down_samples][:, None].sqrt()))
        elif self.uncertainty_type == "histogram":
            cum_probs_up, cum_probs_down = bin_cumulative_probs[up_samples], bin_cumulative_probs[down_samples]
            bin_borders_up, bin_borders_down = bin_borders[up_samples], bin_borders[down_samples]

             # Get y_true's corresponding bin index
            bin_idx_up, bin_idx_down = (bin_borders_up < y_true[up_samples].unsqueeze(1)).sum(dim=-1, keepdim=True), (bin_borders_down < y_true[down_samples].unsqueeze(1)).sum(dim=-1, keepdim=True)
            
            assert (bin_idx_up.min() > 0) and (bin_idx_down.min() > 0), "y_true is outside histogram bounds"

            # Assuming the mass of each bin is spread uniformly across it, we can obtain the cumulative probs by of y_true by piecewise linear interpolation
            x0_up, x1_up = bin_borders_up.gather(1, bin_idx_up-1).squeeze(1), bin_borders_up.gather(1, bin_idx_up).squeeze(1)
            y0_up, y1_up = cum_probs_up.gather(1, bin_idx_up-1).squeeze(1), cum_probs_up.gather(1, bin_idx_up).squeeze(1)
            cdf_up = y0_up + ((y_true[up_samples] - x0_up) / (x1_up - x0_up).clamp_min(1e-12))*(y1_up-y0_up)
            
            x0_down, x1_down = bin_borders_down.gather(1, bin_idx_down-1).squeeze(1), bin_borders_down.gather(1, bin_idx_down).squeeze(1)
            y0_down, y1_down = cum_probs_down.gather(1, bin_idx_down-1).squeeze(1), cum_probs_down.gather(1, bin_idx_down).squeeze(1)
            cdf_down = y0_down + ((y_true[down_samples] - x0_down) / (x1_down - x0_down).clamp_min(1e-12))*(y1_down-y0_down)

            # Obtain the calibrated p percentile
            k_up, k_down = [int(np.floor(len(cdf_up) * th)) for th in self.thresholds], [int(np.ceil(len(cdf_down) * (1-th))) for th in self.thresholds]    
            p_up, p_down = torch.tensor([np.sort(cdf_up)[k-1] for k in k_up]), torch.tensor([np.sort(cdf_down)[k-1] for k in k_down])
            
            # Perform piecewise linear interpolation once again to obtain the threshold quantiles
            bin_idx_up, bin_idx_down = (cum_probs_up[:, :, None] < p_up).sum(dim=1), (cum_probs_down[:, :, None] < p_down).sum(dim=1)
            assert (bin_idx_up.min() > 0) and (bin_idx_down.min() > 0), "Desired quantile is outside histogram bounds"
            
            x0_up, x1_up = cum_probs_up.gather(1, bin_idx_up-1), cum_probs_up.gather(1, bin_idx_up)
            y0_up, y1_up = bin_borders_up.gather(1, bin_idx_up-1), bin_borders_up.gather(1, bin_idx_up)
            quantiles_up = y0_up + ((p_up - x0_up) / (x1_up - x0_up).clamp_min(1e-12))*(y1_up-y0_up)
            
            x0_down, x1_down = cum_probs_down.gather(1, bin_idx_down-1), cum_probs_down.gather(1, bin_idx_down)
            y0_down, y1_down = bin_borders_down.gather(1, bin_idx_down-1), bin_borders_down.gather(1, bin_idx_down)
            quantiles_down = y0_down + ((p_down - x0_down) / (x1_down - x0_down).clamp_min(1e-12))*(y1_down-y0_down)

        # Quantiles are physically contrained to  >= 0
        quantiles_up, quantiles_down = quantiles_up.clamp_min(0), quantiles_down.clamp_min(0)

        # Compute the true and predicted required reserves
        true_req_up, true_req_down = errors[up_samples].abs()[:, None], errors[down_samples][:, None]
        pred_req_up, pred_req_down = y_pred[up_samples][:, None] - quantiles_up, quantiles_down - y_pred[down_samples][:, None]

        # Compute excess and shortage - shortage should be almost equal to desired threshold 
        S_up, S_down = torch.nn.functional.relu(true_req_up - pred_req_up), torch.nn.functional.relu(true_req_down - pred_req_down)
        E_up, E_down = torch.nn.functional.relu(pred_req_up - true_req_up), torch.nn.functional.relu(pred_req_down - true_req_down)

        # Compute shortage cost
        shortage_cost = (1 - torch.tensor(self.thresholds)) / torch.tensor(self.thresholds)
        return {
            "excess_up": E_up.mean(dim=0) / true_req_up.mean(),
            "excess_down": E_down.mean(dim=0) / true_req_down.mean(),
            "excess+shortage_up": E_up.mean(dim=0) + shortage_cost*S_up.mean(dim=0),
            "excess+shortage_down": E_down.mean(dim=0) + shortage_cost*S_down.mean(dim=0),
            "shortage_freq_up": (S_up > 0).float().mean(dim=0),
            "shortage_freq_down": (S_down > 0).float().mean(dim=0)
        }