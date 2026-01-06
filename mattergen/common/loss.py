# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F

from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.losses import SummedFieldLoss, denoising_score_matching
from mattergen.diffusion.model_target import ModelTarget
from mattergen.diffusion.training.field_loss import FieldLoss, d3pm_loss
from mattergen.diffusion.wrapped.wrapped_normal_loss import wrapped_normal_loss


class MaterialsLoss(SummedFieldLoss):
    def __init__(
        self,
        reduce: Literal["sum", "mean"] = "mean",
        d3pm_hybrid_lambda: float = 0.0,
        include_pos: bool = True,
        include_cell: bool = True,
        include_atomic_numbers: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        model_targets = {"pos": ModelTarget.score_times_std, "cell": ModelTarget.score_times_std}
        self.fields_to_score = []
        self.categorical_fields = []
        loss_fns: Dict[str, FieldLoss] = {}
        if include_pos:
            self.fields_to_score.append("pos")
            loss_fns["pos"] = partial(
                wrapped_normal_loss,
                reduce=reduce,
                model_target=model_targets["pos"],
            )
        if include_cell:
            self.fields_to_score.append("cell")
            loss_fns["cell"] = partial(
                denoising_score_matching,
                reduce=reduce,
                model_target=model_targets["cell"],
            )
        if include_atomic_numbers:
            model_targets["atomic_numbers"] = ModelTarget.logits
            self.fields_to_score.append("atomic_numbers")
            self.categorical_fields.append("atomic_numbers")
            loss_fns["atomic_numbers"] = partial(
                d3pm_loss,
                reduce=reduce,
                d3pm_hybrid_lambda=d3pm_hybrid_lambda,
            )
        self.reduce = reduce
        self.d3pm_hybrid_lambda = d3pm_hybrid_lambda
        super().__init__(
            loss_fns=loss_fns,
            weights=weights,
            model_targets=model_targets,
        )


class EnergyAwareMaterialsLoss(MaterialsLoss):
    """Extends MaterialsLoss with stability-aware energy prediction loss.

    This loss adds supervision on per-structure energy predictions using
    DFT-computed formation energies when available in the training data.
    The energy loss encourages the model to learn stability-aware representations.

    Args:
        energy_weight (float): Weight for energy loss term. Default: 0.1
        energy_target_key (str): Key for target energy in batch. Default: "formation_energy_per_atom"
        energy_pred_key (str): Key for predicted energy in model output. Default: "predicted_energy"
        **kwargs: Arguments passed to MaterialsLoss
    """

    def __init__(
        self,
        energy_weight: float = 0.1,
        energy_target_key: str = "formation_energy_per_atom",
        energy_pred_key: str = "predicted_energy",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.energy_weight = energy_weight
        self.energy_target_key = energy_target_key
        self.energy_pred_key = energy_pred_key

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption,
        batch: BatchedData,
        noisy_batch: BatchedData,
        score_model_output: BatchedData,
        t: torch.Tensor,
        node_is_unmasked: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total loss including denoising and optional energy supervision.

        Returns:
            Tuple of (loss, metrics_dict) where metrics_dict includes energy loss if computed.
        """
        # Compute base denoising losses
        base_loss, metrics_dict = super().__call__(
            multi_corruption=multi_corruption,
            batch=batch,
            noisy_batch=noisy_batch,
            score_model_output=score_model_output,
            t=t,
            node_is_unmasked=node_is_unmasked,
        )

        # Add energy supervision if available
        has_pred_energy = (
            hasattr(score_model_output, self.energy_pred_key)
            or self.energy_pred_key in score_model_output
        )
        has_target_energy = (
            hasattr(batch, self.energy_target_key)
            or self.energy_target_key in batch
        )

        if has_pred_energy and has_target_energy:
            pred_energy = score_model_output[self.energy_pred_key]
            target_energy = batch[self.energy_target_key]

            # MSE loss on energy
            energy_loss = F.mse_loss(pred_energy, target_energy)
            metrics_dict["energy"] = energy_loss.item()

            # Add weighted energy loss to total
            total_loss = base_loss + self.energy_weight * energy_loss
        else:
            total_loss = base_loss

        return total_loss, metrics_dict
