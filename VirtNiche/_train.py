from typing import Dict, List, Union

import numpy as np
import torch
from scvi.train import TrainingPlan
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from ._constants import LOSS_KEYS
from ._module import VirtNicheModule
from ._utils import VirtNiche_metric


class VirtNicheTrainingPlan(TrainingPlan):
    """Training plan for the VirtNiche model."""

    def __init__(
        self,
        module: VirtNicheModule,
        n_steps_kl_warmup: Union[int, None] = None,
        n_epochs_kl_warmup: Union[int, None] = None,
        n_epochs_warmup: Union[int, None] = None,
        checkpoint_freq: int = 20,
        latent_lr=1e-4,
        latent_wd=1e-4,
        decoder_lr=1e-4,
        decoder_wd=1e-4,
        niche_lr: float = None,
        niche_wd: float = None,
        step_size_lr: int = 45,
        batch_size: int = 256,
        cosine_scheduler: bool = False,
        scheduler_max_epochs: int = 1000,
        scheduler_final_lr: float = 1e-5,
        attribute_nn_lr: Dict[str, float] = None,
        attribute_nn_wd: Dict[str, float] = None,
    ):
        super().__init__(
            module=module,
            lr=latent_lr,
            weight_decay=latent_wd,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=False,
            lr_factor=None,
            lr_patience=None,
            lr_threshold=None,
            lr_scheduler_metric=None,
            lr_min=None,
        )

        if isinstance(attribute_nn_lr, Dict):
            self.attribute_nn_lr = attribute_nn_lr
        elif attribute_nn_lr is None:
            self.attribute_nn_lr = {attribute_: self.latent_lr for attribute_ in self.module.ordered_networks}
        else:
            self.attribute_nn_lr = {attribute_: attribute_nn_lr for attribute_ in self.module.ordered_networks}

        if isinstance(attribute_nn_wd, Dict):
            self.attribute_nn_wd = attribute_nn_wd
        elif attribute_nn_wd is None:
            self.attribute_nn_wd = {attribute_: self.latent_wd for attribute_ in self.module.ordered_networks}
        else:
            self.attribute_nn_wd = {attribute_: attribute_nn_wd for attribute_ in self.module.ordered_networks}

        self.n_epochs_warmup = n_epochs_warmup if n_epochs_warmup is not None else 0

        self.decoder_wd = decoder_wd
        self.decoder_lr = decoder_lr

        self.latent_wd = latent_wd
        self.latent_lr = latent_lr
        
        self.niche_lr = self.latent_lr if niche_lr is None else niche_lr
        self.niche_wd = self.latent_wd if niche_wd is None else niche_wd

        self.checkpoint_freq = checkpoint_freq

        self.scheduler = CosineAnnealingLR if cosine_scheduler else StepLR
        self.scheduler_params = (
            {"T_max": scheduler_max_epochs, "eta_min": scheduler_final_lr}
            if cosine_scheduler
            else {"step_size": step_size_lr}
        )

        self.step_size_lr = step_size_lr
        self.batch_size = batch_size

        self.automatic_optimization = False
        self.iter_count = 0
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self._epoch_keys = []

        self.epoch_keys = [
            "generative_mean_accuracy",  # accuracy in prediction of mean gene exp.
            "generative_var_accuracy",  # accuracy in prediction of variance of gene exp.
            "VirtNiche_metric",  # combination of metrics above
            LOSS_KEYS.RECONSTRUCTION,
            LOSS_KEYS.UNKNOWN_ATTRIBUTE_PENALTY,
            "cell_r2_mean",
            "cell_pearson_mean",
            "cell_spearman_mean",
            "neg_total_loss",   
        ]

        self.epoch_history = {"mode": [], "epoch": []}
        for key in self.epoch_keys:
            self.epoch_history[key] = []

    def configure_optimizers(self):
        """Set up optimizers."""
        optimizers = []
        schedulers = []

        # latent codes
        latent_groups = [
                {
                    "params": list(filter(lambda p: p.requires_grad, self.module.latent_codes.parameters())),
                    "lr": self.latent_lr,
                    "weight_decay": self.latent_wd,
                }
            ]
        # --- add niche params if enabled ---
        if getattr(self.module, "use_niche_aggregation", False):
            niche_params = []
            if hasattr(self.module, "niche_W"):
                niche_params += list(self.module.niche_W.parameters())
            if hasattr(self.module, "niche_combine"):
                niche_params += list(self.module.niche_combine.parameters())
            if hasattr(self.module, "niche_a"):
                niche_params += [self.module.niche_a]

            niche_params = list(filter(lambda p: p.requires_grad, niche_params))
            if len(niche_params) > 0:
                latent_groups.append(
                    {
                        "params": niche_params,
                        "lr": self.niche_lr,
                        "weight_decay": self.niche_wd,
                    }
                )

        optimizers.append(torch.optim.Adam(latent_groups))
        
        
        
        
        
        
        # latent decoder
        optimizers.append(
            torch.optim.Adam(
                [
                    {
                        "params": list(
                            filter(
                                lambda p: p.requires_grad,
                                self.module.decoder.parameters(),
                            )
                        ),
                        "lr": self.decoder_lr,
                        "weight_decay": self.decoder_wd,
                    }
                ]
            )
        )
        # categorical classes
        optimizers.append(
            torch.optim.Adam(
                [
                    {
                        "params": list(
                            filter(
                                lambda p: p.requires_grad,
                                self.module.categorical_embeddings.parameters(),
                            )
                        ),
                        "lr": self.latent_lr,
                        "weight_decay": self.latent_wd,
                    }
                ]
            )
        )
        # ordered classes
        for attribute_, nn_ in self.module.ordered_networks.items():
            params_class = list(filter(lambda p: p.requires_grad, nn_.parameters()))
            optimizers.append(
                torch.optim.Adam(
                    [
                        {
                            "params": params_class,
                            "lr": self.attribute_nn_lr[attribute_],
                            "weight_decay": self.attribute_nn_wd[attribute_],
                            # betas=(0.5, 0.999),
                        }
                    ]
                )
            )



        

        
        if self.step_size_lr is not None:
            for optimizer in optimizers:
                schedulers.append(self.scheduler(optimizer, **self.scheduler_params))
            return optimizers, schedulers
        else:
            return optimizers
        



    @property
    def epoch_keys(self):
        """Epoch keys getter."""
        return self._epoch_keys

    @epoch_keys.setter
    def epoch_keys(self, epoch_keys: List):
        self._epoch_keys.extend(epoch_keys)

    def training_step(self, batch):
        """Training step."""
        optimizers = self.optimizers()
        # model update
        for optimizer in optimizers:
            optimizer.zero_grad()

        inf_outputs, gen_outputs, losses = self.module.forward(
            batch,
        )

        loss = (
            losses[LOSS_KEYS.RECONSTRUCTION]
            + self.module.unknown_attribute_penalty * losses[LOSS_KEYS.UNKNOWN_ATTRIBUTE_PENALTY]
        )

        self.manual_backward(loss)
        
        

 
        
        
        
        for optimizer in optimizers:
            optimizer.step()

        results = {
            LOSS_KEYS.RECONSTRUCTION: losses[LOSS_KEYS.RECONSTRUCTION].item(),
            LOSS_KEYS.UNKNOWN_ATTRIBUTE_PENALTY: losses[LOSS_KEYS.UNKNOWN_ATTRIBUTE_PENALTY].item(),
        }
        results["neg_total_loss"] = -loss.item()   
        self.iter_count += 1

        for key in self.epoch_keys:
            if key not in results:
                results.update({key: 0.0})

        self.training_step_outputs.append(results)
        return results

    def on_train_epoch_end(self):
        """Training epoch end."""
        outputs = self.training_step_outputs
        self.epoch_history["epoch"].append(self.current_epoch)
        self.epoch_history["mode"].append("train")

        for key in self.epoch_keys:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs]))
            self.log(key, self.epoch_history[key][-1], prog_bar=True)

        if self.current_epoch > 1 and self.current_epoch % self.step_size_lr == 0:
            schedulers = self.lr_schedulers()
            for scheduler in schedulers:
                scheduler.step()

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        inf_outputs, gen_outputs, losses = self(batch)

        r2_mean, r2_var = self.module.r2_metric(batch, gen_outputs)
        
    
        cell_metrics = self.module.cellwise_metrics(batch, gen_outputs)
        cell_r2_mean = float(np.nanmean(cell_metrics["r2"]))
        cell_pearson_mean = float(np.nanmean(cell_metrics["pearson"]))
        cell_spearman_mean = float(np.nanmean(cell_metrics["spearman"]))
        
        results = {}
        for key in losses:
            results.update({key: losses[key].item()})
        total_loss = (
            losses[LOSS_KEYS.RECONSTRUCTION]
            + self.module.unknown_attribute_penalty * losses[LOSS_KEYS.UNKNOWN_ATTRIBUTE_PENALTY]
        )
        results["neg_total_loss"] = -total_loss.item()

        results.update({"generative_mean_accuracy": r2_mean})
        results.update({"generative_var_accuracy": r2_var})
        results.update({"VirtNiche_metric": VirtNiche_metric(r2_mean, r2_var)})

        results.update(
            {
                "cell_r2_mean": cell_r2_mean,
                "cell_pearson_mean": cell_pearson_mean,
                "cell_spearman_mean": cell_spearman_mean,
            }
        )
        self.validation_step_outputs.append(results)
        return results

    def on_validation_epoch_end(self):
        """Validation step end."""
        outputs = self.validation_step_outputs
        self.epoch_history["epoch"].append(self.current_epoch)
        self.epoch_history["mode"].append("valid")
        for key in self.epoch_keys:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs]))
            self.log(f"val_{key}", self.epoch_history[key][-1], prog_bar=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """Test step end."""
        self.epoch_history["epoch"].append(self.current_epoch)
        self.epoch_history["mode"].append("test")
        for key in self.epoch_keys:
            self.epoch_history[key].append(np.mean([output[key] for output in outputs]))
            self.log(f"test_{key}", self.epoch_history[key][-1], prog_bar=True)


