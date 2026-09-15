from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from scvi import REGISTRY_KEYS, settings
from scvi.distributions import NegativeBinomial, Poisson
from scvi.module import Classifier
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import Decoder, DecoderSCVI, FCLayers
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
import torch.nn.functional as F   
from torch.distributions import Categorical, Normal

from ._constants import LOSS_KEYS

__all__ = ["RegularizedEmbedding", "VirtNicheModule"]


class RegularizedEmbedding(nn.Module):
    """Regularized embedding module."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        sigma: float,
        embed: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=n_input,
            embedding_dim=n_output,
        )
        self.sigma = sigma if embed else 0
        self.embed = embed

    def forward(self, x):
        """Forward pass."""
        x_ = self.embedding(x)
        if self.training and self.sigma != 0:
            noise = torch.zeros_like(x_)
            noise.normal_(mean=0, std=self.sigma)

            x_ = x_ + noise
        x_ = x_ * self.embed
        return x_


class VirtNicheModule(BaseModuleClass):
    """The :mod:`VirtNiche` module.

    Parameters
    ----------
    n_genes
        Number of input genes.
    n_samples
        Number of layers.
    x_loc
        The expression data location.
    ordered_attributes_map
        Dictionary of ordered classes and their dimensions.
    categorical_attributes_map
        Dictionary for categorical classes, containing categorical values with keys as each category name and values
        as the categorical integer assignment.
    n_latent
        Latent dimension.
    n_latent_attribute_ordered
        Latent dimension of ordered attributes.
    n_latent_attribute_categorical
        Latent dimension of categorical attributes.
    gene_likelihood
        The gene_likelihood model.
    reconstruction_penalty
        MSE error to reconstruction loss.
    use_batch_norm
        Use batch norm in layers.
    use_layer_norm
        Use layer norm in layers.
    unknown_attribute_noise_param
        Noise strength added to encoding of unknown attributes.
    unknown_attributes
        Whether to include learning for unknown attributes
    attribute_dropout_rate
        Dropout rate.
    attribute_nn_width
        Ordered attributes autoencoder layers' width.
    attribute_nn_depth
        Ordered attributes autoencoder number of layers.
    attribute_nn_activation
        Use activation in ordered attributes.
    decoder_width
        Decoder layers' width.
    decoder_depth
        Decoder number of layers.
    decoder_activation
        Use activation in decoder.
    eval_r2_ordered
        Evaluate the R2 w.r.t. the ordered attribute. Set to `True` only if ordered attributes are binned.
    decoder_dropout_rate
        Decoder dropout rate.
    seed
        Random seed.
    """

    def __init__(
        self,
        n_genes: int,
        n_samples: int,
        x_loc: str,
        ordered_attributes_map: Optional[Dict[str, int]] = None,
        categorical_attributes_map: Optional[Dict[str, Dict]] = None,
        n_latent: int = 32,
        n_latent_attribute_categorical: int = 4,
        n_latent_attribute_ordered: int = 16,
        gene_likelihood: Literal["normal", "nb", "poisson"] = "normal",
        reconstruction_penalty: float = 1e2,
        unknown_attribute_penalty: float = 1e1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        unknown_attribute_noise_param: float = 1e-1,
        unknown_attributes: bool = True,
        attribute_dropout_rate: Dict[str, float] = None,
        decoder_width: int = 512,
        decoder_depth: int = 4,
        decoder_activation: bool = True,
        attribute_nn_width: Dict[str, int] = None,
        attribute_nn_depth: Dict[str, int] = None,
        attribute_nn_activation: bool = True,
        eval_r2_ordered: bool = False,
        decoder_dropout_rate: float = 0.1,
        seed: int = 0,
        sample: bool = False,
        neighbors_index: Optional[np.ndarray] = None,
        full_categorical_attributes: Optional[Dict[str, np.ndarray]] = None,
        full_ordered_attributes: Optional[Dict[str, np.ndarray]] = None,
        agg_mode: str = "none",  # "none","mean","max","gcn","attn","gat"
    ):
        super().__init__()
        gene_likelihood = gene_likelihood.lower()
        assert gene_likelihood in ["normal", "nb", "poisson"], gene_likelihood

        default_width = 256
        default_depth = 2
        torch.manual_seed(seed)
        np.random.seed(seed)
        settings.seed = seed

        self.ae_loss_fn = nn.GaussianNLLLoss()
        self.ae_loss_mse_fn = nn.MSELoss()
        self.reconstruction_penalty = reconstruction_penalty
        self.unknown_attribute_penalty = unknown_attribute_penalty
        self.mm_regression_loss_fn = nn.BCEWithLogitsLoss()

        self.n_genes = n_genes
        self.n_latent = n_latent
        self.x_loc = x_loc
        self.n_latent_attribute_categorical = n_latent_attribute_categorical
        self.n_latent_attribute_ordered = n_latent_attribute_ordered
        self.gene_likelihood = gene_likelihood
        self.sample = sample
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.eval_r2_ordered = eval_r2_ordered

        self.n_decoder_input = n_latent + (
            n_latent_attribute_categorical * len(categorical_attributes_map)
            + n_latent_attribute_ordered * len(ordered_attributes_map)
        )
        self.categorical_attributes_map = (
            categorical_attributes_map if isinstance(categorical_attributes_map, Dict) else {}
        )
        self.ordered_attributes_map = ordered_attributes_map if isinstance(ordered_attributes_map, Dict) else {}

        if isinstance(attribute_nn_width, Dict):
            self.attribute_nn_width = attribute_nn_width
        elif attribute_nn_width is None:
            self.attribute_nn_width = {attribute_: default_width for attribute_ in self.ordered_attributes_map}
        else:
            self.attribute_nn_width = {attribute_: attribute_nn_width for attribute_ in self.ordered_attributes_map}

        if isinstance(attribute_nn_depth, Dict):
            self.attribute_nn_depth = attribute_nn_depth
        elif attribute_nn_depth is None:
            self.attribute_nn_depth = {attribute_: default_depth for attribute_ in self.ordered_attributes_map}
        else:
            self.attribute_nn_depth = {attribute_: attribute_nn_depth for attribute_ in self.ordered_attributes_map}

        if isinstance(attribute_dropout_rate, Dict):
            self.attribute_dropout_rate = attribute_dropout_rate
        elif attribute_dropout_rate is None:
            self.attribute_dropout_rate = {
                attribute_: decoder_dropout_rate for attribute_ in self.ordered_attributes_map
            }
        else:
            self.attribute_dropout_rate = {
                attribute_: attribute_dropout_rate for attribute_ in self.ordered_attributes_map
            }

        self.latent_codes = RegularizedEmbedding(
            n_input=n_samples, n_output=n_latent, sigma=unknown_attribute_noise_param, embed=unknown_attributes
        )

        # Create Embeddings
        # 1. ordered classes
        reps_ordered = []
        self.ordered_networks = nn.ModuleDict()
        for attribute_, len_ in self.ordered_attributes_map.items():
            if "_rep" in attribute_:
                reps_ordered.append(attribute_)
            else:
                self.ordered_networks[attribute_] = FCLayers(
                    n_in=len_,
                    n_out=self.n_latent_attribute_ordered,
                    n_layers=self.attribute_nn_depth[attribute_],
                    n_hidden=self.attribute_nn_width[attribute_],
                    dropout_rate=self.attribute_dropout_rate[attribute_],
                    bias=False,
                    use_activation=attribute_nn_activation,
                )
        for attribute_ in reps_ordered:
            self.ordered_networks[attribute_] = self.ordered_networks[attribute_.split("_rep")[0]]

        # 2. categorical classes
        self.categorical_embeddings = nn.ModuleDict()
        reps_categorical = []
        for attribute_, unique_categories in self.categorical_attributes_map.items():
            if "_rep" in attribute_:
                reps_categorical.append(attribute_)
            else:
                self.categorical_embeddings[attribute_] = torch.nn.Embedding(
                    len(unique_categories),
                    n_latent_attribute_categorical,
                )
        for attribute_ in reps_categorical:
            self.categorical_embeddings[attribute_] = self.categorical_embeddings[attribute_.split("_rep")[0]]

        # Decoder components
        if self.gene_likelihood in ["nb", "poisson"]:
            self.decoder = DecoderSCVI(
                n_input=self.n_decoder_input,
                n_output=n_genes,
                n_hidden=decoder_width,
                n_layers=decoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                scale_activation="softmax",
            )
            self.px_r = torch.nn.Parameter(torch.randn(n_genes))
        else:
            self.decoder = Decoder(
                n_input=self.n_decoder_input,
                n_output=n_genes,
                n_hidden=decoder_width,
                n_layers=decoder_depth,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_activation=decoder_activation,
            )
        # --- Niche aggregation related ---
        self.agg_mode = agg_mode.lower() if isinstance(agg_mode, str) else "none"
        self.use_niche_aggregation = neighbors_index is not None and self.agg_mode != "none"

        if self.use_niche_aggregation:
            # 1.  (n_samples, K)
            if isinstance(neighbors_index, np.ndarray):
                neighbors_index = torch.from_numpy(neighbors_index.astype("int64"))
            elif isinstance(neighbors_index, torch.Tensor):
                neighbors_index = neighbors_index.long()
            else:
                raise TypeError("neighbors_index must be np.ndarray or torch.Tensor")

            self.register_buffer("neighbors_index", neighbors_index)  # [n_cells, K]

            # 2.  categorical attributes
            self.full_categorical_attr_names = {}
            if full_categorical_attributes is not None:
                for attribute_, arr in full_categorical_attributes.items():
                    tensor = torch.as_tensor(arr, dtype=torch.long)
                    name = f"{attribute_}_categorical_all"
                    self.register_buffer(name, tensor)
                    self.full_categorical_attr_names[attribute_] = name

            # 3. continues attributes
            self.full_ordered_attr_names = {}
            if full_ordered_attributes is not None:
                for attribute_, arr in full_ordered_attributes.items():
                    tensor = torch.as_tensor(arr, dtype=torch.float32)
                    if tensor.ndim == 1:
                        tensor = tensor.view(-1, 1)
                    name = f"{attribute_}_ordered_all"
                    self.register_buffer(name, tensor)
                    self.full_ordered_attr_names[attribute_] = name

            if self.agg_mode in ["attn", "gat", "gcn"]:
                self.niche_W = nn.Linear(self.n_decoder_input, self.n_decoder_input, bias=False)

            if self.agg_mode in ["attn", "gat"]:
                self.niche_a = nn.Parameter(torch.empty(2 * self.n_decoder_input, 1))
                nn.init.xavier_uniform_(self.niche_a.data)

            if self.agg_mode in ["mean", "max", "gcn", "attn", "gat"]:
                self.niche_combine = nn.Linear(2 * self.n_decoder_input, self.n_decoder_input)

        
        
    def _get_inference_input(self, tensors: Dict[Any, Any], **kwargs):
        x = tensors[self.x_loc]  # batch_size, n_genes
        sample_indices = tensors[REGISTRY_KEYS.INDICES_KEY].long().ravel()

        categorical_attribute_dict = {}
        for attribute_ in self.categorical_attributes_map:
            categorical_attribute_dict[attribute_] = tensors[attribute_].view(
                -1,
            )

        ordered_attribute_dict = {}
        for attribute_ in self.ordered_attributes_map:
            ordered_attribute_dict[attribute_] = tensors[attribute_]

        input_dict = {
            "genes": x,
            "sample_indices": sample_indices,
            "categorical_attribute_dict": categorical_attribute_dict,
            "ordered_attribute_dict": ordered_attribute_dict,
        }
        return input_dict

    def get_inference_input(self, tensors: Dict[Any, Any], **kwargs) -> Dict[str, Any]:
        """Convert tensors to valid inference input.

        Parameters
        ----------
        tensors
            Considered inputs.
        kwargs
            Additional arguments

        Returns
        -------
        Dictionary with the module's expected input tensors (`genes`, `sample_indices`, `categorical_attribute_dict`, and `ordered_attribute_dict`).
        """
        return self._get_inference_input(tensors, **kwargs)

    @auto_move_data
    def _inference_attribute_embeddings(
        self,
        genes,
        categorical_attribute_dict,
        ordered_attribute_dict,
        nullify_attribute=None,
    ):
        """Inference over attribute embeddings."""
        nullify_attribute = [] if nullify_attribute is None else nullify_attribute
        inference_output = {}
        batch_size = genes.shape[0]
        for attribute_, embedding_ in self.categorical_embeddings.items():
            latent_i = embedding_(categorical_attribute_dict[attribute_].long())
            latent_i = latent_i.view(batch_size, self.n_latent_attribute_categorical).unsqueeze(
                0
            )  # 1, batch_size, n_latent_attribute_categorical
            if attribute_ in nullify_attribute:
                latent_i = torch.zeros_like(latent_i)
            inference_output[attribute_] = latent_i

        for attribute_, network_ in self.ordered_networks.items():
            latent_i = network_(ordered_attribute_dict[attribute_])
            latent_i = latent_i.view(batch_size, self.n_latent_attribute_ordered).unsqueeze(0)
            if attribute_ in nullify_attribute:
                latent_i = torch.zeros_like(latent_i)
            inference_output[attribute_] = latent_i

        return inference_output

    def _get_latent_unknown_attributes(
        self,
        sample_indices,
    ):
        """Get the module's latent unknown attributes representation."""
        latent_unknown_attributes = self.latent_codes(sample_indices)

        return latent_unknown_attributes

    def _get_full_latent_for_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Compute the complete embedding for cells identified by global row indices.
        Shape (N, n_decoder_input)
        """
        indices = indices.long()
        # 1. unknown attributes
        latent_unknown = self._get_latent_unknown_attributes(sample_indices=indices)  # (N, n_latent)
        latent_vecs = [latent_unknown]

        # 2. categorical attributes
        for attribute_ in self.categorical_attributes_map:
            name = self.full_categorical_attr_names.get(attribute_, None)
            if name is None:
                continue
            labels_all = getattr(self, name)         # (n_samples,)
            labels = labels_all[indices]             # (N,)
            emb = self.categorical_embeddings[attribute_](labels)  # (N, n_latent_attribute_categorical)
            latent_vecs.append(emb)

        # 3. ordered attributes
        for attribute_ in self.ordered_attributes_map:
            name = self.full_ordered_attr_names.get(attribute_, None)
            if name is None:
                continue
            vals_all = getattr(self, name)           # (n_samples, d_attr)
            vals = vals_all[indices]                 # (N, d_attr)
            emb = self.ordered_networks[attribute_](vals)  # (N, n_latent_attribute_ordered)
            latent_vecs.append(emb)

        return torch.cat(latent_vecs, dim=-1)
    def _aggregate_niche(self, latent_center: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
        """
        Aggregate the center-cell representation with the representations of its spatial neighbors.
        """
        if not self.use_niche_aggregation:
            return latent_center

        B, D = latent_center.shape
        # neighbors_index: (n_samples, K)
        neighbor_idx = self.neighbors_index[sample_indices.long()]   # (B, K)
        K = neighbor_idx.shape[1]

        # Flatten neighbor indices, compute neighbor embeddings, and restore the batch shape.
        neighbor_idx_flat = neighbor_idx.reshape(-1)                 # (B*K,)
        neighbors_latent_flat = self._get_full_latent_for_indices(neighbor_idx_flat)  # (B*K, D)
        neighbors_latent = neighbors_latent_flat.view(B, K, D)       # (B, K, D)
        return self.aggregate_from_latents(latent_center, neighbors_latent)
        
    def aggregate_from_latents(
        self,
        latent_center: torch.Tensor,
        neighbors_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate center and neighbor latent representations.

        This method performs neighborhood aggregation directly from the provided
        latent representations and does not require ``neighbors_index``.

        Parameters
        ----------
        latent_center
            Latent representation of the center cells with shape ``(B, D)``, where
            ``B`` is the number of center cells and ``D`` is the latent dimension.

        neighbors_latent
            Latent representations of the neighbor cells with shape ``(B, K, D)``,
            where ``K`` is the number of neighbors per center cell.

        Returns
        -------
        torch.Tensor
            Aggregated latent representation of the center cells with shape
            ``(B, D)``.
        """
        if not self.use_niche_aggregation:
            return latent_center

        if neighbors_latent.ndim != 3:
            raise ValueError(f"neighbors_latent must be 3D (B, K, D), got shape {neighbors_latent.shape}.")

        B, D = latent_center.shape
        B2, K, D2 = neighbors_latent.shape
        if B2 != B or D2 != D:
            raise ValueError(
                f"latent_center shape {latent_center.shape} and neighbors_latent shape {neighbors_latent.shape} "
                "are incompatible."
            )

        mode = self.agg_mode

        if mode == "mean":
            neigh_agg = neighbors_latent.mean(dim=1)                    # (B, D)
            out = 0.5 * (latent_center + neigh_agg)

        elif mode == "max":
            neigh_agg, _ = neighbors_latent.max(dim=1)                  # (B, D)
            out = 0.5 * (latent_center + neigh_agg)

        elif mode == "gcn":
            Wh_c = self.niche_W(latent_center)                          # (B, D)
            Wh_n = self.niche_W(neighbors_latent)                       # (B, K, D)
            neigh_agg = Wh_n.mean(dim=1)                                # (B, D)
            out = F.relu(self.niche_combine(torch.cat([Wh_c, neigh_agg], dim=-1)))  # (B, D)

        elif mode in ["attn", "gat"]:
            Wh_c = self.niche_W(latent_center)                          # (B, D)
            Wh_n = self.niche_W(neighbors_latent)                       # (B, K, D)
            B_, K_, D_ = Wh_n.shape

            Wh_c_exp = Wh_c.unsqueeze(1).expand(-1, K_, -1)             # (B, K, D)
            cat = torch.cat([Wh_c_exp, Wh_n], dim=-1)                   # (B, K, 2D)
            e = F.leaky_relu(torch.matmul(cat, self.niche_a).squeeze(-1))  # (B, K)
            alpha = F.softmax(e, dim=1).unsqueeze(-1)                   # (B, K, 1)
            neigh_agg = (alpha * Wh_n).sum(dim=1)                       # (B, D)

            out = F.elu(self.niche_combine(torch.cat([Wh_c, neigh_agg], dim=-1)))  # (B, D)

        else:
            out = latent_center

        return out



    @auto_move_data
    def inference(
        self,
        genes: torch.Tensor,
        sample_indices: torch.Tensor,
        categorical_attribute_dict: Dict[Any, Any],
        ordered_attribute_dict: Dict[Any, Any],
        nullify_attribute: Optional[List] = None,
    ) -> Dict[str, Any]:
        """Apply module inference.

        Parameters
        ----------
        genes
            Input expression.
        sample_indices
            Indices in the :class:`~anndata.AnnData` object of the input samples.
        categorical_attribute_dict
            Dictionary with categorical attributes as keys and the attribute sample labels as values.
        ordered_attribute_dict
            Dictionary with ordered attributes as keys and the attribute sample values as values.
        nullify_attribute
            Attributes to exclude from inferred latent space.

        Returns
        -------
        Dictionary with the module's expected input tensors (`genes`, `sample_indices`, `categorical_attribute_dict`, and `ordered_attribute_dict`).
        """
        nullify_attribute = [] if nullify_attribute is None else nullify_attribute
        inference_output = {}
        x_ = genes
        library = torch.log(genes.sum(1)).unsqueeze(1)

        latent_unknown_attributes = self._get_latent_unknown_attributes(sample_indices=sample_indices)

        latent_classes = self._inference_attribute_embeddings(
            genes=x_,
            categorical_attribute_dict=categorical_attribute_dict,
            ordered_attribute_dict=ordered_attribute_dict,
            nullify_attribute=nullify_attribute,
        )

        latent_vecs = [latent_unknown_attributes.squeeze()]
        for key_, latent_ in latent_classes.items():
            latent_vecs.append(latent_.squeeze())  
            inference_output[key_] = latent_.squeeze()


        latent_raw = torch.cat(latent_vecs, dim=-1)   # (B, D)

        if self.use_niche_aggregation:
            latent = self._aggregate_niche(latent_raw, sample_indices)  # (B, D)
        else:
            latent = latent_raw

        inference_output["latent_raw"] = latent_raw                     
        inference_output["latent"] = latent                              
        inference_output["latent_unknown_attributes"] = latent_unknown_attributes
        inference_output["library"] = library

        return inference_output

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        input_dict = {
            "latent": inference_outputs["latent"],
            "library": inference_outputs["library"],
        }
        return input_dict

    @auto_move_data
    def generative(
        self,
        latent: torch.Tensor,
        library: torch.Tensor = None,
    ) -> Dict[str, Any]:
        """Runs the generative step.

        Parameters
        ----------
        latent
            The concatenated decomposed latent space.  
        library
            Library sizes for each cell.

        Returns
        -------
        Dictionary with the generative predictions of the expression distribution.
        """
        if self.gene_likelihood in ["nb", "poisson"]:
            px_scale, _, px_rate, _ = self.decoder(
                dispersion="gene",
                z=latent,
                library=library,
            )
            px_r = torch.exp(self.px_r)
            px = (
                NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
                if self.gene_likelihood == "nb"
                else Poisson(px_rate)
            )  # , scale=px_scale)

            if self.sample:
                total = 0
                acc = 0.0
                for _ in range(40):
                    s = px.sample((5,))
                    if s.dim() == 4:
                        s = s.squeeze(1)
                    acc = acc + s.sum(0)
                    total += 5
                samples = acc / total
            else:
                samples = px.sample().squeeze(0)

            return {
                "means": px.mean,
                "variances": px.variance,
                "distribution": px,
                "samples": samples,
            }

        else:
            p_m, p_v = self.decoder(x=latent)
            px = Normal(loc=p_m, scale=p_v.sqrt())
            if self.sample:
                total = 0
                acc = 0.0
                for _ in range(40):
                    s = px.sample((5,))
                    if s.dim() == 4:
                        s = s.squeeze(1)
                    acc = acc + s.sum(0)
                    total += 5
                samples = acc / total
            else:
                samples = px.sample().squeeze(0)
            return {
                "means": px.loc,
                "variances": px.variance,
                "distribution": px,
                "samples": samples,
            }

    @auto_move_data
    def loss(
        self,
        tensors: Dict[str, torch.Tensor],
        inference_outputs: Dict[Literal["latent_unknown_attributes"], torch.Tensor],
        generative_outputs: Dict[Literal["distribution", "means", "variances"], torch.Tensor],
    ) -> Dict[str, float]:
        """Computes the module's loss.

        Parameters
        ----------
        tensors
            Considered model inputs.
        inference_outputs
            Inference step outputs.
        generative_outputs
            Generative step outputs.

        Returns
        -------
        The loss elements.
        """
        x_ = tensors[self.x_loc]
        means = generative_outputs["means"]
        variances = generative_outputs["variances"]

        if self.gene_likelihood in ["nb", "poisson"]:
            reconstruction_loss = -generative_outputs["distribution"].log_prob(x_).sum(-1)
            reconstruction_loss = reconstruction_loss.mean()
        else:
            reconstruction_loss = self.ae_loss_fn(input=means, target=x_, var=variances)

        reconstruction_loss += self.reconstruction_penalty * self.ae_loss_mse_fn(input=means, target=x_)

        unknown_attribute_penalty_loss_val = self.unknown_attribute_penalty_loss(
            inference_outputs["latent_unknown_attributes"]
        )

        return {
            LOSS_KEYS.RECONSTRUCTION: reconstruction_loss,
            LOSS_KEYS.UNKNOWN_ATTRIBUTE_PENALTY: unknown_attribute_penalty_loss_val,
        }

    @staticmethod
    def unknown_attribute_penalty_loss(latent_unknown_attributes: torch.Tensor) -> float:
        """Computes the content penalty term in the loss."""
        return torch.sum(latent_unknown_attributes**2, dim=1).mean()

    @torch.no_grad()
    def r2_metric(
            self,
            tensors: Dict[str, torch.Tensor],
            generative_outputs: Dict[str, torch.Tensor],
        ) -> Tuple[float, float]:
            """Evaluate the :math:`R^2` metric over gene expression.
            """


            x = tensors[self.x_loc].detach().cpu().numpy()  # (batch_size, n_genes)
            batch_size = x.shape[0]

            group_feat_list = []

            # 2.1 ordered attributes
            if self.eval_r2_ordered:
                for ordered_attribute_, dim_ in self.ordered_attributes_map.items():
                    attr_vals = tensors[ordered_attribute_]
                    if dim_ > 1:
                        # obsm: (batch_size, dim_)
                        group_feat_list.append(attr_vals.to(torch.float32))
                    else:
                        # obs: (batch_size,) -> (batch_size, 1)
                        group_feat_list.append(attr_vals.view(-1, 1).to(torch.float32))

            # 2.2 categorical attributes
            for categorical_attribute_ in self.categorical_attributes_map:
                cat_vals = tensors[categorical_attribute_].view(-1, 1)
                group_feat_list.append(cat_vals.to(torch.float32))

            if len(group_feat_list) == 0:
                group_features = torch.zeros(batch_size, 1, device=self.device)
            else:
                group_features = torch.cat(group_feat_list, dim=1)  # (batch_size, D_total)


            unique_groups, group_ids = torch.unique(
                group_features,
                dim=0,
                return_inverse=True,
            )

            r2_mean = 0.0
            r2_var = 0.0
            k = 0


            pred_x_mean = (
                torch.nan_to_num(
                    generative_outputs["means"],
                    nan=0,
                    neginf=0,
                    posinf=100,
                )
                .detach()
                .cpu()
                .numpy()
            )  # (batch_size, n_genes)

            pred_x_var = (
                torch.nan_to_num(
                    generative_outputs["variances"],
                    nan=0,
                    neginf=0,
                    posinf=100,
                )
                .detach()
                .cpu()
                .numpy()
            )  # (batch_size, n_genes)


            group_ids_np = group_ids.detach().cpu().numpy()

            for g in range(unique_groups.shape[0]):
                index_mask = (group_ids_np == g)
                if index_mask.sum() > 2:
                    x_index = x[index_mask]
                    means_index = pred_x_mean[index_mask]
                    variances_index = pred_x_var[index_mask]

                    true_mean_index = np.nanmean(x_index, axis=0)
                    pred_mean_index = np.nanmean(means_index, axis=0)

                    true_var_index = np.nanvar(x_index, axis=0)
                    if self.gene_likelihood in ["nb", "poisson"]:
                        pred_var_index = np.nanvar(means_index, axis=0)
                    else:
                        pred_var_index = np.nanmean(variances_index, axis=0)

                    r2_mean += r2_score(true_mean_index, pred_mean_index)
                    r2_var += r2_score(true_var_index, pred_var_index)
                    k += 1
                else:
                    continue

            if k > 0:
                return r2_mean / k, r2_var / k
            else:
                return r2_mean, r2_var
    @torch.no_grad()
    def cellwise_metrics(
        self,
        tensors: Dict[str, torch.Tensor],
        generative_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """Compute cell-level agreement metrics across genes.

        For each cell, the method compares the observed and predicted expression
        profiles across all genes and computes the coefficient of determination
        (R²), Pearson correlation, and Spearman rank correlation.

        Parameters
        ----------
        tensors
            Input tensors containing the observed gene-expression matrix.

        generative_outputs
            Outputs from the generative model containing the predicted expression
            means.

        Returns
        -------
        dict of str to numpy.ndarray
            Cell-level metric arrays. Each array has shape ``(batch_size,)`` and
            contains one value per cell. The returned dictionary contains:

            - ``"r2"``: coefficient of determination across genes.
            - ``"pearson"``: Pearson correlation across genes.
            - ``"spearman"``: Spearman rank correlation across genes.

        Notes
        -----
        A metric is returned as ``NaN`` when fewer than two finite gene values are
        available or when either the observed or predicted expression profile has
        zero variance.
        """
        # true
        x = tensors[self.x_loc].detach().cpu().numpy()  # (batch_size, n_genes)

        # predict mean
        pred_x = (
            torch.nan_to_num(
                generative_outputs["means"],
                nan=0,
                neginf=0,
                posinf=100,
            )
            .detach()
            .cpu()
            .numpy()
        )  # (batch_size, n_genes)

        n_cells, n_genes = x.shape
        r2 = np.full(n_cells, np.nan, dtype=np.float32)
        pearson = np.full(n_cells, np.nan, dtype=np.float32)
        spearman = np.full(n_cells, np.nan, dtype=np.float32)

        for i in range(n_cells):
            y_true = x[i]
            y_pred = pred_x[i]

            mask = np.isfinite(y_true) & np.isfinite(y_pred)
            if mask.sum() < 2:
                continue

            yt = y_true[mask]
            yp = y_pred[mask]

            if np.var(yt) == 0 or np.var(yp) == 0:
                continue

            # 1) R2
            r2[i] = r2_score(yt, yp)

            # 2) Pearson
            c = np.corrcoef(yt, yp)
            pearson[i] = c[0, 1]

            # 3) Spearman
            rank_true = np.argsort(np.argsort(yt))
            rank_pred = np.argsort(np.argsort(yp))
            cs = np.corrcoef(rank_true, rank_pred)
            spearman[i] = cs[0, 1]

        return {
            "r2": r2,
            "pearson": pearson,
            "spearman": spearman,
        }


    @torch.no_grad()
    def get_expression(
        self,
        tensors: Dict[str, torch.Tensor],
        sample: Optional[bool] = None,
        **inference_kwargs: Any,
    ) -> Tuple[torch.tensor, ...]:
        """Computes gene expression means and standard deviation.

        Parameters
        ----------
        tensors
            Considered inputs.
        inference_kwargs
            Additional arguments.

        Returns
        -------
        Prediction of gene expression mean and standard deviation.
        """
        return_sample = self.sample if sample is None else sample
        original_sample = self.sample
        self.sample = return_sample
        try:
            _, generative_outputs = self.forward(
                tensors,
                compute_loss=False,
                inference_kwargs=inference_kwargs,
            )
        finally:
            self.sample = original_sample

        mus = torch.nan_to_num(generative_outputs["means"], nan=0, neginf=0, posinf=100)  # batch_size, n_genes
        stds = torch.nan_to_num(generative_outputs["variances"], nan=0, neginf=0, posinf=100)  # batch_size, n_genes
        if sample is True:
            samples = torch.nan_to_num(generative_outputs["samples"], nan=0, neginf=0, posinf=100)
            return mus, stds, samples
        return mus, stds

