from __future__ import annotations

from typing import Any

import torch
from gymnasium import Space
from gymnasium.spaces import Discrete
from src.utils import (
    abilities,
    doubles_chunk_obs_len,
    doubles_glob_obs_len,
    items,
    moves,
    side_obs_len,
)
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.distributions import Distribution, MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import nn


class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args: Any, num_frames: int, **kwargs: Any):
        self.num_frames = num_frames
        self.actor_grad = True
        super().__init__(
            *args,
            **kwargs,
            net_arch=[],
            activation_fn=torch.nn.ReLU,
            features_extractor_class=AttentionExtractor,
            features_extractor_kwargs={"num_frames": num_frames},
            share_features_extractor=False,
            # optimizer_kwargs={"weight_decay": 1e-5},
        )

    @classmethod
    def clone(cls, model: BaseAlgorithm) -> MaskedActorCriticPolicy:
        assert isinstance(model.policy, MaskedActorCriticPolicy)
        new_policy = cls(
            model.observation_space,
            model.action_space,
            model.lr_schedule,
            num_frames=model.policy.num_frames,
        )
        new_policy.load_state_dict(model.policy.state_dict())
        return new_policy

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_logits, value_logits = self.get_logits(obs, actor_grad=True)
        distribution = self.get_dist_from_logits(obs, action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        if isinstance(distribution, MultiCategoricalDistribution):
            distribution2 = self.get_dist_from_logits(obs, action_logits, actions[:, :1])
            assert isinstance(distribution2, MultiCategoricalDistribution)
            actions2 = distribution2.get_actions(deterministic=deterministic)
            distribution.distribution[1] = distribution2.distribution[1]
            actions[:, 1] = actions2[:, 1]
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, value_logits, log_prob

    def evaluate_actions(
        self, obs: PyTorchObs, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        assert isinstance(obs, torch.Tensor)
        if obs.device != actions.device:
            actions = actions.to(obs.device)
        action_logits, value_logits = self.get_logits(obs, self.actor_grad)
        distribution = self.get_dist_from_logits(obs, action_logits)
        if isinstance(distribution, MultiCategoricalDistribution):
            distribution2 = self.get_dist_from_logits(obs, action_logits, actions[:, :1])
            assert isinstance(distribution2, MultiCategoricalDistribution)
            distribution.distribution[1] = distribution2.distribution[1]
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return value_logits, log_prob, entropy

    def get_logits(self, obs: torch.Tensor, actor_grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        actor_context = torch.enable_grad() if actor_grad else torch.no_grad()
        features = self.extract_features(obs)  # type: ignore
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            with actor_context:
                latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        with actor_context:
            action_logits = self.action_net(latent_pi)
        value_logits = self.value_net(latent_vf)
        return action_logits, value_logits

    def get_dist_from_logits(
        self, obs: torch.Tensor, action_logits: torch.Tensor, action: torch.Tensor | None = None
    ) -> Distribution:
        mask = self.get_mask(obs, action)
        distribution = self.action_dist.proba_distribution(action_logits + mask)
        return distribution

    def get_mask(self, obs: torch.Tensor, ally_actions: torch.Tensor | None = None) -> torch.Tensor:
        if isinstance(self.action_space, Discrete):
            mask = obs[:, 0, : self.action_space.n]  # type: ignore
            mask = torch.where(mask.sum(dim=1, keepdim=True) == mask.size(1), 0.0, mask)
            mask = torch.where(mask == 1, float("-inf"), mask)
            return mask
        else:
            act_len = self.action_space.nvec[0]  # type: ignore
            if ally_actions is None:
                mask = obs[:, 0, 0, : 2 * act_len]
            else:
                mask = obs[:, 0, 0, act_len : 2 * act_len]
                ally_switched = (1 <= ally_actions) & (ally_actions <= 6)
                ally_terastallized = ally_actions >= 87
                # creating a (batch_size, act_len) size array of 0..act_len - 1 ranges
                indices = torch.arange(act_len, device=mask.device).unsqueeze(0).expand_as(mask)
                # marking values in indices as being invalid actions due to ally action
                ally_mask = (
                    ((27 <= indices) & (indices < 87))
                    | ((indices >= 87) & ally_terastallized)
                    | ((indices == ally_actions) & ally_switched)
                )
                mask = torch.where(ally_mask, 1.0, mask)
                mask = torch.cat([obs[:, 0, 0, :act_len], mask], dim=1)
            mask = torch.where(mask == 1, float("-inf"), mask)
            return mask


class AttentionExtractor(BaseFeaturesExtractor):
    num_pokemon: int = 12
    embed_len: int = 32
    proj_len: int = 128

    def __init__(self, observation_space: Space[Any], num_frames: int):
        super().__init__(observation_space, features_dim=self.proj_len)
        self.ability_embed = nn.Embedding(len(abilities), self.embed_len)
        self.item_embed = nn.Embedding(len(items), self.embed_len)
        self.move_embed = nn.Embedding(len(moves), self.embed_len)
        self.feature_proj = nn.Linear(
            doubles_chunk_obs_len + 6 * (self.embed_len - 1), self.proj_len
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.proj_len))
        self.frame_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.proj_len,
                nhead=4,
                dim_feedforward=self.proj_len,
                dropout=0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=3,
        )
        self.frame_encoding: torch.Tensor
        self.register_buffer("frame_encoding", torch.eye(num_frames).unsqueeze(0))
        self.frame_proj = nn.Linear(self.proj_len + num_frames, self.proj_len)
        self.mask: torch.Tensor
        self.register_buffer(
            "mask", nn.Transformer.generate_square_subsequent_mask(num_frames).transpose(0, 1)
        )
        self.meta_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.proj_len,
                nhead=4,
                dim_feedforward=self.proj_len,
                dropout=0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        num_frames = x.size(1)
        # embedding
        start = doubles_glob_obs_len + side_obs_len
        x = torch.cat(
            [
                x[:, :, :, :start],
                self.ability_embed.forward(x[:, :, :, start].long()),
                self.item_embed.forward(x[:, :, :, start + 1].long()),
                self.move_embed.forward(x[:, :, :, start + 2].long()),
                self.move_embed.forward(x[:, :, :, start + 3].long()),
                self.move_embed.forward(x[:, :, :, start + 4].long()),
                self.move_embed.forward(x[:, :, :, start + 5].long()),
                x[:, :, :, start + 6 :],
            ],
            dim=3,
        )
        # frame encoder
        x = x.view(batch_size * num_frames, self.num_pokemon, -1)
        x = self.feature_proj.forward(x)
        token = self.cls_token.expand(batch_size * num_frames, -1, -1)
        x = torch.cat([token, x], dim=1)
        x = self.frame_encoder.forward(x)[:, 0, :]
        x = x.view(batch_size, num_frames, -1)
        # meta encoder
        frame_encoding = self.frame_encoding.expand(batch_size, -1, -1)
        x = torch.cat([x, frame_encoding], dim=2)
        x = self.frame_proj.forward(x)
        return self.meta_encoder.forward(x, mask=self.mask, is_causal=True)[:, 0, :]
