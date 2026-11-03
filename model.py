"""Shared-trunk actor-critic.

One trunk feeds both heads, so the critic sees the same learned features as the
actor (SB3 builds two independent MLPs; after imitation its critic is noise and
PPO's advantages are noise too).

The actor emits 15 logits split into 4 independent Categoricals — the macro is
`MultiDiscrete([5, 2, 5, 3])`, not a joint 150-way choice.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from features import FEAT_DIM


class MultiCategorical:
    """Product of independent Categoricals (torch's Independent rejects a list)."""

    def __init__(self, dists):
        self.dists = dists

    def sample(self):
        return torch.stack([d.sample() for d in self.dists], dim=-1)

    def mode(self):
        return torch.stack([d.probs.argmax(dim=-1) for d in self.dists], dim=-1)

    def log_prob(self, actions):
        return sum(d.log_prob(actions[..., i]) for i, d in enumerate(self.dists))

    def entropy(self):
        return sum(d.entropy() for d in self.dists)


class ActorCritic(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, 15)  # 5 + 2 + 5 + 3
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)

    @staticmethod
    def make_dist(logits):
        return MultiCategorical([
            Categorical(logits=logits[..., :5]),     # crop_focus
            Categorical(logits=logits[..., 5:7]),    # buy_land
            Categorical(logits=logits[..., 7:12]),   # n_hires
            Categorical(logits=logits[..., 12:15]),  # sell_mode
        ])
