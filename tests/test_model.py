import torch
from torch.distributions import Categorical

import features
from model import ActorCritic


def test_forward_shapes():
    model = ActorCritic()
    logits, value = model(torch.zeros(7, features.FEAT_DIM))
    assert logits.shape == (7, 15)
    assert value.shape == (7,)


def test_dist_slices_match_nvec():
    # A wrong slice here silently maps actions to the wrong head.
    model = ActorCritic()
    logits, _ = model(torch.zeros(5, features.FEAT_DIM))
    dist = model.make_dist(logits)
    sample = dist.sample()
    assert sample.shape == (5, 4)
    nvec = [5, 2, 5, 3]
    for i, n in enumerate(nvec):
        assert int(sample[:, i].min()) >= 0 and int(sample[:, i].max()) < n

    expected = sum(
        Categorical(logits=logits[..., start:start + n]).log_prob(sample[..., i]).sum()
        for i, (start, n) in enumerate([(0, 5), (5, 2), (7, 5), (12, 3)])
    )
    assert torch.allclose(dist.log_prob(sample).sum(), expected)


def test_entropy_is_sum_of_factor_entropies():
    model = ActorCritic()
    logits, _ = model(torch.zeros(3, features.FEAT_DIM))
    dist = model.make_dist(logits)
    expected = sum(
        Categorical(logits=logits[..., start:start + n]).entropy().sum()
        for start, n in [(0, 5), (5, 2), (7, 5), (12, 3)]
    )
    assert torch.allclose(dist.entropy().sum(), expected)
    assert torch.all(dist.entropy() > 0)
