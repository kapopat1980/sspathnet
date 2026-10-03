import numpy as np
import pytest
import torch
import torch.nn.functional as F

K = 38


def test_parameter_budget(p2):
    n = p2.count_params(p2.SSPathNet(K)) / 1e6
    assert 9.70 < n < 9.90, n                      # paper: 9.797 M


def test_forward_shapes(p2):
    net = p2.SSPathNet(K).eval()
    logits, emb = net(torch.rand(2, 3, 64, 64))
    assert logits.shape == (2, K) and emb.shape == (2, 256)


@pytest.mark.parametrize("idx", range(9))
def test_every_ablation_variant_trains(p2, idx):
    name, kw, _ = p2.ABLATIONS[idx]
    net = p2.SSPathNet(K, **kw).train()
    logits, _ = net(torch.rand(2, 3, 64, 64))
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()
    dead = [n for n, p in net.named_parameters() if p.grad is None]
    assert not dead, (name, dead[:3])


def test_ablation_param_counts_match_paper(p2):
    expect = {"A1_stem_rgb_only": 9.75, "A5_no_attention": 8.54, "A6_agg_none": 9.54, "A7_agg_concat": 10.00}
    for name, kw, _ in p2.ABLATIONS:
        if name in expect:
            assert abs(p2.count_params(p2.SSPathNet(K, **kw)) / 1e6 - expect[name]) < 0.01, name


def test_dct_basis_is_orthonormal(p2):
    b = p2.BlockDCT(4)
    w = b.w[:16, 0].reshape(16, 16)
    assert torch.allclose(w @ w.T, torch.eye(16), atol=1e-5)


def test_hsv_features(p2):
    x = torch.rand(2, 3, 8, 8)
    h = p2.RGB2HSVFeat()(x)
    assert h.shape == (2, 4, 8, 8) and h.min() >= -1.0001 and h.max() <= 1.0001
    grey = torch.full((1, 3, 4, 4), 0.4)
    hg = p2.RGB2HSVFeat()(grey)
    assert torch.allclose(hg[:, :3], torch.tensor([0.0, 0.0, 0.0]).view(1, 3, 1, 1).expand(1, 3, 4, 4), atol=1e-5)  # zero chroma
    assert torch.allclose(hg[:, 3], torch.full((1, 4, 4), 0.4))


def test_window_partition_roundtrip(p2):
    t = torch.randn(2, 8, 10, 10)
    assert torch.equal(p2.window_reverse(p2.window_partition(t, 5), 5, 2, 8, 10, 10), t)


def test_attention_handles_non_divisible_sizes(p2):
    out = p2.WindowCrossAttention(64, 2, 5)(torch.randn(2, 64, 8, 8))
    assert out.shape == (2, 64, 8, 8)


def test_deformable_conv_starts_as_ordinary_depthwise_conv(p2):
    torch.manual_seed(0)
    d, r = p2.DeformDWConv(8, deform=True), p2.DeformDWConv(8, deform=False)
    r.weight.data, r.bias.data = d.weight.data.clone(), d.bias.data.clone()
    x = torch.randn(2, 8, 12, 12)
    assert torch.allclose(d(x), r(x), atol=1e-5)    # zero-initialised offsets


def test_adaptive_activation_limits(p2):
    act = p2.AdaptiveAct(4)
    act.raw_a.data.fill_(30.0)                      # a -> 1  => identity
    x = torch.randn(3, 4)
    assert torch.allclose(act(x), x, atol=1e-4)
    act.raw_a.data.fill_(-30.0)                     # a -> 0  => x * sigmoid(beta x)
    beta = F.softplus(act.raw_beta)
    assert torch.allclose(act(x), x * torch.sigmoid(beta * x), atol=1e-5)


def test_center_loss_moves_centres_towards_embeddings(p2):
    cl = p2.CenterLoss(3, 4)
    emb, y = torch.ones(6, 4), torch.tensor([0, 0, 1, 1, 2, 2])
    l0 = cl(emb, y).item()
    l1 = cl(emb, y).item()
    assert l1 < l0 and cl.centers.abs().sum() > 0


def test_gpu_augment_keeps_shape_and_range(p2):
    x = torch.rand(4, 3, 64, 64)
    y = p2.gpu_augment(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1


def test_baseline_parameter_counts(p2):
    expect = dict(mobilenet_v2=2.27, resnet18=11.20, efficientnet_b0=4.06, densenet121=6.99)
    for name, m in expect.items():
        assert abs(p2.count_params(p2.build_baseline(name, K)) / 1e6 - m) < 0.01, name


def test_splits_are_deterministic_stratified_and_disjoint(p2):
    p2.DATA_Y = torch.from_numpy(np.repeat(np.arange(K), 40))
    a, b = p2.fixed_split(), p2.fixed_split()
    for k in a:
        assert np.array_equal(a[k], b[k])
    s = [set(a[k].tolist()) for k in ("train", "val", "test")]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
    assert sum(map(len, s)) == K * 40
    assert len(np.unique(p2.DATA_Y.numpy()[a["test"]])) == K       # every class in the test split


def test_split_code_identical_in_both_parts(p1, p2):
    import inspect
    assert inspect.getsource(p1.fixed_split) == inspect.getsource(p2.fixed_split)
    assert p1.SEED == p2.SEED == 42


def test_cross_dataset_class_mapper(p2):
    pv = ["Apple___Apple_scab", "Apple___healthy", "Pepper,_bell___healthy", "Pepper,_bell___Bacterial_spot",
          "Tomato___Late_blight", "Tomato___Septoria_leaf_spot"]
    m = p2.map_classes(["Tomato leaf late blight", "Tomato Septoria leaf spot", "Apple leaf", "Bell_pepper leaf", "grape leaf"], pv)
    assert m["Tomato leaf late blight"] == "Tomato___Late_blight"
    assert m["Tomato Septoria leaf spot"] == "Tomato___Septoria_leaf_spot"
    assert m["Apple leaf"] == "Apple___healthy"
    assert m["Bell_pepper leaf"] == "Pepper,_bell___healthy"
    assert "grape leaf" not in m
