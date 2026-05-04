"""
QFE-COD Model Definitions
Extracted from qfe-cod.ipynb training notebook.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
import timm

# ── DWT Decomposition ─────────────────────────────────────────────────────────
class DWTDecomp(nn.Module):
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        x = x[:, :, :H - H%2, :W - W%2]
        x00, x01, x10, x11 = x[:,:,0::2,0::2], x[:,:,0::2,1::2], x[:,:,1::2,0::2], x[:,:,1::2,1::2]
        LL = (x00 + x01 + x10 + x11) * 0.5
        LH = (-x00 - x01 + x10 + x11) * 0.5
        HL = (-x00 + x01 - x10 + x11) * 0.5
        HH = (x00 - x01 - x10 + x11) * 0.5
        return LL, LH, HL, HH

# ── FSA ───────────────────────────────────────────────────────────────────────
class FSA(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(in_ch, 2)
        self.attn = nn.MultiheadAttention(embed_dim=2, num_heads=2, batch_first=True)
        self.out_proj = nn.Linear(8, 8)
    def forward(self, ll, lh, hl, hh):
        bands = [ll, lh, hl, hh]
        tokens = [self.proj(self.pool(b).squeeze(-1).squeeze(-1)) for b in bands]
        tokens = torch.stack(tokens, dim=1)
        out, _ = self.attn(tokens, tokens, tokens)
        return self.out_proj(out.reshape(out.size(0), -1))

# ── SimpleSSM / Mamba ─────────────────────────────────────────────────────────
class SimpleSSM(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3, groups=d_inner)
        self.out_proj = nn.Linear(d_inner, d_model)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        residual = x
        xz = self.in_proj(x)
        x2, z = xz.chunk(2, dim=-1)
        x2 = self.conv1d(x2.transpose(1,2))[:, :, :x.size(1)].transpose(1,2)
        x2 = F.silu(x2)
        y = x2 * F.silu(z)
        return self.norm(self.out_proj(y) + residual)

class MambaBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ssm = SimpleSSM(channels)
    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        x_seq = self.ssm(x_seq)
        return x_seq.transpose(1, 2).reshape(B, C, H, W)

# ── Backbones ─────────────────────────────────────────────────────────────────
class PVTv2Backbone(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = timm.create_model('pvt_v2_b4', pretrained=pretrained, features_only=True, out_indices=(0,1,2,3))
        self.channels = self.model.feature_info.channels()
    def forward(self, x):
        return self.model(x)

# ── FPN Decoder ───────────────────────────────────────────────────────────────
class FPNDecoder(nn.Module):
    def __init__(self, in_channels=(64,128,320,512), out_ch=64):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out_ch, 1) for c in in_channels])
        self.mamba = nn.ModuleList([MambaBlock(out_ch) for _ in in_channels])
        self.fuse = nn.ModuleList([
            nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU())
            for _ in in_channels[:-1]
        ])
    def forward(self, feats):
        lats = [self.lat[i](feats[i]) for i in range(4)]
        p = self.mamba[3](lats[3])
        outs = [p]
        for i in range(2, -1, -1):
            p = F.interpolate(p, size=lats[i].shape[2:], mode='bilinear', align_corners=False)
            p = self.fuse[i](p + lats[i])
            p = self.mamba[i](p)
            outs.append(p)
        return outs

# ── HQCM (Dummy fallback for non-CUDA) ───────────────────────────────────────
class DummyHQCM(nn.Module):
    def __init__(self, nq=8):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(nq, nq*4), nn.GELU(), nn.Linear(nq*4, nq))
        self.ln = nn.LayerNorm(nq)
    def forward(self, x):
        return self.ln(self.fc(x))

# ── Quantum HQCM ─────────────────────────────────────────────────────────────
def make_quantum_hqcm(nq=8, nl=4):
    try:
        import pennylane as qml
        class QuantumHQCM(nn.Module):
            def __init__(self):
                super().__init__()
                self.nq = nq
                self.dev = qml.device('default.qubit', wires=nq)
                @qml.qnode(self.dev, interface='torch', diff_method='backprop')
                def circuit(inputs, weights):
                    qml.AngleEmbedding(inputs, wires=range(nq), rotation='Y')
                    qml.StronglyEntanglingLayers(weights, wires=range(nq))
                    return [qml.expval(qml.PauliZ(i)) for i in range(nq)]
                self.circuit = circuit
                self.weights = nn.Parameter(torch.randn(qml.StronglyEntanglingLayers.shape(n_layers=nl, n_wires=nq)) * 0.1)
                self.pre_ln = nn.LayerNorm(nq)
                self.post_ln = nn.LayerNorm(nq)
            def forward(self, x):
                orig_dtype, orig_device = x.dtype, x.device
                x = torch.tanh(self.pre_ln(x)) * math.pi
                x_cpu = x.detach().cpu().double()
                w_cpu = self.weights.detach().cpu().double()
                results = [torch.stack(self.circuit(x_cpu[i], w_cpu)) for i in range(x_cpu.shape[0])]
                return self.post_ln(torch.stack(results).to(dtype=orig_dtype, device=orig_device))
        return QuantumHQCM()
    except Exception:
        return DummyHQCM(nq)

# ── Spline + Q-WaveKAN Head ───────────────────────────────────────────────────
class SplineActivation(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.knots = nn.Parameter(torch.linspace(-3, 3, n))
        self.coeffs = nn.Parameter(torch.zeros(n))
    def forward(self, x):
        d = x.unsqueeze(-1) - self.knots.view(1, 1, 1, -1)
        return x + (torch.exp(-0.5 * d**2) * self.coeffs).sum(-1)

class QWaveKANHead(nn.Module):
    def __init__(self, in_ch=64, fd=8):
        super().__init__()
        self.q_gate = nn.Sequential(nn.Linear(fd, in_ch), nn.Sigmoid())
        self.spline = SplineActivation()
        self.seg_head = nn.Conv2d(in_ch, 1, 1)
        self.uncert_head = nn.Conv2d(in_ch, 1, 1)
        self.edge_head = nn.Conv2d(in_ch, 1, 1)
    def forward(self, feat, fv):
        g = self.spline(feat * self.q_gate(fv)[:, :, None, None])
        return self.seg_head(g), self.uncert_head(g).abs(), self.edge_head(g)

class DualDomainFusion(nn.Module):
    def __init__(self, sc, fd=8):
        super().__init__()
        self.fe = nn.Sequential(nn.Linear(fd, sc), nn.Sigmoid())
        self.conv = nn.Sequential(nn.Conv2d(sc, sc, 3, padding=1), nn.BatchNorm2d(sc), nn.ReLU())
    def forward(self, sp, fv):
        return self.conv(sp * self.fe(fv)[:, :, None, None] + sp)

# ── Full QFE-COD Model ────────────────────────────────────────────────────────
class QFECOD(nn.Module):
    def __init__(self, pretrained=False, use_quantum=False):
        super().__init__()
        self.backbone = PVTv2Backbone(pretrained)
        ch = self.backbone.channels
        self.dwt = DWTDecomp()
        self.fsa = FSA(ch[3])
        self.hqcm = make_quantum_hqcm() if use_quantum else DummyHQCM()
        self.fpn = FPNDecoder(ch, 64)
        self.fusion = DualDomainFusion(64)
        self.head = QWaveKANHead()
    def forward(self, x):
        feats = self.backbone(x)
        fv = self.hqcm(self.fsa(*self.dwt(feats[3])))
        fine = self.fpn(feats)[-1]
        fused = self.fusion(fine, fv)
        seg, unc, edge = self.head(fused, fv)
        H, W = x.shape[2:]
        return (
            F.interpolate(seg, (H, W), mode='bilinear', align_corners=False),
            F.interpolate(unc, (H, W), mode='bilinear', align_corners=False),
            F.interpolate(edge, (H, W), mode='bilinear', align_corners=False),
            fused
        )

# ── CAM Classifier ────────────────────────────────────────────────────────────
class CAMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=False, features_only=False, num_classes=0)
        fd = self.backbone.num_features
        self.head = nn.Sequential(nn.Identity(), nn.Linear(fd, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(self, x):
        return self.head(self.backbone(x))

# ── SINetV2 Baseline ──────────────────────────────────────────────────────────
class SINetV2(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        bb = timm.create_model('resnet50', pretrained=pretrained, features_only=True, out_indices=(1,2,3,4))
        self.enc = bb
        self.chs = bb.feature_info.channels()
        self.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in self.chs])
        self.sm = nn.ModuleList([nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()) for _ in range(4)])
        self.im = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, groups=64), nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.seg_head = nn.Conv2d(64, 1, 1)
        self.edge_head = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        feats = self.enc(x)
        lats = [self.lat[i](feats[i]) for i in range(4)]
        p = self.sm[3](lats[3])
        for i in range(2, -1, -1):
            p = self.sm[i](F.interpolate(p, size=lats[i].shape[2:], mode='bilinear', align_corners=False) + lats[i])
        p = self.im(p)
        H, W = x.shape[2:]
        seg = F.interpolate(self.seg_head(p), (H, W), mode='bilinear', align_corners=False)
        edge = F.interpolate(self.edge_head(p), (H, W), mode='bilinear', align_corners=False)
        return seg, torch.zeros_like(seg), edge, p
