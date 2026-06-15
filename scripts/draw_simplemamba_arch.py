from graphviz import Digraph

dot = Digraph("SimpleMamba_Architecture", format="png")
dot.attr(rankdir="LR", bgcolor="white", splines="ortho")
dot.attr("node", shape="rect", style="rounded,filled", fontname="Arial", fontsize="11",
         color="#5B6472", fillcolor="#F6F8FA")
dot.attr("edge", color="#5B6472", arrowsize="0.8")

dot.node("input", "Input Video\n[1,160,3,128,128]", fillcolor="#E8F1FF")

with dot.subgraph(name="cluster_fusion") as c:
    c.attr(label="Fusion Stem", color="#A9B7C6", style="rounded")
    c.node("rgb", "RGB Appearance Branch\nConv2d + BN + ReLU + MaxPool\n[160,3,128,128] → [160,12,32,32]",
           fillcolor="#FDF2E9")
    c.node("diff", "Temporal Difference Branch\nFrame Difference + Conv2d\n[160,12,128,128] → [160,12,32,32]",
           fillcolor="#FDF2E9")
    c.node("fusion", "Weighted Feature Fusion\nα·RGB + β·Diff\n→ [160,24,16,16]",
           fillcolor="#FFF7DC")
    c.edge("rgb", "fusion")
    c.edge("diff", "fusion")

dot.node("reshape", "Reshape + Permute\n[1,24,160,16,16]", fillcolor="#EEF6EA")
dot.node("stem3", "3D Conv Stem\nConv3d + BN\n[1,24,160,16,16]\n→ [1,96,80,16,16]", fillcolor="#EEF6EA")
dot.node("mask", "Spatial Attention Mask\nSigmoid + Normalization\n[1,96,80,16,16]", fillcolor="#EAF6F6")
dot.node("pool", "Global Spatial Average Pooling\nMean over H,W\n[1,96,80]", fillcolor="#EAF6F6")
dot.node("rearrange", "Temporal Token Rearrangement\nb c t → b t c\n[1,80,96]", fillcolor="#F0ECFA")

with dot.subgraph(name="cluster_block") as c:
    c.attr(label="Block_mamba × 24", color="#A9B7C6", style="rounded")
    c.node("ln1", "LayerNorm", fillcolor="#F0ECFA")
    c.node("mamba", "Bidirectional Mamba Encoder\nEncoderLayer × 2\n[1,80,96] → [1,80,96]",
           fillcolor="#EDE7F6")
    c.node("res1", "Residual Add", fillcolor="#F6F8FA")
    c.node("ln2", "LayerNorm", fillcolor="#F0ECFA")
    c.node("fan", "FANLayer\nLinear → cos / sin\n+ Gated GELU\n[1,80,96] → [1,80,96]",
           fillcolor="#EDE7F6")
    c.node("res2", "Residual Add", fillcolor="#F6F8FA")

    c.edge("ln1", "mamba")
    c.edge("mamba", "res1")
    c.edge("res1", "ln2")
    c.edge("ln2", "fan")
    c.edge("fan", "res2")

dot.node("permute1", "Permute\n[1,80,96] → [1,96,80]", fillcolor="#F0ECFA")
dot.node("upsample", "Temporal Upsample ×2\n[1,96,80] → [1,96,160]", fillcolor="#E8F1FF")
dot.node("conv1d", "1D Conv rPPG Head\nConv1d\n[1,96,160] → [1,1,160]", fillcolor="#E8F1FF")
dot.node("permute2", "Permute\n[1,1,160] → [1,160,1]", fillcolor="#F0ECFA")
dot.node("edl", "Evidential Uncertainty Head\nLogNormalInvGamma\nLinear: 1 → 4\nμ, v, α, β", fillcolor="#FCE8E6")
dot.node("output", "Output\nUncertainty Params\n[1,160,4]", fillcolor="#FCE8E6")

dot.edge("input", "rgb")
dot.edge("input", "diff")
dot.edge("fusion", "reshape")
dot.edge("reshape", "stem3")
dot.edge("stem3", "mask")
dot.edge("mask", "pool")
dot.edge("pool", "rearrange")
dot.edge("rearrange", "ln1")
dot.edge("res2", "permute1")
dot.edge("permute1", "upsample")
dot.edge("upsample", "conv1d")
dot.edge("conv1d", "permute2")
dot.edge("permute2", "edl")
dot.edge("edl", "output")

dot.render("simplemamba_architecture_simple", cleanup=True)
print("Saved: simplemamba_architecture_simple.png")