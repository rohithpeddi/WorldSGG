"""Compact, vector-native paper figures; preserves the detailed reference set.

Run: python scripts/paper_figures/generate_publication.py
Outputs: worldwise_paper, worldwise_plus_paper, worldwise_pp_paper.
"""
from common import Diagram, COLORS, INK, parser
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch


def tag(d, x, y, text, color="blue"):
    label = d.text(x, y, text, size=6.8, color=COLORS[color][1])
    label.set_bbox(dict(facecolor="white", edgecolor="none", pad=1.5))
    label.set_zorder(5)


def panel(d, x, y, w, h, title):
    d.ax.add_patch(FancyBboxPatch((x, y), w, h,
                  boxstyle="round,pad=0,rounding_size=.12",
                  facecolor="#FAFBFC", edgecolor="#DCE3E8", lw=.65, zorder=0))
    d.text(x+.18, y+h-.28, title, ha="left", size=7.1, weight="bold")


def tokens(d, x, y, count=5, masked=(), size=.28, gap=.1):
    """Schematic token sequence: solid = visible, hatched = masked."""
    for i in range(count):
        fill, edge = COLORS["orange" if i in masked else "blue"]
        d.ax.add_patch(Rectangle((x+i*(size+gap), y), size, size,
                                facecolor=fill, edgecolor=edge, lw=.75,
                                hatch="///" if i in masked else None, zorder=3))


def grid(d, x, y, color="blue"):
    fill, edge = COLORS[color]
    for row in range(3):
        for col in range(5):
            d.ax.add_patch(Rectangle((x+col*.19,y+row*.19),.16,.16,
                                    facecolor=fill,edgecolor=edge,lw=.45,zorder=3))


def graph(d, x, y):
    centers = [(x,y+.42),(x+.8,y+.75),(x+1.35,y+.12)]
    for a,b in [(0,1),(0,2),(1,2)]:
        d.arrow([centers[a],centers[b]], color=COLORS["orange"][1])
    for i,(cx,cy) in enumerate(centers):
        d.ax.add_patch(Circle((cx,cy),.115,facecolor=COLORS["orange"][0],
                              edgecolor=COLORS["orange"][1],lw=.9,zorder=4))


def footer(d, text):
    d.ax.plot([.4,15.6],[.93,.93],color="#DCE3E8",lw=.8)
    d.text(.4,.62,text,ha="left",size=6.8)
    d.text(.4,.24,"T: frames   N: world slots   K: relation pairs   C: object classes   |   Widths follow the main configurations.",
           ha="left",size=6.4,color="#516271")


def worldwise_paper():
    d=Diagram("WorldWise", "Recover persistent object tokens through masked world auto-encoding",height=9.8)
    d.section(.4,8.12,"(a) Geometry-conditioned recovery")
    d.box(.4,5.9,2.6,1.6,"Decoded ROIs","Object ROI: 1024\nProject → 256","gray")
    d.box(3.65,5.9,2.55,1.6,"Scaffold fusion","ROI or [MASK]\n832 → 256")
    d.box(6.85,5.9,2.8,1.6,"Temporal retrieval","Same-object memory\nVisible keys / values","teal")
    d.box(10.3,5.9,2.4,1.6,"Object reasoning","+ Visibility\nSpatial attention","teal")
    d.box(13.35,5.9,2.25,1.6,"Node head","256 → C\nObject logits","orange")
    for a,b in [(3,3.65),(6.2,6.85),(9.65,10.3),(12.7,13.35)]:
        d.arrow([(a,6.7),(b,6.7)])
    tag(d,7.99,7.77,"(T, N, 256)","teal")
    tag(d,11.5,7.77,"(T, N, 256)","teal")
    d.box(.4,4.25,5.8,.85,"Geo 256 + cam 128 + motion 64 + ego 128",color="gray")
    d.arrow([(4.92,5.1),(4.92,5.9)])
    tag(d,4.92,5.5,"576 + 256")
    d.arrow([(9.96,6.7),(9.96,4.3),(9.45,4.3)],dashed=True,color=COLORS["purple"][1])
    d.box(6.85,3.7,2.6,1.25,"Reconstruction","256 → 256\nEMA target","purple")
    d.text(8.15,3.38,"after visibility embedding",size=6.1,color=COLORS["purple"][1])
    d.section(13.35,5.18,"(b) Relations")
    d.box(10.3,2.65,2.4,1.7,"Pair fusion","896 → 256\nRelation attention\nTemporal edges")
    d.arrow([(11.5,5.9),(11.5,4.35)])
    tag(d,11.5,4.67,"person + object")
    d.box(13.35,2.65,2.25,1.7,"Edge heads","256 → 128\n→ 3 / 6 / 17","orange")
    d.arrow([(12.7,3.5),(13.35,3.5)])
    tag(d,14.45,4.63,"(T, K, 256)")
    d.text(11.5,2.06,"Union 1024 → 64\nText 2 × 128; geometry 8 → 64",size=6.2)
    d.arrow([(11.5,2.42),(11.5,2.65)])
    graph(d,13.85,1.4)
    d.arrow([(14.5,2.65),(14.5,2.2)])
    panel(d,.4,1.45,5.8,2.35,"(c) One persistent slot across time")
    tokens(d,.85,2.43,count=5,masked=(2,),size=.45,gap=.55)
    for i,t in enumerate(["t₁","t₂","t₃","t₄","t₅"]):
        d.text(1.075+i,2.13,t,size=6.8)
    d.arrow([(1.075,2.92),(1.075,3.08),(3.075,3.08),(3.075,2.92)],color=COLORS["teal"][1])
    d.arrow([(5.075,2.92),(5.075,3.08),(3.075,3.08),(3.075,2.92)],color=COLORS["teal"][1])
    d.text(3.3,1.72,"Observed tokens recover the masked appearance.",size=6.7)
    footer(d,"Training only: 30% visible masking + clean simulated-unseen labels + EMA reconstruction; τ = 0.5, λ_vlm = 0.")
    return d


def worldwise_plus_paper():
    d=Diagram("WorldWise+", "Change the visual representation; retain the WorldWise reasoning core",height=9.8)
    d.section(.4,8.12,"(a) Frozen foundation features")
    d.section(6.4,8.12,"(b) Two appearance projectors")
    d.box(.4,5.8,2.6,1.65,"DINOv3-L","L16 / L20 / L24n\n1024 per layer","gray")
    d.box(.4,3.55,2.6,1.65,"Optional π³","F4 / F14 / G14\n1024 per layer","gray",dashed=True)
    d.box(3.65,5.8,2.1,1.65,"ROI pooling","7 × 7 → mean\n3 layers → 3072")
    d.arrow([(3,6.62),(3.65,6.62)])
    d.arrow([(3,4.37),(3.3,4.37),(3.3,6.1),(3.65,6.1)],dashed=True)
    d.box(3.65,3.55,2.1,1.65,"Same boxes","Object ROIs\nUnion ROIs","gray")
    d.arrow([(4.7,5.2),(4.7,5.8)])
    d.box(6.4,5.8,3.5,1.65,"Object projector","3072 / 6144 → 256\n(T, N, 256)","teal")
    d.box(6.4,3.55,3.5,1.65,"Union projector","3072 / 6144 → 64\n(T, K, 64)","teal")
    d.arrow([(5.75,6.62),(6.4,6.62)])
    d.arrow([(6.05,6.62),(6.05,4.37),(6.4,4.37)])
    panel(d,10.6,3.55,5,3.9,"(c) Optional per-dimension fusion")
    d.box(10.95,5.65,1.85,.75,"DINO → d",color="blue")
    d.box(13.4,5.65,1.85,.75,"π³ → d",color="orange")
    d.arrow([(11.875,5.65),(11.875,5.3),(13.1,5.3),(13.1,4.96)])
    d.arrow([(14.325,5.65),(14.325,5.3),(13.1,5.3),(13.1,4.96)])
    d.box(11.3,4.13,3.6,.83,"Gated weighted sum",color="teal")
    d.text(13.1,3.82,"d = 256 (object) or 64 (union)",size=6.7)
    d.text(13.1,6.75,"Two 3072-d streams; gates of shape (2, d)",size=6.6)
    d.section(.4,2.99,"(d) Inherited WorldWise computation")
    d.box(.4,1.48,3.6,1.05,"Scaffold fusion","832 → 256; (T, N, 256)")
    d.box(4.7,1.48,3.6,1.05,"Retrieve + reason","(T, N, 256) at each stage","teal")
    d.box(9,1.48,3.5,1.05,"Relation reasoning","896 → 256; (T, K, 256)")
    d.box(13.2,1.48,2.4,1.05,"Graph heads","C; 3 / 6 / 17","orange")
    for a,b in [(4,4.7),(8.3,9),(12.5,13.2)]:
        d.arrow([(a,2),(b,2)])
    d.arrow([(8.15,5.8),(8.15,5.51),(6.19,5.51),(6.19,3.26),(.2,3.26),(.2,2),(.4,2)])
    d.arrow([(6.5,1.48),(6.5,1.15),(14.4,1.15),(14.4,1.48)])
    tag(d,10.5,1.15,"node readout")
    d.arrow([(8.15,3.55),(8.15,3.2),(10.75,3.2),(10.75,2.53)])
    footer(d,"Headline: DINOv3 only (3072-d). Optional fusion: DINOv3 + π³ (6144-d). Same masking, EMA and scene-graph losses.")
    return d


def worldwise_pp_paper():
    d=Diagram("WorldWise++", "Ground persistent entities in image tokens and train detection jointly",height=10.3)
    d.section(.4,8.62,"(a) Persistent queries + image memory")
    d.box(.4,6.75,3.4,1.35,"World-slot scaffold","ROI 3072 → 256\nFusion 832 → 256")
    d.box(.4,4.95,3.4,1.05,"Free object queries","(T, Q, 256); Q = 30","orange")
    panel(d,.4,1.58,3.4,2.95,"Frozen token grids")
    grid(d,.86,3.04,"blue")
    grid(d,2.29,3.04,"orange")
    d.text(1.28,2.78,"DINOv3",size=6.5)
    d.text(2.7,2.78,"π³",size=6.5)
    d.text(2.1,2.25,"Align + PCA: 1024 → 256\nCell gate + 2-D position",size=6.8)
    d.text(2.1,1.79,"M: (T, P, 256)",size=7,weight="bold")
    panel(d,4.6,1.58,6,6.52,"(b) Entity decoder × 4  |  8 heads, width 256")
    d.box(5.05,6.35,5.1,.95,"Temporal attention — slots only","(N, T, 256) → (N, T, 256)","teal")
    d.box(5.05,4.95,5.1,.95,"Spatial attention — all entities","(T, N+Q, 256) → (T, N+Q, 256)","teal")
    d.box(5.05,3.55,5.1,.95,"Image cross-attention","Queries → M; output (T, N+Q, 256)","teal")
    d.box(5.05,2.15,5.1,.95,"Feed-forward + residual","256 → 1024 → 256","teal")
    for y1,y2 in [(6.35,5.9),(4.95,4.5),(3.55,3.1)]:
        d.arrow([(7.6,y1),(7.6,y2)])
    d.arrow([(3.8,7.42),(4.19,7.42),(4.19,6.82),(5.05,6.82)])
    d.arrow([(3.8,5.475),(4.2,5.475),(4.2,5.425),(5.05,5.425)])
    d.arrow([(3.8,2.2),(4.35,2.2),(4.35,4.025),(5.05,4.025)])
    d.section(11.35,8.62,"(c) Task readouts")
    d.box(11.35,6.75,4.25,1.35,"Persistent world slots","Classes C; reconstruction 256\nBox residual 4; corner residual 8 × 3")
    d.box(11.35,4.95,4.25,1.35,"Joint object detection","Classes C+1; 2-D box 4\n3-D OBB corners 8 × 3","orange")
    d.box(11.35,2.25,4.25,1.95,"Attention-based relations","Weights 2Lh = 64 + product 256\n320 → 64; pair fusion 896 → 256\nTemporal edges → 3 / 6 / 17")
    d.arrow([(10.15,2.63),(10.95,2.63),(10.95,7.42),(11.35,7.42)])
    d.arrow([(10.95,5.62),(11.35,5.62)])
    d.arrow([(10.95,2.9),(11.35,2.9)])
    d.arrow([(10.15,5.425),(10.72,5.425),(10.72,3.6),(11.35,3.6)],dashed=True,color=COLORS["purple"][1])
    tag(d,13.475,4.56,"Spatial weights from every layer","purple")
    d.text(7.6,1.82,"All stages preserve the 256-d entity width.",size=6.6)
    d.text(13.475,1.7,"Union ROI features are replaced.",size=6.6)
    d.text(.4,1.16,"P = HₚWₚ image tokens; L = 4; h = 8. Free-query detections are evaluated separately from the world graph.",ha="left",size=6.5)
    footer(d,"Training: inherited WorldWise losses + matched-query detection + visible-slot box/corner refinement.")
    return d


if __name__ == "__main__":
    p=parser(__doc__)
    p.add_argument("--variant",choices=("all","worldwise","worldwise_plus","worldwise_pp"),default="all")
    args=p.parse_args()
    builders={"worldwise":worldwise_paper,"worldwise_plus":worldwise_plus_paper,"worldwise_pp":worldwise_pp_paper}
    for name,build in builders.items():
        if args.variant in ("all",name):
            build().save(name+"_paper",args)


