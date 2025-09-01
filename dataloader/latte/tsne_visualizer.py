import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial import ConvexHull
import plotly.express as px
from tqdm import tqdm


class TSNEVisualizer:
    """
    End-to-end t-SNE visualizations for rule/text embeddings.

    Methods:
        fit_tsne_2d / fit_tsne_3d
        plot_scatter_2d
        plot_scatter_3d
        plot_perplexity_grid
        plot_cluster_hulls
        plot_outliers
        plot_distance_heatmap
        plot_kmeans_colors
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        meta: None,
        label_col: None,
        text_col: None,
        outdir: "figs",
        pca_dim: int = 50,
        random_state: int = 42,
    ):
        self.X_raw = np.asarray(embeddings)
        self.meta = meta if meta is not None else pd.DataFrame(index=np.arange(self.X_raw.shape[0]))
        self.label_col = label_col if (label_col and label_col in self.meta) else None
        self.text_col = text_col if (text_col and text_col in self.meta) else None
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Optional PCA pre-reduction
        if pca_dim and pca_dim < self.X_raw.shape[1]:
            self.X = PCA(pca_dim, random_state=random_state).fit_transform(self.X_raw)
        else:
            self.X = self.X_raw

        # Placeholders
        self.Y2 = None
        self.Y3 = None
        self.df2d = None
        self.df3d = None

    # ---------- Core fitting ---------- #

    def fit_tsne_2d(self, perplexity=30, metric="euclidean", n_iter=1000, init="pca", **kwargs):
        self.Y2 = self._run_tsne(n_components=2, perplexity=perplexity, metric=metric,
                                 n_iter=n_iter, init=init, **kwargs)
        self.df2d = self.meta.copy()
        self.df2d["x"], self.df2d["y"] = self.Y2[:, 0], self.Y2[:, 1]
        return self.df2d

    def fit_tsne_3d(self, perplexity=30, metric="euclidean", n_iter=1000, init="pca", **kwargs):
        self.Y3 = self._run_tsne(n_components=3, perplexity=perplexity, metric=metric,
                                 n_iter=n_iter, init=init, **kwargs)
        self.df3d = self.meta.copy()
        self.df3d["x"], self.df3d["y"], self.df3d["z"] = self.Y3[:, 0], self.Y3[:, 1], self.Y3[:, 2]
        return self.df3d

    def _run_tsne(self, **params):
        tsne = TSNE(random_state=self.random_state, verbose=1, **params)
        return tsne.fit_transform(self.X)

    # ---------- Plotters ---------- #

    def plot_scatter_2d(self, filename="tsne_2d.png", title=None, annotate_max=100):
        self._require(self.df2d, "Call fit_tsne_2d first.")
        title = title or "t-SNE 2D"
        labels = self._labels(self.df2d)
        uniq = labels.unique()
        cmap = cm.get_cmap("tab20", len(uniq))

        plt.figure(figsize=(8, 6))
        for i, u in enumerate(uniq):
            m = labels == u
            plt.scatter(self.df2d.loc[m, "x"], self.df2d.loc[m, "y"],
                        s=8, alpha=0.7, label=u, color=cmap(i))
        if self.text_col:
            sub = self.df2d.sample(min(annotate_max, len(self.df2d)), random_state=0)
            for _, r in sub.iterrows():
                plt.text(r.x, r.y, str(r[self.text_col])[:30]+"…", fontsize=6, alpha=0.6)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.title(title)
        plt.tight_layout()
        plt.savefig(self.outdir / filename, dpi=300); plt.close()

    def plot_scatter_3d(self, filename="tsne_3d.html", title="Interactive 3D t-SNE"):
        self._require(self.df3d, "Call fit_tsne_3d first.")
        color = self.df3d[self.label_col] if self.label_col else None
        fig = px.scatter_3d(
            self.df3d, x="x", y="y", z="z", color=color,
            hover_data=self.df3d.columns, opacity=0.7, title=title, size_max=6
        )
        fig.write_html(self.outdir / filename)

    def plot_perplexity_grid(self, perplexities=(5, 15, 30, 50, 80), filename="tsne_perplexity_grid.png"):
        label_series = self._labels(self.meta)
        fig, axes = plt.subplots(1, len(perplexities), figsize=(4*len(perplexities), 4), squeeze=False)
        cmap = cm.get_cmap("tab20", len(label_series.unique()))
        for ax, p in zip(axes[0], perplexities):
            Y = self._run_tsne(n_components=2, perplexity=p)
            df = pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1], "lab": label_series.astype(str)})
            for i, u in enumerate(label_series.unique()):
                m = df["lab"] == u
                ax.scatter(df.loc[m, "x"], df.loc[m, "y"], s=6, alpha=0.6, color=cmap(i))
            ax.set_title(f"perp={p}")
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        fig.savefig(self.outdir / filename, dpi=300); plt.close(fig)

    def plot_cluster_hulls(self, filename="tsne_hulls.png"):
        self._require(self.df2d, "Call fit_tsne_2d first.")
        labels = self._labels(self.df2d)
        uniq = labels.unique()
        cmap = cm.get_cmap("tab10", len(uniq))
        plt.figure(figsize=(8, 6))
        for i, u in enumerate(uniq):
            pts = self.df2d[labels == u][["x", "y"]].values
            plt.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.5, color=cmap(i), label=u)
            if len(pts) > 3:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                plt.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.15, color=cmap(i))
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        plt.title("t-SNE with cluster hulls")
        plt.tight_layout()
        plt.savefig(self.outdir / filename, dpi=300); plt.close()

    def plot_outliers(self, k=5, pct=95, filename="tsne_outliers.png"):
        self._require(self.df2d, "Call fit_tsne_2d first.")
        X = self.df2d[["x", "y"]].values
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dists, _ = nbrs.kneighbors(X)
        score = dists[:, -1]
        thresh = np.percentile(score, pct)

        plt.figure(figsize=(8, 6))
        plt.scatter(self.df2d["x"], self.df2d["y"], s=6, alpha=0.5, label="inliers")
        outliers = score >= thresh
        plt.scatter(self.df2d.loc[outliers, "x"], self.df2d.loc[outliers, "y"],
                    s=12, edgecolor="red", facecolor="none", label="outliers")
        plt.legend(); plt.title(f"Outliers (>{pct}th kNN dist)")
        plt.tight_layout()
        plt.savefig(self.outdir / filename, dpi=300); plt.close()

    def plot_distance_heatmap(self, filename="tsne_distance_heatmap.png"):
        self._require(self.Y2, "Call fit_tsne_2d first.")
        D = pairwise_distances(self.Y2, metric="euclidean")
        plt.figure(figsize=(6, 5))
        plt.imshow(D, aspect="auto", cmap="viridis")
        plt.colorbar(label="t-SNE distance")
        plt.title("Pairwise distances in t-SNE space")
        plt.tight_layout()
        plt.savefig(self.outdir / filename, dpi=300); plt.close()

    def plot_kmeans_colors(self, n_clusters=10, filename="tsne_kmeans.png"):
        self._require(self.df2d, "Call fit_tsne_2d first.")
        km = KMeans(n_clusters=min(n_clusters, len(self.df2d)), random_state=self.random_state).fit(self.Y2)
        df = self.df2d.copy()
        df["kmeans"] = km.labels_.astype(str)
        self._scatter_static(df, color_col="kmeans",
                             filename=filename, title="t-SNE colored by KMeans clusters")

    # ---------- Internals ---------- #

    def _scatter_static(self, df, color_col, filename, title):
        uniq = df[color_col].unique()
        cmap = cm.get_cmap("tab20", len(uniq))
        plt.figure(figsize=(8, 6))
        for i, u in enumerate(uniq):
            m = df[color_col] == u
            plt.scatter(df.loc[m, "x"], df.loc[m, "y"], s=8, alpha=0.7, label=u, color=cmap(i))
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
        plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2"); plt.title(title)
        plt.tight_layout()
        plt.savefig(self.outdir / filename, dpi=300); plt.close()

    def _labels(self, df):
        return df[self.label_col].astype(str) if self.label_col else pd.Series(["all"] * len(df), index=df.index)

    @staticmethod
    def _require(obj, msg):
        if obj is None:
            raise RuntimeError(msg)


# ---------------- Example CLI wrapper ---------------- #

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    version = 2
    ap.add_argument("--meta")
    ap.add_argument("--label_col")
    ap.add_argument("--text_col")
    ap.add_argument("--outdir", default=f"figs/v{version}")
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--metric", default="euclidean", choices=["euclidean", "cosine"])
    ap.add_argument("--n_iter", type=int, default=1000)
    args = ap.parse_args()

    # Construct Embeddings
    rule_embeddings_root_dir = os.path.join(os.path.dirname(__file__), '../../lattedata/embeddings')
    rule_embedding_dict = {}
    rule_embeddings_array = []
    for video_name in tqdm(os.listdir(rule_embeddings_root_dir), desc="Loading rule embeddings"):
        video_rule_embeddings_path = os.path.join(rule_embeddings_root_dir, video_name)
        if video_name.endswith(".json"):
            with open(video_rule_embeddings_path, 'r') as f:
                video_rule_embeddings_dict = json.load(f)

        # Update the rule embedding dictionary
        for rule_name, rule_embedding in video_rule_embeddings_dict.items():
            if rule_name not in rule_embedding_dict:
                rule_embedding_dict[rule_name] = []
            # If its present, append the rule embedding
            rule_embedding_dict[rule_name].append(rule_embedding)
            rule_embeddings_array.append(rule_embedding)

    rule_embeddings_array = np.array(rule_embeddings_array)
    unique_rule_embeddings = np.unique(rule_embeddings_array, axis=0)
    meta = pd.read_csv(args.meta) if args.meta else None

    # Print Statistics for the Embeddings
    print(f"Total unique rule embeddings: {len(unique_rule_embeddings)}")
    print(f"Shape of rule embeddings array: {rule_embeddings_array.shape}")
    print(f"Shape of unique rule embeddings: {unique_rule_embeddings.shape}")


    viz = TSNEVisualizer(
        embeddings=rule_embeddings_array,
        meta=meta,
        label_col=args.label_col,
        text_col=args.text_col,
        outdir=args.outdir,
        pca_dim=args.pca_dim,
    )

    viz.fit_tsne_2d(perplexity=args.perplexity, metric=args.metric, n_iter=args.n_iter)
    viz.fit_tsne_3d(perplexity=args.perplexity, metric=args.metric, n_iter=args.n_iter)

    viz.plot_scatter_2d()
    viz.plot_scatter_3d()
    viz.plot_perplexity_grid()
    viz.plot_cluster_hulls()
    viz.plot_outliers()
    viz.plot_distance_heatmap()
    viz.plot_kmeans_colors()

    print(f"Done. Outputs in {Path(args.outdir).resolve()}")
