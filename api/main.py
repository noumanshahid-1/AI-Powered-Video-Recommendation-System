# api/main.py
# Run:  uvicorn api.main:app --reload --port 8000

import os, json, threading, webbrowser, pickle, csv
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
import torch

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, PlainTextResponse

API_VERSION = "LIGHTGCN-IMDB-V1"

# ---- Artifacts & session storage ----
ART_DIR = Path(os.getenv("RECSYS_ARTIFACTS", "./artifacts"))
SESS_DIR = ART_DIR / "sessions"
SESS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Global model/data holders ----
model_cf: Dict[str, Any] | None = None
X_content: sparse.csr_matrix | None = None
popularity: np.ndarray | None = None
pop_rank: np.ndarray | None = None  # for novelty

item2idx: Dict[int, int] = {}
idx2item: Dict[int, int] = {}
item_title: Dict[int, str] = {}
item_genre: Dict[int, str] = {}

HAS_POSTER: np.ndarray | None = None


# LightGCN embeddings
USER_F: np.ndarray | None = None
ITEM_F: np.ndarray | None = None


ALPHA_BEST = 0.7

# In-memory per-user feedback (likes/skips)
user_feedback: Dict[int, Dict[str, set]] = defaultdict(
    lambda: {"like": set(), "skip": set()}
)


def _get_sets(u: int):
    d = user_feedback[u]
    return d["like"], d["skip"]


# In-memory explanation cache (per (user, item, alpha))
EXPLAIN_CACHE: Dict[Tuple[int, int, float], Dict[str, Any]] = {}

# ---- FastAPI app ----
app = FastAPI(title="Offline Recsys API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Utils ----------------


def _norm01(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.size == 0:
        return v
    mn, mx = float(np.nanmin(v)), float(np.nanmax(v))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(v, dtype=np.float32)
    out = (v - mn) / (mx - mn)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def _to_1d(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


from pathlib import Path

def poster_path_for_movie(movie_id: int | None) -> str | None:
    """
    Match the Streamlit app behaviour:
    use posters/{movieId}.jpg if the file exists, else None.
    Frontend will turn this into /posters/<filename>.
    """
    if movie_id is None or int(movie_id) <= 0:
        return None
    filename = f"{int(movie_id)}.jpg"
    path = POSTER_DIR / filename
    return filename if path.exists() else None


# ---------------- CF / CB scoring ----------------


def cf_scores_for_user(u: int) -> np.ndarray:
    """
    CF scores from LightGCN embeddings.

    Assumes USER_F shape: [n_users, d], ITEM_F shape: [n_items, d]
    and that API `user_id` == internal LightGCN user index (0..n_users-1).
    """
    global USER_F, ITEM_F, model_cf
    if USER_F is None or ITEM_F is None or model_cf is None:
        return np.zeros(0, dtype=np.float32)

    n_items = int(model_cf.get("n_items_full", ITEM_F.shape[0]))
    if u < 0 or u >= USER_F.shape[0]:
        return np.zeros(n_items, dtype=np.float32)

    user_vec = USER_F[int(u)]
    scores = np.asarray(user_vec @ ITEM_F.T, dtype=np.float32)

    if scores.shape[0] != n_items:
        tmp = np.zeros(n_items, dtype=np.float32)
        n = min(n_items, scores.shape[0])
        tmp[:n] = scores[:n]
        scores = tmp

    scores[~np.isfinite(scores)] = 0.0
    return scores


def cb_scores_for_user(u: int) -> np.ndarray:
    like_set, skip_set = _get_sets(u)
    has_like = len(like_set) > 0
    has_skip = len(skip_set) > 0
    if not has_like and not has_skip:
        if X_content is None:
            return np.zeros(0, dtype=np.float32)
        return np.zeros(X_content.shape[0], dtype=np.float32)

    prof = None
    if has_like:
        rows_l = np.fromiter(like_set, dtype=np.int32)
        prof = X_content[rows_l].mean(axis=0)
    if has_skip:
        rows_s = np.fromiter(skip_set, dtype=np.int32)
        neg = X_content[rows_s].mean(axis=0)
        prof = (prof if prof is not None else 0) + (-0.5) * neg

    vec = prof.A1 if hasattr(prof, "A1") else np.asarray(prof).ravel()
    nrm = np.linalg.norm(vec)
    if nrm > 0:
        vec = vec / nrm
    sims = X_content.dot(vec)
    sims = sims.A1 if hasattr(sims, "A1") else np.asarray(sims).ravel()
    sims[~np.isfinite(sims)] = 0.0
    return sims.astype(np.float32)


def _pairwise_content_sims(item_indices: np.ndarray) -> np.ndarray:
    """Return cosine-ish similarity (dot of normalized vectors) among given items."""
    if X_content is None or not hasattr(X_content, "tocsr"):
        return np.eye(len(item_indices), dtype=np.float32)
    X = X_content[item_indices].tocsr().astype(np.float32)
    norms = np.sqrt((X.multiply(X)).sum(axis=1)).A1 + 1e-8
    X = X.multiply(1.0 / norms[:, None])
    M = (X @ X.T).toarray().astype(np.float32)
    np.fill_diagonal(M, 0.0)
    return M


def fuse_scores(
    u: int, alpha: float, k: int, cand_pool: int = 10000, novelty: bool = False
):
    """
    Returns:
      top_idx, top_scores, disp_cf, disp_cb, sim_matrix (for client-side MMR)
    """
    if model_cf is None:
        return np.array([], dtype=int), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(
            (0, 0), dtype=np.float32
        )

    n = int(model_cf["n_items_full"])
    like_set, skip_set = _get_sets(u)
    seen = like_set | skip_set
    seen_arr = (
        np.fromiter(seen, dtype=np.int32) if len(seen) else np.array([], dtype=np.int32)
    )

    cf = cf_scores_for_user(u).astype(np.float32, copy=False)
    cb = cb_scores_for_user(u).astype(np.float32, copy=False)

    if len(seen_arr):
        cf[seen_arr] = -1e9
        cb[seen_arr] = -1e9

    pop = popularity.copy() if popularity is not None else np.zeros_like(cf)
    if len(seen_arr):
        pop[seen_arr] = -1e9

    kth = min(max(1, cand_pool), n) - 1
    pool = np.unique(
        np.concatenate(
            [
                np.argpartition(-cf, kth=kth)[: (kth + 1)],
                np.argpartition(-cb, kth=kth)[: (kth + 1)],
                np.argpartition(-pop, kth=kth)[: (kth + 1)],
            ]
        )
    )

    cf_pool_n = _norm01(cf[pool])
    cb_pool_n = _norm01(cb[pool])
    fused = float(alpha) * cf_pool_n + (1.0 - float(alpha)) * cb_pool_n

    # Optional novelty: penalize popular items via pop_rank
    if novelty and pop_rank is not None:
        r = pop_rank[pool]  # 0 = most popular
        novelty_boost = 1.0 / np.log1p(2.0 + r.astype(np.float32))
        fused = fused * novelty_boost

    order = np.argsort(-fused)
    top_idx = pool[order][:k]
    top_scores = fused[order][:k].astype(np.float32)

    cf_top = cf[top_idx]
    cb_top = cb[top_idx]

    if np.ptp(cf_top) == 0.0 and np.ptp(cb_top) == 0.0:
        disp_cf = np.full(cf_top.shape, float(alpha), dtype=np.float32)
        disp_cb = np.full(cb_top.shape, float(1.0 - alpha), dtype=np.float32)
    else:
        disp_cf = (float(alpha) * _norm01(cf_top)).astype(np.float32)
        disp_cb = (1.0 - float(alpha)) * _norm01(cb_top)
        disp_cb = disp_cb.astype(np.float32)

    sim_matrix = _pairwise_content_sims(top_idx)

    for arr in (top_scores, disp_cf, disp_cb, sim_matrix):
        if isinstance(arr, np.ndarray):
            arr[~np.isfinite(arr)] = 0.0

    return top_idx, top_scores, disp_cf, disp_cb, sim_matrix


# ------------- Personalized Explain (alpha-aware + cached) -------------


def explain(
    u: int, rec_i: int, alpha: float = 0.7, top_from_history: int = 3
) -> Dict[str, Any]:
    """
    Personalized rationale for a (user, item, alpha):
      - why (string)
      - genres (string)
      - similar_from_likes / similar_from_skips (lists: {title, genres, sim, pop_percentile})
      - cf, cb raw scores for this user-item
      - pop_percentile (if available)
    Cached per (user_id, item_index, alpha) to avoid cross-user leakage.
    """
    key = (int(u), int(rec_i), float(alpha))
    if key in EXPLAIN_CACHE:
        return EXPLAIN_CACHE[key]

    like_set, skip_set = _get_sets(u)
    g = (item_genre.get(rec_i, "") or "").replace(
        "(no genres listed)", ""
    ).replace("|", " / ")

    # Popularity percentile (100 = most popular)
    if popularity is not None and pop_rank is not None:
        rank = int(pop_rank[rec_i])
        pop_pct = float(100.0 * (1.0 - (rank / max(1, popularity.shape[0] - 1))))
    else:
        pop_pct = None

    def _top_similar(src_set: set, k=top_from_history):
        if not src_set or X_content is None:
            return []
        rows = np.fromiter(src_set, dtype=np.int32)
        v = X_content[rec_i]
        sims = _to_1d(X_content[rows] @ v.T)
        if sims.size == 0:
            return []
        order = np.argsort(-sims)[:k]
        out = []
        for j in order:
            ii = int(rows[j])
            # popularity percentile for the source item
            if popularity is not None and pop_rank is not None:
                r2 = int(pop_rank[ii])
                pct2 = float(
                    100.0 * (1.0 - (r2 / max(1, popularity.shape[0] - 1)))
                )
            else:
                pct2 = None
            out.append(
                {
                    "item_index": ii,
                    "movieId": int(idx2item.get(ii, -1)),
                    "title": item_title.get(ii, f"Item {ii}"),
                    "genres": (item_genre.get(ii, "") or "")
                    .replace("(no genres listed)", "")
                    .replace("|", " / "),
                    "sim": float(sims[j]),
                    "pop_percentile": pct2,
                }
            )
        return out

    liked_list = _top_similar(like_set)
    skipped_list = _top_similar(skip_set)

    # Compose 'why' text
    max_like = max([x["sim"] for x in liked_list], default=0.0)
    max_skip = max([x["sim"] for x in skipped_list], default=0.0)

    if max_skip > max_like and max_skip > 0:
        # Stronger similarity to things you skipped – tell the truth
        why_text = "Similar to items you skipped (down-weighted)."
    elif max_like > 0:
        # Clear “because you liked X, Y” explanation
        titles = ", ".join([f"‘{x['title']}’" for x in liked_list[:2]]) or "your likes"
        why_text = f"Because you liked {titles}."
    else:
        # No strong content match either way
        if not like_set and not skip_set:
            # True cold-start
            if pop_pct is not None and pop_pct > 60:
                why_text = "Because it’s a popular, well-rated movie to start with."
            else:
                why_text = "Because it’s a strong starting point from the model."
        else:
            # You have some history, but this item is more about diversity / balance
            why_text = "Because it fits your overall taste and adds some variety."

    # CF/CB raw scores for this user-item (not normalized display)
    try:
        cf_v = float(cf_scores_for_user(u)[rec_i])
    except Exception:
        cf_v = 0.0
    try:
        cb_v = float(cb_scores_for_user(u)[rec_i])
    except Exception:
        cb_v = 0.0

    out = {
        "user_id": int(u),
        "item_index": int(rec_i),
        "movieId": int(idx2item.get(rec_i, -1)),
        "title": item_title.get(rec_i, f"Item {rec_i}"),
        "genres": g,
        "why": why_text,
        "similar_from_likes": liked_list,
        "similar_from_skips": skipped_list,
        "cf": cf_v,
        "cb": cb_v,
        "pop_percentile": pop_pct,
    }
    EXPLAIN_CACHE[key] = out
    return out


# ------------- Session reset utilities -------------


def _clear_if_exists(name, cleared):
    g = globals()
    if name in g and hasattr(g[name], "clear"):
        try:
            g[name].clear()
            cleared.append(name)
        except Exception:
            pass


@app.post("/session/clear_all")
def clear_all_sessions():
    """
    Wipes in-memory interaction/session caches so the app starts 'fresh'.
    Also clears explanation cache and per-user feedback.
    """
    cleared: List[str] = []

    # Clear some generic globals if they exist
    for var in [
        "LIKES_BY_USER",
        "SKIPS_BY_USER",
        "USER_EVENTS",
        "INTERACTION_LOG",
        "SESSIONS",
        "SESSION_STORE",
        "RECENT_ACTIONS",
    ]:
        _clear_if_exists(var, cleared)

    # Clear our actual stores
    try:
        user_feedback.clear()
        cleared.append("user_feedback")
    except Exception:
        pass
    try:
        EXPLAIN_CACHE.clear()
        cleared.append("EXPLAIN_CACHE")
    except Exception:
        pass

    # Remove some optional on-disk session files
    for path in [
        "data/session_store.pkl",
        "data/session_store.json",
        "session_store.pkl",
        "session_store.json",
    ]:
        try:
            if os.path.exists(path):
                os.remove(path)
                cleared.append(path)
        except Exception:
            pass

    # Also wipe saved per-user JSON sessions
    try:
        if SESS_DIR.exists():
            for p in SESS_DIR.glob("user_*.json"):
                p.unlink(missing_ok=True)
            cleared.append(str(SESS_DIR))
    except Exception:
        pass

    return {"ok": True, "cleared": cleared}


@app.post("/session/clear")
def clear_user_session(user_id: int):
    """
    Clears one user's likes/skips and any explain cache entries for that user.
    """
    like_set, skip_set = _get_sets(user_id)
    like_set.clear()
    skip_set.clear()

    # Clear explain cache entries for this user
    to_del = [k for k in EXPLAIN_CACHE.keys() if k[0] == int(user_id)]
    for k in to_del:
        EXPLAIN_CACHE.pop(k, None)

    return {
        "ok": True,
        "user_id": user_id,
        "cleared": {"likes": 0, "skips": 0, "explain_keys": len(to_del)},
    }


# ------------- Startup: load LightGCN artifacts -------------


def _extract_user_item_embeddings(obj, n_users: int, n_items: int):
    """
    Be robust to different save formats: dict with state_dict, or full module.
    """
    if isinstance(obj, dict):
        state = obj.get("state_dict") or obj.get("model_state_dict") or obj
    elif hasattr(obj, "state_dict"):
        state = obj.state_dict()
    else:
        return None, None

    tensor_items = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}

    # 1) Prefer shapes that match (n_users, d) and (n_items, d)
    user_candidates = []
    item_candidates = []
    for name, t in tensor_items.items():
        if t.dim() != 2:
            continue
        rows, cols = t.shape
        if rows == n_users:
            user_candidates.append((name, t))
        if rows == n_items:
            item_candidates.append((name, t))

    if user_candidates and item_candidates:
        u_name, u_t = user_candidates[0]
        i_name, i_t = item_candidates[0]
        print(f"Using embeddings from: user='{u_name}', item='{i_name}'")
        return u_t, i_t

    # 2) Fallback: name-based heuristics
    def pick_by_name(substrs_user, substrs_item):
        U = I = None
        for name, t in tensor_items.items():
            lname = name.lower()
            if U is None and all(s in lname for s in substrs_user):
                U = t
            if I is None and all(s in lname for s in substrs_item):
                I = t
        return U, I

    for pattern_u, pattern_i in [
        (["user", "emb"], ["movie", "emb"]),
        (["user", "embedding"], ["movie", "embedding"]),
        (["user"], ["movie"]),
    ]:
        U, I = pick_by_name(pattern_u, pattern_i)
        if U is not None and I is not None:
            print("Using embeddings by name pattern:", pattern_u, pattern_i)
            return U, I

    print("Available tensor keys in LightGCN state_dict:")
    for name, t in tensor_items.items():
        try:
            shape = tuple(t.shape)
        except Exception:
            shape = "<??>"
        print("   ", name, shape)
    return None, None


@app.on_event("startup")
def _load_artifacts():
    global model_cf, X_content, popularity, pop_rank
    global item2idx, idx2item, item_title, item_genre, ALPHA_BEST
    global USER_F, ITEM_F, poster_lookup_by_movie

    try:
        # ----- 1) ID mappings -----
        with open(ART_DIR / "movie_to_idx.pkl", "rb") as f:
            movie_to_idx = pickle.load(f)
        with open(ART_DIR / "idx_to_movie.pkl", "rb") as f:
            idx_to_movie = pickle.load(f)
        with open(ART_DIR / "user_to_idx.pkl", "rb") as f:
            user_to_idx = pickle.load(f)

        item2idx = {int(m): int(i) for m, i in movie_to_idx.items()}
        idx2item = {int(i): int(m) for i, m in idx_to_movie.items()}
        user_to_idx = {int(u): int(i) for u, i in user_to_idx.items()}

        n_items = max(item2idx.values()) + 1 if item2idx else 0
        n_items = max(n_items, len(idx2item))
        n_users = max(user_to_idx.values()) + 1 if user_to_idx else 0

        # ----- 2) Load LightGCN model & embeddings -----
        lgcn_obj = torch.load(ART_DIR / "best_lightgcn_model.pth", map_location="cpu")
        U_torch, I_torch = _extract_user_item_embeddings(lgcn_obj, n_users, n_items)
        if U_torch is None or I_torch is None:
            raise RuntimeError(
                "Could not find user/item embeddings inside best_lightgcn_model.pth"
            )

        USER_F = U_torch.detach().cpu().numpy().astype(np.float32)
        ITEM_F = I_torch.detach().cpu().numpy().astype(np.float32)
        n_users, n_items = USER_F.shape[0], ITEM_F.shape[0]

        # Minimal model_cf dict still used by other code
        model_cf = {
            "n_users_full": int(n_users),
            "n_items_full": int(n_items),
        }

        # ----- 3) Content embeddings -----
        cont = np.load(ART_DIR / "movie_content_embeddings.npy")
        if cont.shape[0] != n_items:
            raise RuntimeError(
                f"movie_content_embeddings rows ({cont.shape[0]}) "
                f"!= item factors ({n_items})"
            )
        X_content = sparse.csr_matrix(cont.astype(np.float32))

        # ----- 4) Titles & genres from movies.csv -----
        movie_meta: Dict[int, Tuple[str, str]] = {}
        with open(ART_DIR / "movies.csv", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = int(row["movieId"])
                title = row.get("title", "") or ""
                genres = row.get("genres", "") or ""
                genres = (
                    genres.replace("(no genres listed)", "").replace("|", " / ")
                )
                movie_meta[mid] = (title, genres)

        item_title.clear()
        item_genre.clear()
        for idx, mid in idx2item.items():
            t, g = movie_meta.get(mid, ("", ""))
            item_title[int(idx)] = t
            item_genre[int(idx)] = g

        # ----- 5) Popularity from ratings.csv -----
        ratings_path = ART_DIR / "ratings.csv"
        popularity = np.zeros(n_items, dtype=np.float32)
        if ratings_path.exists():
            ratings = pd.read_csv(ratings_path)
            counts = ratings.groupby("movieId")["rating"].size()
            for mid, cnt in counts.items():
                idx = item2idx.get(int(mid))
                if idx is not None and 0 <= idx < n_items:
                    popularity[int(idx)] = float(cnt)

        order = np.argsort(-popularity)
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)
        pop_rank = inv.astype(np.int32)

        # ----- 6) Precompute which item indices actually have a poster file -----
        global HAS_POSTER
        HAS_POSTER = np.zeros(n_items, dtype=bool)
        if POSTER_DIR.exists():
            for idx, mid in idx2item.items():
                p = POSTER_DIR / f"{mid}.jpg"
                if p.exists():
                    HAS_POSTER[int(idx)] = True
        print(f"Found posters for {HAS_POSTER.sum()} out of {n_items} items.")


        # ----- 7) Optional config (alpha_best, etc.) -----
        cfg_path = ART_DIR / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
                ALPHA_BEST = float(cfg.get("alpha_best", ALPHA_BEST))

        print("LightGCN artifacts loaded from:", ART_DIR)
        print(f"Users: {n_users} | Items: {n_items}")

    except Exception as e:
        raise RuntimeError(f"Failed loading LightGCN artifacts from {ART_DIR}: {e}")


# ------------- Service endpoints -------------


@app.get("/health")
def health():
    import inspect, time

    return {
        "status": "ok",
        "version": API_VERSION,
        "main_file": inspect.getfile(health),
        "artifacts": str(ART_DIR),
        "time": time.ctime(),
        "counts": {
            "items": int(model_cf["n_items_full"]) if model_cf else 0,
            "titles": len(item_title),
        },
    }


@app.get("/recommendations")
def recommendations(
    user_id: int = Query(...),
    k: int = 10,
    alpha: float | None = None,
    diversified: bool = True,
    novelty: bool = False,
    with_sim: int = 1,
    debug: int = 0,
):
    if model_cf is None:
        raise HTTPException(500, "Model not loaded")

    if alpha is None:
        alpha = ALPHA_BEST
    k = min(10, max(1, int(k)))

    items, scores, comp_cf, comp_cb, simM = fuse_scores(
        user_id, alpha, max(k * 3, k), novelty=bool(novelty)
    )

    # Client-side MMR (we still send sim_matrix; UI may re-rank)
    if diversified and len(items) > k:
        cand = np.array(items, dtype=int)
        sc = np.asarray(scores, dtype=np.float32)
        Xc = simM

        sel: List[int] = []
        while len(sel) < k and len(sel) < len(cand):
            if not sel:
                sel.append(int(np.argmax(sc)))
                sc[sel[-1]] = -1e9
            else:
                sims = np.max(Xc[sel], axis=0)
                mmr = 0.7 * sc - 0.3 * sims
                mmr[sel] = -1e9
                sel.append(int(np.argmax(mmr)))
                sc[sel[-1]] = -1e9

        sel = np.array(sel, dtype=int)
        items = cand[sel].tolist()
        scores = [float(scores[i]) for i in sel]
        cf_sel = [float(comp_cf[i]) for i in sel]
        cb_sel = [float(comp_cb[i]) for i in sel]
        simM = simM[sel][:, sel]
    else:
        items = items[:k].tolist() if hasattr(items, "tolist") else list(items[:k])
        scores = [float(s) for s in scores[:k]]
        cf_sel = [float(x) for x in comp_cf[:k]]
        cb_sel = [float(x) for x in comp_cb[:k]]
        simM = simM[:k, :k]

        # Compute nicer display scores for the top-k items
    scores_np = np.asarray(scores, dtype=np.float32)
    if scores_np.size:
        mn, mx = float(np.min(scores_np)), float(np.max(scores_np))
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn + 1e-8:
            # If all fused scores are effectively the same, just assign a simple rank-based score
            disp_scores = np.linspace(1.0, 0.2, scores_np.size, dtype=np.float32)
        else:
            # Normalize to [0,1]
            disp_scores = (scores_np - mn) / (mx - mn)
    else:
        disp_scores = scores_np

    rows = []
    for idx_pos, (i, s, cf_v, cb_v) in enumerate(zip(items, scores, cf_sel, cb_sel)):
        mid = int(idx2item.get(int(i), -1))

        # Clamp score to [0, +inf) for display
        score_val = float(disp_scores[idx_pos])
        if not np.isfinite(score_val) or score_val < 0:
            score_val = 0.0

        rows.append(
            {
                "item_index": int(i),
                "movieId": mid,
                "title": item_title.get(int(i), f"Item {i}"),
                "genres": (item_genre.get(int(i), "") or "")
                .replace("(no genres listed)", "")
                .replace("|", " / "),
                "score": score_val,
                "cf": float(cf_v),
                "cb": float(cb_v),
                "why": explain(user_id, int(i), alpha=float(alpha))["why"],
                "poster": poster_path_for_movie(mid),
            }
        )

    out: Dict[str, Any] = {
        "results": rows,
        "sim_matrix": (simM.tolist() if with_sim else None),
        "alpha": float(alpha),
        "user_id": int(user_id),
    }
    if debug:
        out["debug"] = {
            "k": k,
            "novelty": bool(novelty),
        }
    return out


@app.get("/item/{i}")
def item_meta(i: int):
    mid = int(idx2item.get(int(i), -1))
    return {
        "movieId": mid,
        "title": item_title.get(int(i), f"Item {i}"),
        "genres": (item_genre.get(int(i), "") or "")
        .replace("(no genres listed)", "")
        .replace("|", " / "),
        "poster": poster_path_for_movie(mid),
    }


@app.get("/search")
def search(q: str, limit: int = 20):
    ql = q.lower()
    hits = []
    for i, t in item_title.items():
        if ql in (t or "").lower():
            mid = int(idx2item.get(int(i), -1))
            hits.append(
                {
                    "item_index": int(i),
                    "movieId": mid,
                    "title": t,
                    "genres": (item_genre.get(int(i), "") or "")
                    .replace("(no genres listed)", "")
                    .replace("|", " / "),
                    "poster": poster_path_for_movie(mid),
                }
            )
            if len(hits) >= limit:
                break
    return {"query": q, "results": hits}


@app.get("/similar")
def similar(
    q: str | None = None,
    item_index: int | None = None,
    movieId: int | None = None,
    user_id: int | None = None,
    alpha: float | None = None,
    k: int = 10,
):
    if model_cf is None:
        raise HTTPException(500, "Model not loaded")

    if alpha is None:
        alpha = ALPHA_BEST

    # Resolve base item index "i"
    i = None
    if item_index is not None:
        i = int(item_index)
    elif movieId is not None:
        i = item2idx.get(int(movieId))
    elif q:
        ql = q.lower().strip()
        for idx, title in item_title.items():
            if ql in (title or "").lower():
                i = int(idx)
                break

    if i is None or i < 0 or i >= model_cf["n_items_full"]:
        raise HTTPException(400, "No base item found for 'similar'")

    # Content-based similarity from X_content
    if X_content is None:
        raise HTTPException(500, "Content matrix not loaded")

    v = X_content[i]
    sims = _to_1d(X_content @ v.T)
    sims[i] = -1e9  # exclude self

    order = np.argsort(-sims)

    # Prefer items that actually have posters (if we know that info)
    if HAS_POSTER is not None and HAS_POSTER.size > 0:
        with_poster = [idx for idx in order if HAS_POSTER[idx]]
        without_poster = [idx for idx in order if not HAS_POSTER[idx]]
        order = np.array(with_poster + without_poster, dtype=int)
    else:
        order = np.asarray(order, dtype=int)

    order = order[: int(k)]


    # ---- CF/CB contributions for the "similar" rows (normalized 0..1) ----
    if user_id is not None:
        cf_all = cf_scores_for_user(user_id)
        cb_all = cb_scores_for_user(user_id)
        cf_top = cf_all[order] if cf_all.size else np.zeros_like(order, dtype=np.float32)
        cb_top = cb_all[order] if cb_all.size else np.zeros_like(order, dtype=np.float32)
        cf_disp = _norm01(cf_top)
        cb_disp = _norm01(cb_top)
    else:
        cf_disp = np.zeros_like(order, dtype=np.float32)
        cb_disp = np.zeros_like(order, dtype=np.float32)

    rows = []
    for rank, j in enumerate(order):
        mid = int(idx2item.get(int(j), -1))

        # Genres with fallback
        g_raw = (item_genre.get(int(j), "") or "").replace("(no genres listed)", "").replace("|", " / ")
        g = g_raw or "N/A"
        sim_val = float(sims[j])
        if not np.isfinite(sim_val) or sim_val < 0:
            sim_val = 0.0
        rows.append(
            {
                "item_index": int(j),
                "movieId": mid,
                "title": item_title.get(int(j), f"Item {j}"),
                "genres": g,
                "score": sim_val,
                "cf": float(cf_disp[rank]),
                "cb": float(cb_disp[rank]),
                "why": f"Similar to “{item_title.get(i, f'Item {i}')}”",
                "poster": poster_path_for_movie(mid),
            }
        )

    return {
        "base": {"item_index": int(i), "title": item_title.get(i, f"Item {i}")},
        "results": rows,
    }



@app.post("/interact")
def interact(payload: Dict[str, Any]):
    if model_cf is None:
        raise HTTPException(500, "Model not loaded")

    u = int(payload.get("user_id", -1))
    i = int(payload.get("item_index", -1))
    action = str(payload.get("action", "")).lower().strip()
    if u < 0 or i < 0 or action not in {"like", "skip"}:
        raise HTTPException(400, "Missing/invalid user_id, item_index, or action")

    like_set, skip_set = _get_sets(u)
    n_items = model_cf["n_items_full"]
    if i < 0 or i >= n_items:
        raise HTTPException(400, f"item_index {i} out of range")

    if action == "like":
        like_set.add(i)
        if i in skip_set:
            skip_set.discard(i)
    else:
        skip_set.add(i)
        if i in like_set:
            like_set.discard(i)

    # Clear explain cache entries for this user so “Why?” reflects new feedback
    to_del = [k for k in EXPLAIN_CACHE.keys() if k[0] == int(u)]
    for k in to_del:
        EXPLAIN_CACHE.pop(k, None)

    return {
        "ok": True,
        "user_id": u,
        "item_index": i,
        "action": action,
        "likes": len(like_set),
        "skips": len(skip_set),
    }


@app.post("/ui/interact")
def interact_alias(payload: Dict[str, Any]):
    return interact(payload)


# ---- Session save/load ----


@app.get("/session/save")
def session_save(user_id: int):
    like_set, skip_set = _get_sets(user_id)
    data = {
        "user_id": user_id,
        "likes": sorted(int(x) for x in like_set),
        "skips": sorted(int(x) for x in skip_set),
    }
    p = SESS_DIR / f"user_{user_id}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {
        "ok": True,
        "saved_to": str(p),
        "counts": {"likes": len(like_set), "skips": len(skip_set)},
    }


@app.get("/session/load")
def session_load(user_id: int):
    p = SESS_DIR / f"user_{user_id}.json"
    if not p.exists():
        raise HTTPException(404, f"No saved session for user {user_id}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    like_set, skip_set = _get_sets(user_id)
    like_set.clear()
    skip_set.clear()
    for i in data.get("likes", []):
        like_set.add(int(i))
    for i in data.get("skips", []):
        skip_set.add(int(i))

    # Clear this user's explain cache after load
    to_del = [k for k in EXPLAIN_CACHE.keys() if k[0] == int(user_id)]
    for k in to_del:
        EXPLAIN_CACHE.pop(k, None)

    return {
        "ok": True,
        "user_id": user_id,
        "likes": len(like_set),
        "skips": len(skip_set),
    }


@app.get("/explain")
def explain_endpoint(user_id: int, item_index: int, alpha: float = 0.7):
    if model_cf is None:
        raise HTTPException(500, "Model not loaded")
    if item_index < 0 or item_index >= model_cf["n_items_full"]:
        raise HTTPException(400, "item_index out of range")
    return explain(user_id, item_index, alpha=alpha)


# ---- Static UI hosting ----

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# Posters live next to /web at the project root
POSTER_DIR = WEB_DIR.parent / "posters"

if POSTER_DIR.exists():
    app.mount("/posters", StaticFiles(directory=POSTER_DIR), name="posters")


@app.get("/_debug_static")
def _debug_static():
    p = str(WEB_DIR)
    exists = WEB_DIR.exists()
    idx = WEB_DIR / "index.html"
    return {
        "WEB_DIR": p,
        "exists": bool(exists),
        "index.html exists": bool(idx.exists()),
        "index.html size": (idx.stat().st_size if idx.exists() else 0),
    }


if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")

    @app.get("/")
    def root():
        idx = WEB_DIR / "index.html"
        if idx.exists():
            return FileResponse(idx)
        return RedirectResponse(url="/ui/")
else:

    @app.get("/")
    def root_missing():
        msg = f"UI folder not found: {WEB_DIR}. Create 'web/index.html' next to 'api/'."
        return PlainTextResponse(msg, status_code=404)


# ---- Auto-open browser (optional) ----

AUTO_OPEN_URL = "http://127.0.0.1:8000/"


def _open_browser_later(url=AUTO_OPEN_URL, delay: float = 0.8):
    try:
        threading.Timer(delay, lambda: webbrowser.open(url, new=1)).start()
    except Exception as e:
        print("Auto-open failed:", e)


@app.on_event("startup")
def _auto_open_ui():
    if os.getenv("RECSYS_AUTO_OPEN", "1") == "1":
        _open_browser_later()
