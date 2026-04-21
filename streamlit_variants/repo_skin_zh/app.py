from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

PAGES = [
    ("Home", "home", BASE_DIR / "Home.py"),
    ("2D Transform", "2d-transform", BASE_DIR / "pages" / "1_2D_Transform.py"),
    ("3D Transform", "3d-transform", BASE_DIR / "pages" / "2_3D_Transform.py"),
    ("2x3 Projection", "2x3-projection", BASE_DIR / "pages" / "3_2x3_Projection.py"),
    ("3x2 Lifting", "3x2-lifting", BASE_DIR / "pages" / "4_3x2_Lifting.py"),
    ("PCA Demo", "pca-demo", BASE_DIR / "pages" / "5_PCA_Demo.py"),
    ("SVDImgCompression", "svd-img-compression", BASE_DIR / "pages" / "6_SVDImgCompression.py"),
    ("PCAImgCompression", "pca-img-compression", BASE_DIR / "pages" / "7_PCAImgCompression.py"),
    ("LSE", "lse", BASE_DIR / "pages" / "8_LSE.py"),
]

PATH_TO_SLUG = {
    "Home.py": "home",
    "pages/1_2D_Transform.py": "2d-transform",
    "pages/2_3D_Transform.py": "3d-transform",
    "pages/3_2x3_Projection.py": "2x3-projection",
    "pages/4_3x2_Lifting.py": "3x2-lifting",
    "pages/5_PCA_Demo.py": "pca-demo",
    "pages/6_SVDImgCompression.py": "svd-img-compression",
    "pages/7_PCAImgCompression.py": "pca-img-compression",
    "pages/8_LSE.py": "lse",
}
SLUG_TO_PAGE = {slug: (label, path) for label, slug, path in PAGES}
DEFAULT_SLUG = "home"


def _normalize_page_target(target: str) -> str:
    cleaned = target.replace("\\", "/").strip()
    if cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return PATH_TO_SLUG.get(cleaned, DEFAULT_SLUG)


def _switch_page(target: str) -> None:
    st.session_state["active_page_slug"] = _normalize_page_target(target)
    st.query_params["page"] = st.session_state["active_page_slug"]
    st.rerun()


def _select_slug() -> str:
    query_slug = st.query_params.get("page", DEFAULT_SLUG)
    if isinstance(query_slug, list):
        query_slug = query_slug[0] if query_slug else DEFAULT_SLUG
    if query_slug not in SLUG_TO_PAGE:
        query_slug = DEFAULT_SLUG

    state_slug = st.session_state.get("active_page_slug", query_slug)
    if state_slug not in SLUG_TO_PAGE:
        state_slug = DEFAULT_SLUG
    return state_slug


st.set_page_config(page_title="martrixvis", layout="wide")
original_set_page_config = st.set_page_config
st.set_page_config = lambda *args, **kwargs: None
st.switch_page = _switch_page

current_slug = _select_slug()
labels = [label for label, _, _ in PAGES]
current_index = [slug for _, slug, _ in PAGES].index(current_slug)

with st.sidebar:
    selected_label = st.radio("Navigation", labels, index=current_index)

selected_slug = next(slug for label, slug, _ in PAGES if label == selected_label)
st.session_state["active_page_slug"] = selected_slug
st.query_params["page"] = selected_slug

_, selected_path = SLUG_TO_PAGE[selected_slug]
runpy.run_path(str(selected_path), run_name="__main__")

st.set_page_config = original_set_page_config
