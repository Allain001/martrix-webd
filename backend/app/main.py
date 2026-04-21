from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from app.core.matrix_ops import (
    compute_determinant_lu,
    compute_eigenvalue_qr,
    compute_inverse_gauss_jordan,
    compute_rank,
    solve_linear_system,
)


APP_VERSION = "2.1.0"
BACKEND_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = BACKEND_ROOT.parent / "frontend" / "dist"


def _array(data: list[list[float]] | list[float]) -> np.ndarray:
    return np.array(data, dtype=float)


def _display_scalar(value: Any, precision: int = 4) -> str:
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, complex):
        if abs(value.imag) < 1e-10:
            return f"{value.real:.{precision}f}"
        sign = "+" if value.imag >= 0 else "-"
        return f"{value.real:.{precision}f} {sign} {abs(value.imag):.{precision}f}i"

    if isinstance(value, float):
        if abs(value) < 1e-10:
            value = 0.0
        return f"{value:.{precision}f}"

    return str(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(np.real_if_close(value, tol=1000).tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, complex):
        if abs(value.imag) < 1e-10:
            return float(value.real)
        return {
            "real": float(value.real),
            "imag": float(value.imag),
            "display": _display_scalar(value),
        }
    return value


def _solution_label(solution_type: str) -> str:
    labels = {
        "unique_solution": "Unique solution",
        "no_solution": "No solution",
        "infinite_solutions": "Infinitely many solutions",
    }
    return labels.get(solution_type, "Unknown")


def _matrix_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    rows, cols = matrix.shape
    diagnostics = {
        "shape": f"{rows} x {cols}",
        "rows": rows,
        "cols": cols,
        "zero_density": round(float(np.mean(np.abs(matrix) < 1e-12) * 100), 1),
        "rank_estimate": int(np.linalg.matrix_rank(matrix)),
        "is_square": rows == cols,
    }
    if rows == cols:
        try:
            diagnostics["det_hint"] = round(float(np.linalg.det(matrix)), 4)
        except np.linalg.LinAlgError:
            diagnostics["det_hint"] = None
        try:
            diagnostics["condition_number"] = float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError:
            diagnostics["condition_number"] = None
    return diagnostics


def _transform_preview(matrix: np.ndarray) -> dict[str, Any]:
    if matrix.shape != (2, 2):
        return {
            "available": False,
            "message": "The interactive geometry stage currently supports 2x2 matrices.",
        }

    basis = np.eye(2)
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    sample_points = np.array(
        [[1.0, 1.0], [2.0, 1.0], [-1.0, 2.0], [-2.0, -1.0]],
        dtype=float,
    )
    transformed_points = (matrix @ sample_points.T).T

    return {
        "available": True,
        "basis": _json_safe((matrix @ basis.T).T),
        "square": _json_safe((matrix @ square.T).T),
        "sample_points": _json_safe(sample_points),
        "transformed_points": _json_safe(transformed_points),
        "determinant_area_scale": float(np.linalg.det(matrix)),
    }


def _build_response(
    operation: str,
    matrix: np.ndarray,
    b_vector: np.ndarray | None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "operation": operation,
        "matrix": _json_safe(matrix),
        "diagnostics": _matrix_diagnostics(matrix),
        "preview": _transform_preview(matrix),
        "results": {},
    }

    if operation in ("determinant", "all"):
        determinant = compute_determinant_lu(matrix)
        response["results"]["determinant"] = {
            "value": float(determinant["value"]),
            "display": _display_scalar(determinant["value"]),
            "steps": _json_safe(determinant["steps"][:8]),
        }

    if operation in ("inverse", "all"):
        try:
            inverse = compute_inverse_gauss_jordan(matrix)
            response["results"]["inverse"] = {
                "matrix": _json_safe(inverse["matrix"]),
                "steps": _json_safe(inverse["steps"][:10]),
            }
        except Exception as exc:  # noqa: BLE001
            response["results"]["inverse"] = {"error": str(exc)}

    if operation in ("eigen", "all"):
        try:
            eigen = compute_eigenvalue_qr(matrix)
            response["results"]["eigen"] = {
                "values": [_display_scalar(value) for value in eigen["values"]],
                "iterations": int(eigen["iterations"]),
                "method": eigen["method"],
                "converged": bool(eigen["converged"]),
                "note": eigen["note"],
                "convergence": _json_safe(eigen["convergence"][:40]),
            }
        except Exception as exc:  # noqa: BLE001
            response["results"]["eigen"] = {"error": str(exc)}

    if operation in ("rank", "all"):
        rank = compute_rank(matrix)
        response["results"]["rank"] = {
            "rank": int(rank["rank"]),
            "nullity": int(rank["nullity"]),
            "rref": _json_safe(rank["rref"]),
        }

    if operation in ("solve", "all"):
        if b_vector is not None:
            solution = solve_linear_system(matrix, b_vector)
            response["results"]["solution"] = {
                "type": solution["type"],
                "type_label": _solution_label(solution["type"]),
                "x": _json_safe(solution["x"]),
                "particular": _json_safe(solution["particular"]),
                "null_space_basis": _json_safe(solution["null_space_basis"]),
                "rref": _json_safe(solution["rref"]),
                "pivot_cols": _json_safe(solution["pivot_cols"]),
                "free_cols": _json_safe(solution["free_cols"]),
                "steps": _json_safe(solution["steps"][:12]),
            }
        else:
            response["results"]["solution"] = {
                "warning": "No vector b was provided, so Ax=b was skipped.",
            }

    return response


SITE_SECTIONS = [
    {
        "id": "lab",
        "label": "Matrix Lab",
        "description": "Interactive linear algebra workspace inspired by graph-first math tools.",
    },
    {
        "id": "cases",
        "label": "Case Gallery",
        "description": "Load curated transformations and explain them with one click.",
    },
    {
        "id": "explain",
        "label": "Explainable Results",
        "description": "Turn determinants, rank, inverse, eigenvalues, and Ax=b into visible stories.",
    },
]

QUICKSTART_STEPS = [
    {
        "title": "先加载一个案例",
        "body": "第一次访问时可以直接点击案例卡片，不需要先理解所有参数。",
    },
    {
        "title": "再拖动画布中的探针点",
        "body": "图形视图会把矩阵变换可视化，让线性代数从公式变成动态过程。",
    },
    {
        "title": "最后读取右侧解释",
        "body": "结果区会把 determinant、rank、eigen、Ax=b 的结论和步骤同步讲清楚。",
    },
]

FAQ_ITEMS = [
    {
        "question": "它还是 Streamlit 本地演示吗？",
        "answer": "不是。当前版本是 React + FastAPI 的同源单站点结构，目标是公网可访问的网站。",
    },
    {
        "question": "别人打开链接能看到同样的矩阵状态吗？",
        "answer": "可以。矩阵、向量、运算模式和当前案例都能写进 URL 参数中恢复。",
    },
    {
        "question": "为什么要做成 GeoGebra 风格？",
        "answer": "因为这种左侧代数、中央图形、右侧解释的工作台布局更适合数学交互，也更容易让非专业观众看懂。",
    },
]

DEPLOYMENT_ITEMS = [
    {
        "title": "Same-Origin",
        "body": "前端静态资源和 API 由同一个 FastAPI 站点提供，部署更简单。",
    },
    {
        "title": "Shareable State",
        "body": "矩阵、向量、运算模式和案例都能通过 URL 恢复，便于分享。",
    },
    {
        "title": "Container Ready",
        "body": "项目已经提供 Dockerfile，可继续接到 Render、Railway 或自有云服务器。",
    },
]


DEMO_CASES = [
    {
        "id": "shear-intuition",
        "title": "Shear intuition",
        "subtitle": "Best for showing how the unit square becomes a slanted tile.",
        "story": "This case is ideal for explaining why a matrix can preserve area trend while visibly changing shape.",
        "teaching_focus": "剪切会改变形状，但不一定立刻让观众误以为是旋转或缩放。",
        "action_hint": "先拖动探针点，再观察单位方格如何整体倾斜。",
        "matrix": [[1.0, 0.8], [0.0, 1.0]],
        "b": [1.0, 2.0],
        "operation": "all",
    },
    {
        "id": "rotation-scale",
        "title": "Rotation and scale",
        "subtitle": "Useful for eigen-directions, rotation, and area scaling at the same time.",
        "story": "This case combines rotation and stretching, so it is useful when you want a more dynamic and product-like first impression.",
        "teaching_focus": "适合把图形变化、面积缩放和特征值讨论串在一起。",
        "action_hint": "切换全链路分析，然后观察 determinant 和 eigen 的结果区联动。",
        "matrix": [[1.2, -0.6], [0.6, 1.2]],
        "b": [2.0, 1.0],
        "operation": "all",
    },
    {
        "id": "reflection",
        "title": "Reflection",
        "subtitle": "A quick way to explain negative determinants and orientation flip.",
        "story": "This is the fastest case for explaining why the sign of the determinant matters visually.",
        "teaching_focus": "重点讲清楚 orientation flip，也就是方向翻转。",
        "action_hint": "让观众先看原图，再看右图中图形如何镜像翻折。",
        "matrix": [[1.0, 0.0], [0.0, -1.0]],
        "b": [1.0, -1.0],
        "operation": "determinant",
    },
    {
        "id": "linear-system",
        "title": "Solve Ax = b",
        "subtitle": "Connect geometric intuition with a concrete unique-solution system.",
        "story": "This case anchors the experience back to solving equations, which is helpful for teaching and defense narration.",
        "teaching_focus": "把图形直觉和 Ax=b 的唯一解讲法连接起来。",
        "action_hint": "先看结果区中的 solution，再带着观众读步骤卡片。",
        "matrix": [[2.0, 1.0], [1.0, 3.0]],
        "b": [4.0, 7.0],
        "operation": "solve",
    },
]


class AnalyzeRequest(BaseModel):
    matrix: list[list[float]] = Field(..., min_length=1)
    operation: Literal[
        "determinant",
        "inverse",
        "eigen",
        "rank",
        "solve",
        "all",
    ] = "all"
    b: list[float] | None = None

    @model_validator(mode="after")
    def validate_dimensions(self) -> "AnalyzeRequest":
        row_lengths = {len(row) for row in self.matrix}
        if len(row_lengths) != 1:
            raise ValueError("Each matrix row must have the same length.")
        if self.b is not None and len(self.b) != len(self.matrix):
            raise ValueError("Vector b must have the same length as the number of matrix rows.")
        return self


app = FastAPI(
    title="MatrixVis Website Backend",
    version=APP_VERSION,
    description="MatrixVis public web app backend and matrix analysis APIs.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": APP_VERSION,
        "frontend_dist_ready": FRONTEND_DIST.exists(),
    }


@app.get("/api/site-content")
def site_content() -> dict[str, Any]:
    return {
        "title": "MatrixVis",
        "subtitle": "A public-facing linear algebra web app with GeoGebra-style interaction.",
        "sections": SITE_SECTIONS,
        "quickstart": QUICKSTART_STEPS,
        "faq": FAQ_ITEMS,
        "deployment": DEPLOYMENT_ITEMS,
    }


@app.get("/api/demo-cases")
def demo_cases() -> dict[str, Any]:
    return {"items": DEMO_CASES}


@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict[str, Any]:
    try:
        matrix = _array(payload.matrix)
        b_vector = _array(payload.b) if payload.b is not None else None
        return _build_response(payload.operation, matrix, b_vector)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def serve_index() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str) -> FileResponse:
        candidate = FRONTEND_DIST / full_path
        if candidate.exists() and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
