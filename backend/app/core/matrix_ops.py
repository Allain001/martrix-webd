from __future__ import annotations

from typing import Any

import numpy as np


EPSILON = 1e-10


def _clean_small_values(matrix: np.ndarray, tol: float = EPSILON) -> np.ndarray:
    """Zero-out tiny floating-point noise for cleaner UI output."""
    cleaned = np.array(
        matrix,
        dtype=complex if np.iscomplexobj(matrix) else float,
        copy=True,
    )
    cleaned[np.abs(cleaned) < tol] = 0
    return np.real_if_close(cleaned, tol=1000)


def compute_determinant_lu(matrix: np.ndarray) -> dict[str, Any]:
    """Compute a determinant with LU-style elimination and pivot logging."""
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Determinant requires a square matrix.")

    a = matrix.astype(float).copy()
    l = np.eye(rows)
    p = np.eye(rows)
    steps: list[dict[str, Any]] = []
    intermediate_states = [a.copy()]
    det_sign = 1

    for pivot_col in range(rows - 1):
        max_row = np.argmax(np.abs(a[pivot_col:, pivot_col])) + pivot_col

        if max_row != pivot_col:
            a[[pivot_col, max_row]] = a[[max_row, pivot_col]]
            p[[pivot_col, max_row]] = p[[max_row, pivot_col]]
            if pivot_col > 0:
                l[[pivot_col, max_row], :pivot_col] = l[[max_row, pivot_col], :pivot_col]
            det_sign *= -1
            steps.append(
                {
                    "step": pivot_col + 1,
                    "type": "pivot",
                    "description": (
                        f"Swap row {pivot_col + 1} with row {max_row + 1} "
                        "to select a stable pivot."
                    ),
                    "matrix": _clean_small_values(a),
                    "formula": f"P({pivot_col + 1},{max_row + 1})A",
                }
            )
            intermediate_states.append(a.copy())

        if abs(a[pivot_col, pivot_col]) < EPSILON:
            return {
                "value": 0.0,
                "steps": steps,
                "intermediate_states": intermediate_states,
                "L": _clean_small_values(l),
                "U": _clean_small_values(a),
                "P": _clean_small_values(p),
            }

        for row in range(pivot_col + 1, rows):
            l[row, pivot_col] = a[row, pivot_col] / a[pivot_col, pivot_col]
            a[row, pivot_col:] -= l[row, pivot_col] * a[pivot_col, pivot_col:]

        steps.append(
            {
                "step": pivot_col + 1,
                "type": "elimination",
                "description": f"Eliminate entries below column {pivot_col + 1}.",
                "matrix": _clean_small_values(a),
                "formula": f"L({pivot_col + 1})",
            }
        )
        intermediate_states.append(a.copy())

    determinant = det_sign * np.prod(np.diag(a))
    return {
        "value": float(determinant),
        "steps": steps,
        "intermediate_states": intermediate_states,
        "L": _clean_small_values(l),
        "U": _clean_small_values(a),
        "P": _clean_small_values(p),
    }


def compute_inverse_gauss_jordan(matrix: np.ndarray) -> dict[str, Any]:
    """Compute the inverse using Gauss-Jordan elimination."""
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Inverse requires a square matrix.")

    augmented = np.hstack([matrix.astype(float).copy(), np.eye(rows)])
    steps: list[dict[str, Any]] = [
        {
            "step": 0,
            "description": "Start from the augmented matrix [A | I].",
            "matrix": _clean_small_values(augmented),
        }
    ]

    for pivot_row in range(rows):
        max_row = pivot_row + np.argmax(np.abs(augmented[pivot_row:, pivot_row]))
        pivot = augmented[max_row, pivot_row]
        if abs(pivot) < EPSILON:
            raise ValueError("This matrix is singular, so an inverse does not exist.")

        if max_row != pivot_row:
            augmented[[pivot_row, max_row]] = augmented[[max_row, pivot_row]]
            steps.append(
                {
                    "step": pivot_row + 1,
                    "description": (
                        f"Swap row {pivot_row + 1} with row {max_row + 1} "
                        "to bring the best pivot into place."
                    ),
                    "matrix": _clean_small_values(augmented),
                }
            )

        pivot = augmented[pivot_row, pivot_row]
        augmented[pivot_row] /= pivot
        steps.append(
            {
                "step": pivot_row + 1,
                "description": f"Normalize row {pivot_row + 1} so the pivot becomes 1.",
                "matrix": _clean_small_values(augmented),
            }
        )

        for row in range(rows):
            if row == pivot_row:
                continue
            factor = augmented[row, pivot_row]
            if abs(factor) < EPSILON:
                continue
            augmented[row] -= factor * augmented[pivot_row]
            steps.append(
                {
                    "step": pivot_row + 1,
                    "description": (
                        f"Use row {pivot_row + 1} to eliminate column {pivot_row + 1} "
                        f"in row {row + 1}."
                    ),
                    "matrix": _clean_small_values(augmented),
                }
            )

    return {
        "matrix": _clean_small_values(augmented[:, rows:]),
        "steps": steps,
        "augmented_final": _clean_small_values(augmented),
    }


def compute_eigenvalue_qr(
    matrix: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> dict[str, Any]:
    """Approximate eigenvalues with QR iteration and fall back when needed."""
    rows, cols = matrix.shape
    if rows != cols:
        raise ValueError("Eigenvalue analysis requires a square matrix.")

    a = matrix.astype(float).copy()
    eigenvectors = np.eye(rows)
    convergence: list[float] = []
    converged = False

    for _ in range(max_iter):
        q, r = np.linalg.qr(a)
        a = r @ q
        eigenvectors = eigenvectors @ q
        off_diagonal = np.linalg.norm(np.tril(a, k=-1))
        convergence.append(float(off_diagonal))
        if off_diagonal < tol:
            converged = True
            break

    if converged:
        eigenvalues = np.diag(a)
        vectors = eigenvectors
        method = "QR iteration"
        note = "QR iteration converged successfully."
    else:
        eigenvalues, vectors = np.linalg.eig(matrix.astype(float))
        method = "NumPy eig fallback"
        note = "QR iteration did not converge in time, so NumPy handled the final result."

    order = np.argsort(np.abs(eigenvalues))[::-1]
    eigenvalues = np.real_if_close(eigenvalues[order], tol=1000)
    vectors = np.real_if_close(vectors[:, order], tol=1000)

    return {
        "values": eigenvalues,
        "vectors": vectors,
        "iterations": len(convergence),
        "convergence": convergence,
        "final_matrix": _clean_small_values(a),
        "converged": converged,
        "method": method,
        "note": note,
    }


def _build_parametric_solution(
    rref: np.ndarray,
    pivot_cols: list[int],
    total_cols: int,
) -> tuple[np.ndarray, list[np.ndarray], list[int]]:
    """Build one particular solution plus a basis for the null space."""
    free_cols = [col for col in range(total_cols) if col not in pivot_cols]
    particular = np.zeros(total_cols)

    for row_idx, pivot_col in enumerate(pivot_cols):
        particular[pivot_col] = rref[row_idx, -1]

    null_space_basis = []
    for free_col in free_cols:
        basis = np.zeros(total_cols)
        basis[free_col] = 1.0
        for row_idx, pivot_col in enumerate(pivot_cols):
            basis[pivot_col] = -rref[row_idx, free_col]
        null_space_basis.append(_clean_small_values(basis))

    return _clean_small_values(particular), null_space_basis, free_cols


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Solve Ax = b and describe whether the system has zero, one, or many solutions."""
    coeffs = np.array(a, dtype=float)
    rhs = np.array(b, dtype=float).reshape(-1, 1)
    rows, cols = coeffs.shape
    if rhs.shape[0] != rows:
        raise ValueError("Vector b must have the same number of rows as A.")

    augmented = np.hstack([coeffs, rhs])
    steps: list[dict[str, Any]] = [
        {
            "step": 0,
            "description": "Start from the augmented matrix [A | b].",
            "matrix": _clean_small_values(augmented),
            "op_type": "initial",
            "pivot_col": None,
            "pivot_row": None,
        }
    ]

    pivot_cols: list[int] = []
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        max_row = pivot_row + np.argmax(np.abs(augmented[pivot_row:, col]))
        pivot = augmented[max_row, col]
        if abs(pivot) < EPSILON:
            continue

        if max_row != pivot_row:
            augmented[[pivot_row, max_row]] = augmented[[max_row, pivot_row]]
            steps.append(
                {
                    "step": len(pivot_cols) + 1,
                    "description": (
                        f"Swap row {pivot_row + 1} with row {max_row + 1} "
                        f"to expose a pivot in column {col + 1}."
                    ),
                    "matrix": _clean_small_values(augmented),
                    "op_type": "swap",
                    "pivot_col": col,
                    "pivot_row": pivot_row,
                    "target_row": max_row,
                }
            )

        pivot = augmented[pivot_row, col]
        augmented[pivot_row] /= pivot
        steps.append(
            {
                "step": len(pivot_cols) + 1,
                "description": f"Normalize row {pivot_row + 1} so the pivot in column {col + 1} becomes 1.",
                "matrix": _clean_small_values(augmented),
                "op_type": "normalize",
                "pivot_col": col,
                "pivot_row": pivot_row,
            }
        )

        for row in range(rows):
            if row == pivot_row:
                continue
            factor = augmented[row, col]
            if abs(factor) < EPSILON:
                continue
            augmented[row] -= factor * augmented[pivot_row]
            steps.append(
                {
                    "step": len(pivot_cols) + 1,
                    "description": (
                        f"Eliminate the entry in row {row + 1}, column {col + 1} "
                        f"using row {pivot_row + 1}."
                    ),
                    "matrix": _clean_small_values(augmented),
                    "op_type": "eliminate",
                    "pivot_col": col,
                    "pivot_row": pivot_row,
                    "target_row": row,
                    "factor": float(factor),
                }
            )

        pivot_cols.append(col)
        pivot_row += 1

    augmented = _clean_small_values(augmented)
    rank_a = len(pivot_cols)
    rank_aug = rank_a
    inconsistent_rows: list[int] = []

    for row in range(rows):
        coeff_zero = np.all(np.abs(augmented[row, :-1]) < EPSILON)
        rhs_nonzero = abs(augmented[row, -1]) >= EPSILON
        if coeff_zero and rhs_nonzero:
            rank_aug = rank_a + 1
            inconsistent_rows.append(row)

    solution: np.ndarray | None = None
    particular: np.ndarray | None = None
    null_space_basis: list[np.ndarray] = []
    free_cols = [col for col in range(cols) if col not in pivot_cols]

    if rank_a < rank_aug:
        solution_type = "no_solution"
    elif rank_a < cols:
        solution_type = "infinite_solutions"
        particular, null_space_basis, free_cols = _build_parametric_solution(
            augmented,
            pivot_cols,
            cols,
        )
    else:
        solution_type = "unique_solution"
        solution = np.zeros(cols)
        for row_idx, col in enumerate(pivot_cols):
            solution[col] = augmented[row_idx, -1]
        solution = _clean_small_values(solution)
        particular = solution

    return {
        "x": solution,
        "type": solution_type,
        "rank_A": rank_a,
        "rank_aug": rank_aug,
        "steps": steps,
        "rref": augmented,
        "pivot_cols": pivot_cols,
        "free_cols": free_cols,
        "particular": particular,
        "null_space_basis": null_space_basis,
        "inconsistent_rows": inconsistent_rows,
    }


def compute_rank(matrix: np.ndarray) -> dict[str, Any]:
    """Compute matrix rank through row reduction."""
    a = matrix.astype(float).copy()
    rows, cols = a.shape

    rank = 0
    pivot_row = 0

    for col in range(cols):
        if pivot_row >= rows:
            break

        max_row = pivot_row
        for row in range(pivot_row + 1, rows):
            if abs(a[row, col]) > abs(a[max_row, col]):
                max_row = row

        if abs(a[max_row, col]) < EPSILON:
            continue

        a[[pivot_row, max_row]] = a[[max_row, pivot_row]]
        a[pivot_row] /= a[pivot_row, col]

        for row in range(rows):
            if row != pivot_row:
                a[row] -= a[row, col] * a[pivot_row]

        rank += 1
        pivot_row += 1

    return {
        "rank": rank,
        "rref": _clean_small_values(a),
        "nullity": cols - rank,
    }
