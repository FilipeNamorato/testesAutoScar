import sys
import os
import numpy as np
import meshio

def to_cm(points: np.ndarray, units: str) -> tuple[np.ndarray, float]:
    """Converte para cm de acordo com 'units'. Retorna (pts_em_cm, fator_aplicado)."""
    P = np.asarray(points, dtype=float)
    if units == "cm":
        return P, 1.0
    if units == "mm":
        return P * 0.1, 0.1
    if units == "um":
        return P * 1e-4, 1e-4
    
    raise ValueError("units inválido. Use: auto | mm | cm | um")

def load_points(input_mesh: str) -> np.ndarray:
    """Lê a malha com meshio e retorna apenas os pontos (N,3)."""

    mesh = meshio.read(input_mesh)
    pts = np.array(mesh.points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3: #matriz 2D com 3 colunas
        raise RuntimeError("Malha sem pontos 3D válidos (esperado N×3).")
    return pts

def main():
    if len(sys.argv) != 3:
        print("Uso: python export_coarse_xyz_min.py <entrada.msh|entrada.vtu> <units>", file=sys.stderr)
        print("Ex.: python export_coarse_xyz_min.py Coarse.vtu cm", file=sys.stderr)
        sys.exit(1)

    input_mesh = sys.argv[1]
    units = sys.argv[2].lower().strip()

    if not os.path.isfile(input_mesh):
        print(f"Arquivo não encontrado: {input_mesh}", file=sys.stderr)
        sys.exit(1)

    # 1) Ler nós
    pts = load_points(input_mesh)

    # 2) Unidades -> cm
    pts_cm, factor = to_cm(pts, units)

    # 3) Exportar CSV (separador vírgula, 6 casas)
    stem = os.path.splitext(os.path.basename(input_mesh))[0]
    out_csv = f"coarse_xyz.csv"
    np.savetxt(out_csv, pts_cm, fmt="%.6f", delimiter=",")

    # 4) Relato enxuto
    bbox_min = pts_cm.min(axis=0).tolist()
    bbox_max = pts_cm.max(axis=0).tolist()
    print("== Export concluído ==")
    print(f"Entrada        : {input_mesh}")
    print(f"Pontos (N)     : {pts.shape[0]}")
    print(f"Units -> cm    : {units}  (fator aplicado: {factor})")
    print(f"Saída          : {out_csv}")
    print(f"Range (cm)      : min={bbox_min}  max={bbox_max}")

if __name__ == "__main__":
    main()
