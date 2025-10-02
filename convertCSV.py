#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    if units == "um":#micrometros
        return P * 1e-4, 1e-4
    
    raise ValueError("units inválido. Use: mm | cm | um")

def load_points(input_mesh: str) -> np.ndarray:
    """Lê a malha com meshio e retorna apenas os pontos (N,3)."""
    mesh = meshio.read(input_mesh)
    pts = np.array(mesh.points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:  # matriz 2D com 3 colunas
        raise RuntimeError("Malha sem pontos 3D válidos (esperado N×3).")
    return pts

def load_tetrahedra(input_mesh: str) -> np.ndarray:
    """Extrai a conectividade dos tetraedros da malha e retorna como array (M,4)."""
    mesh = meshio.read(input_mesh)
    # Caso moderno: dicionário de células
    if hasattr(mesh, "cells_dict") and "tetra" in mesh.cells_dict: # se é cells_dict e se existe tetra
        tets = mesh.cells_dict["tetra"]
    else:
        # Fallback: varrer blocos
        tets = None
        for block in mesh.cells:
            if block.type in ("tetra", "tetra4", "tetra10"):
                tets = block.data
                break
    if tets is None:
        raise RuntimeError("Nenhum tetraedro encontrado na malha.")
    tets = np.asarray(tets, dtype=np.int64)

    # Se a malha vier numerada a partir de 0 (zero-based), soma +1 para ficar 1-based.
    # Depois garante que só retornamos os 4 vértices principais de cada tetraedro
    # (descarta nós extras em tetras de ordem mais alta, como tetra10).
    if tets.min() == 0:  # ajustar para 1-based
        tets = tets + 1
    return tets[:, :4]

def main():
    if len(sys.argv) != 3:
        print("Uso: python export_mesh.py <entrada.msh|entrada.vtu> <units>", file=sys.stderr)
        print("Ex.: python export_mesh.py Coarse.vtu cm", file=sys.stderr)
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

    # 3) Exportar pontos XYZ
    out_xyz = "coarse_xyz.csv"
    np.savetxt(out_xyz, pts_cm, fmt="%.6f", delimiter=",")

    # 4) Exportar tetraedros (se existirem)
    try:
        tets = load_tetrahedra(input_mesh)
        out_tet = "coarse_tetra.csv"
        np.savetxt(out_tet, tets, fmt="%d", delimiter=",")
        tet_info = f"Tetraedros (M) : {tets.shape[0]} → {out_tet}"
    except RuntimeError:
        tet_info = "Nenhum tetraedro encontrado na malha."

    # 5) Relato
    bbox_min = pts_cm.min(axis=0).tolist()
    bbox_max = pts_cm.max(axis=0).tolist()
    print("== Export concluído ==")
    print(f"Entrada        : {input_mesh}")
    print(f"Pontos (N)     : {pts.shape[0]} → {out_xyz}")
    print(f"Units -> cm    : {units}  (fator aplicado: {factor})")
    print(tet_info)
    print(f"Range (cm)     : min={bbox_min}  max={bbox_max}")

if __name__ == "__main__":
    main()
