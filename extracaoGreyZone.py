import argparse
import os
import numpy as np
from scipy.io import loadmat
from skimage.measure import find_contours
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Leitura do .mat: setstruct/Scar/GreyZone/map
# ------------------------------------------------------------
def load_gz_map_from_mat(mat_path):
    md = loadmat(mat_path, simplify_cells=True)
    try:
        lbl = md["setstruct"]["Scar"]["GreyZone"]["map"]
    except Exception as e:
        raise RuntimeError("Não encontrei 'setstruct/Scar/GreyZone/map' no .mat.") from e
    lbl = np.array(lbl)
    if lbl.ndim == 2:
        lbl = lbl[..., None]  # (H,W,1)
    if lbl.ndim != 3:
        raise ValueError("Esperava (H,W,S) para GreyZone.map.")
    return lbl  # 0=bg, 1=GZ, 2=Core


def extract_contours_all_slices(lbl, value, dx=1.0, dy=1.0, invert_y=False):
    #dx e dy em mm/pixel
    H, W, S = lbl.shape #altura, largura e número de fatias
    all_slices = []
    for s in range(S):
        #converte para float para find_contours que percorre a imagem binária
        # e contra curvas onde a imagem assume valor "level"
        mask = (lbl[:, :, s] == value).astype(float) 
        polys = [] # guardar polígonos de cada fatia
        if np.any(mask):
            cs = find_contours(mask, level=0.5)
            for c in cs:
                y, x = c[:, 0], c[:, 1]
                X = x * dx #escalona
                Y = (H - 1 - y) * dy if invert_y else y * dy #trtar o y e inverte se precisar (debug)

                #empilha arrays de uma dimensão de em colunas para formar array bidimensional
                polys.append(np.column_stack([X, Y]))  # (N,2) empilha a lista de polígonos
        all_slices.append(polys)
    return all_slices  # len = S cada item é lista de polígonos (N,2)


def save_xyz_per_slice(polys_per_slice, out_dir, prefix, dz=1.0, z_scale=1.0, start_index=1, z_offset=0.0):
    """
    polys_per_slice: lista de slices: cada item é lista de polígonos (N,2) em (x,y)
    out_dir: diretório de saída
    prefix: prefixo do arquivo (ex.: 'gz' ou 'core')
    dz: espessura da fatia (mm)
    z_scale: fator multiplicativo do espaçamento entre fatias
    start_index: índice 1-based da primeira fatia (apenas para nome do arquivo)
    z_offset: deslocamento adicional em Z (mm), opcional
    """
    os.makedirs(out_dir, exist_ok=True)
    S = len(polys_per_slice)
    written = 0
    for s in range(S):
        polys = polys_per_slice[s]
        if not polys:
            continue  # sem contorno nesta fatia → não salva arquivo
        z = z_offset + (s * (dz * max(0.0, z_scale)))
        fname = os.path.join(out_dir, f"{prefix}_slice_{s+start_index:03d}.txt")
        with open(fname, "w") as f:
            f.write(f"# {prefix} — slice {s+start_index}, polygons: {len(polys)}, z={z:.6f}\n")
            for pi, poly in enumerate(polys, start=1):
                f.write(f"# poly {pi}, points: {len(poly)}\n")
                for x, y in poly:
                    f.write(f"{x:.6f}\t{y:.6f}\t{z:.6f}\n")
                f.write("\n")
        written += 1
    return written

# ------------------------------------------------------------
# Plots (opcionais)
# ------------------------------------------------------------
def plot_2d_slice(gz_slices, core_slices, slice_idx, title=None,
                  gz_color="gold", core_color="blue"):
    s = slice_idx - 1
    fig, ax = plt.subplots(figsize=(6, 6))
    # Greyzone (uma cor única)
    for poly in gz_slices[s]:
        ax.plot(poly[:, 0], poly[:, 1], color=gz_color, linewidth=2.0, label=None)
    # Core (uma cor única)
    for poly in core_slices[s]:
        ax.plot(poly[:, 0], poly[:, 1], color=core_color, linewidth=2.4, label=None)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_3d_stack(gz_slices, core_slices, dz=1.0, z_scale=6.0, elev=25, azim=-60,
                  gz_color="gold", core_color="blue", z_offset=0.0):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    S = len(gz_slices)
    step = dz * max(0.0, z_scale)

    # Greyzone
    for s, polys in enumerate(gz_slices):
        z = z_offset + s * step
        for poly in polys:
            xs, ys = poly[:, 0], poly[:, 1]
            zs = np.full_like(xs, z)
            ax.plot(xs, ys, zs, color=gz_color, linewidth=1.2)

    # Core
    for s, polys in enumerate(core_slices):
        z = z_offset + s * step
        for poly in polys:
            xs, ys = poly[:, 0], poly[:, 1]
            zs = np.full_like(xs, z)
            ax.plot(xs, ys, zs, color=core_color, linewidth=1.4)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, max(1, S * 0.2)))
    plt.title("Contornos empilhados — Greyzone (ouro) • Core (azul)")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Exporta contornos x y z por fatia (Greyzone/Core) a partir do .mat do Segment.")
    ap.add_argument("--mat", required=True, help="Caminho do .mat (Segment)")
    ap.add_argument("--outdir-gz", default="out_gz", help="Diretório de saída para arquivos da Greyzone")
    ap.add_argument("--outdir-core", default="out_core", help="Diretório de saída para arquivos do Core")
    ap.add_argument("--dx", type=float, default=1.0, help="pixel spacing X (mm)")
    ap.add_argument("--dy", type=float, default=1.0, help="pixel spacing Y (mm)")
    ap.add_argument("--dz", type=float, default=1.0, help="espessura da fatia (mm)")
    ap.add_argument("--z-scale", type=float, default=6.0, help="fator para afastar/aproximar fatias em Z (multipl. de dz)")
    ap.add_argument("--z-offset", type=float, default=0.0, help="deslocamento absoluto em Z (mm)")
    ap.add_argument("--invert-y", action="store_true", help="inverte Y (origem embaixo, cartesiano)")
    ap.add_argument("--plot", choices=["none", "2d", "3d"], default="none", help="opcional: plota 2d/3d")
    ap.add_argument("--slice", type=int, default=1, help="fatia 1-based para o plot 2d")
    args = ap.parse_args()

    lbl = load_gz_map_from_mat(args.mat)

    # extrair contornos
    gz_slices   = extract_contours_all_slices(lbl, value=1, dx=args.dx, dy=args.dy, invert_y=args.invert_y)
    core_slices = extract_contours_all_slices(lbl, value=2, dx=args.dx, dy=args.dy, invert_y=args.invert_y)

    # salvar um .txt por fatia (x y z)
    n_gz   = save_xyz_per_slice(gz_slices,   args.outdir_gz,   prefix="gz",
                                dz=args.dz, z_scale=args.z_scale, z_offset=args.z_offset)
    n_core = save_xyz_per_slice(core_slices, args.outdir_core, prefix="core",
                                dz=args.dz, z_scale=args.z_scale, z_offset=args.z_offset)
    print(f"Salvos {n_gz} arquivos GZ em '{args.outdir_gz}' e {n_core} arquivos Core em '{args.outdir_core}'.")

    # plots opcionais
    if args.plot == "2d":
        plot_2d_slice(gz_slices, core_slices, slice_idx=args.slice,
                      title=f"Slice {args.slice} — Greyzone + Core")
    elif args.plot == "3d":
        plot_3d_stack(gz_slices, core_slices, dz=args.dz, z_scale=args.z_scale, z_offset=args.z_offset)

if __name__ == "__main__":
    main()
