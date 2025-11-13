import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_polys_from_xyz_txt(path):
    """
    Lê arquivo 'x y z' com cabeçalhos/linhas vazias possíveis.
    Retorna: lista de polígonos (cada polígono = np.array (N,3)).
    Divide por blocos separados por linha em branco ou cabeçalho '#'.
    """
    polys = []
    cur = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if cur:
                    polys.append(np.array(cur, dtype=float))
                    cur = []
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            cur.append([x, y, z])
    if cur:
        polys.append(np.array(cur, dtype=float))
    return polys

def load_points_from_reference_txt(path, z_value, z_tol=1e-6):
    """
    Lê seu TXT de referência (x y z) e retorna np.array(N,3) apenas com linhas cujo z == z_value (± z_tol).
    """
    pts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            x, y, z = map(float, parts[:3])
            if abs(z - z_value) <= z_tol:
                pts.append([x, y, z])
    return np.array(pts, dtype=float) if pts else np.zeros((0,3), dtype=float)

def apply_2d_xform(X, Y, H=None, W=None,
                   invert_y=False, rotate_k=0, flip_lr=False, flip_ud=False,
                   sx=1.0, sy=1.0, ox=0.0, oy=0.0):
    """
    Aplica transformações 2D de conveniência.
    - invert_y: se True, faz Y := -Y (ou Y := (H-1 - Y) se H informado). Aqui usamos o simples: Y := -Y.
      Se você preferir o modo imagem, passe H e troque a linha indicada abaixo.
    - rotate_k: 0,1,2,3 rotações de 90° (em coordenadas já contínuas).
    - flip_lr/flip_ud: espelhos em X ou Y.
    - sx, sy: escalas
    - ox, oy: offsets
    """
    x = X.copy()
    y = Y.copy()

    # flips
    if flip_lr:
        x = -x
    if flip_ud:
        y = -y

    # rotação 90*k (sentido anti-horário em conv. cartesiana)
    for _ in range(rotate_k % 4):
        x, y = -y, x

    # inversão do Y (duas opções; deixe só UMA ativa)
    if invert_y:
        # opção cartesiana simples:
        y = -y
        # opção "imagem": descomente e forneça H físico correto:
        # if H is not None:
        #     y = (H - y)

    # escala e offset
    x = sx * x + ox
    y = sy * y + oy
    return x, y

def plot_overlay(gz_polys, core_polys, ref_pts,
                 cfg_gz, cfg_core, cfg_ref,
                 title=None):
    fig, ax = plt.subplots(figsize=(7,7))

    # Greyzone (linhas contínuas)
    for P in gz_polys:
        X, Y, Z = P[:,0], P[:,1], P[:,2]
        X, Y = apply_2d_xform(X, Y, **cfg_gz)
        ax.plot(X, Y, linewidth=2.0, color="gold")

    # Core (linhas contínuas)
    for P in core_polys:
        X, Y, Z = P[:,0], P[:,1], P[:,2]
        X, Y = apply_2d_xform(X, Y, **cfg_core)
        ax.plot(X, Y, linewidth=2.4, color="blue")

    # Referência (se houver) — pontos pretos
    if ref_pts.size:
        Xr, Yr, Zr = ref_pts[:,0], ref_pts[:,1], ref_pts[:,2]
        Xr, Yr = apply_2d_xform(Xr, Yr, **cfg_ref)
        ax.scatter(Xr, Yr, s=10, c="k", alpha=0.7, label="referência")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    if title:
        ax.set_title(title)
    if ref_pts.size:
        ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser(description="Overlay: GZ/Core (XYZ por fatia) vs TXT de referência (x y z).")
    ap.add_argument("--gz-slice-file",  required=True, help="Arquivo da GZ da fatia (ex.: out_gz/gz_slice_006.txt)")
    ap.add_argument("--core-slice-file", required=True, help="Arquivo do Core da fatia (ex.: out_core/core_slice_006.txt)")
    ap.add_argument("--ref-file",        required=True, help="TXT de referência com x y z (todas as fatias)")
    ap.add_argument("--slice", type=int, required=True, help="Índice da fatia (z) a comparar (ex.: 6)")
    ap.add_argument("--z-tol", type=float, default=1e-6, help="Tolerância para filtrar z no arquivo de referência")
    # xform para GZ/Core/REF (podem diferir)
    for who in ["gz", "core", "ref"]:
        ap.add_argument(f"--{who}-invert-y", action="store_true", help=f"Inverter Y para {who}")
        ap.add_argument(f"--{who}-rotate-k", type=int, default=0, help=f"Rotação 90*k para {who} (0..3)")
        ap.add_argument(f"--{who}-flip-lr", action="store_true", help=f"Espelho LR (X) para {who}")
        ap.add_argument(f"--{who}-flip-ud", action="store_true", help=f"Espelho UD (Y) para {who}")
        ap.add_argument(f"--{who}-sx", type=float, default=1.0, help=f"Escala X para {who}")
        ap.add_argument(f"--{who}-sy", type=float, default=1.0, help=f"Escala Y para {who}")
        ap.add_argument(f"--{who}-ox", type=float, default=0.0, help=f"Offset X para {who}")
        ap.add_argument(f"--{who}-oy", type=float, default=0.0, help=f"Offset Y para {who}")
    args = ap.parse_args()

    # carrega dados
    gz_polys   = load_polys_from_xyz_txt(args.gz_slice_file)
    core_polys = load_polys_from_xyz_txt(args.core_slice_file)
    ref_pts    = load_points_from_reference_txt(args.ref_file, z_value=float(args.slice), z_tol=args.z_tol)

    # pacotes de config por grupo
    cfg_gz = dict(invert_y=args.gz_invert_y, rotate_k=args.gz_rotate_k,
                  flip_lr=args.gz_flip_lr, flip_ud=args.gz_flip_ud,
                  sx=args.gz_sx, sy=args.gz_sy, ox=args.gz_ox, oy=args.gz_oy)

    cfg_core = dict(invert_y=args.core_invert_y, rotate_k=args.core_rotate_k,
                    flip_lr=args.core_flip_lr, flip_ud=args.core_flip_ud,
                    sx=args.core_sx, sy=args.core_sy, ox=args.core_ox, oy=args.core_oy)

    cfg_ref = dict(invert_y=args.ref_invert_y, rotate_k=args.ref_rotate_k,
                   flip_lr=args.ref_flip_lr, flip_ud=args.ref_flip_ud,
                   sx=args.ref_sx, sy=args.ref_sy, ox=args.ref_ox, oy=args.ref_oy)

    # plota
    title = f"Overlay slice {args.slice} — GZ/Core vs referência"
    plot_overlay(gz_polys, core_polys, ref_pts, cfg_gz, cfg_core, cfg_ref, title=title)

if __name__ == "__main__":
    main()
