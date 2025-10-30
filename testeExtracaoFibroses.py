import argparse
import numpy as np
from scipy.io import loadmat
import h5py
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
from skimage.measure import find_contours

# ============================================================
#                       LEITURA DO .MAT
# ============================================================

def _flatten_py(obj, out, prefix=""):
    """
    Varre recursivamente estruturas Python (dict/list) devolvidas por loadmat
    e produz um dicionário "achatado" com chaves tipo 'setstruct/Scar/Result'.
    """
    # dicionário
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith("__"):  # pula metadados do MATLAB
                continue
            _flatten_py(v, out, f"{prefix}{k}/")
    # lista ou tupla
    elif isinstance(obj, (list, tuple)):
        # salva o container e também indexa cada item
        out[prefix.rstrip("/")] = np.array(obj, dtype=object)
        for i, v in enumerate(obj):
            _flatten_py(v, out, f"{prefix}{i}/")

    # array n dimensional
    else:
        # tenta converter o valor em ndarray
        try:
            out[prefix.rstrip("/")] = np.array(obj)
        except Exception:
            pass

def load_any_mat(path):
    """
    Lê .mat v7 (scipy.io.loadmat)
    Retorna um dicionário achatado com chaves.
    Observação: também indexa cada entrada pelo "último nome" (sufixo).
    """
    flat = {} #Indexação dos sufixos

    # 1) v7
    try:
        md = loadmat(path, simplify_cells=True) 
        #Limpnando chaves que não interessam
        for k in list(md.keys()):
            if k.startswith("__"):
                md.pop(k, None)
        #achatar tudo e colocar em md
        _flatten_py(md, flat, "")
    except Exception:
        pass
    
    
    # 2) Buscar por sufixo (última parte do caminho)
    for k, v in list(flat.items()): #o list(flat.items()) é para evitar erro de dicionário mudando de tamanho
        flat[k.split("/")[-1]] = v

    if not flat:
        raise RuntimeError("Falha ao ler o .mat (nem scipy, nem h5py retornaram dados).")
    return flat


# ============================================================
#             CÁLCULO DE MÁSCARAS E BORDERZONE (2σ–5σ)
# ============================================================

def compute_masks_volume(mat_path, bz_range=(2.0, 5.0)):
    """
    Lê o .mat e calcula a Borderzone (2σ–5σ) para TODAS as fatias.
    Retorna um dict com:
      core (H,W,S), nrf (H,W,S), border (H,W,S), ring (H,W,S)
    Notas:
      - Intensidade: setstruct/Scar/IM (float) ou setstruct/IM
      - Máscaras: setstruct/Scar/NoReflow, /Result, /MyocardMask
      - 'ring' = preenchimento do MyocardMask (por fatia).
    """
    #h,w,s: altura, largura, número de fatias
    
    d = load_any_mat(mat_path)

    # volume de intensidade (necessário para μ e σ por fatia)
    if "setstruct/Scar/IM" in d and d["setstruct/Scar/IM"].ndim == 3:
        img = d["setstruct/Scar/IM"].astype(float)
    elif "setstruct/IM" in d and d["setstruct/IM"].ndim == 3:
        img = d["setstruct/IM"].astype(float)
    else:
        raise RuntimeError("Não encontrei intensidade (setstruct/Scar/IM ou setstruct/IM).")

    # máscaras do Segment
    nrf  = d["setstruct/Scar/NoReflow"].astype(bool)   # Verde (NRF)
    core = d["setstruct/Scar/Result"].astype(bool)     # Azul  (Core/Result)
    myo  = d["setstruct/Scar/MyocardMask"].astype(bool)
    H, W, S = img.shape

    # anel do miocárdio (fill holes slice a slice)
    ring = np.zeros_like(myo, dtype=bool)
    for k in range(S):
        ring[..., k] = binary_fill_holes(myo[..., k])

    # borderzone (mesma dimensão das máscaras) via z-score 2σ–5σ
    border = np.zeros_like(core, dtype=bool)

    for k in range(S):
        # normaliza imagem da fatia p/ [0,1] (estável para z-score)
        img2d = img[..., k]
        mn, mx = img2d.min(), img2d.max()
        if mx > mn:
            img2d = (img2d - mn) / (mx - mn)

        # remoto = anel sem core nem noreflow
        remote = ring[..., k] & (~core[..., k]) & (~nrf[..., k])
        vals = img2d[remote]
        mu = float(vals.mean()) if vals.size else 0.0
        sigma = float(vals.std(ddof=1)) if vals.size > 1 else 0.0

        if sigma > 0:
            z = (img2d - mu) / sigma
            smin, smax = bz_range
            border[..., k] = (
                ring[..., k] & (~core[..., k]) & (~nrf[..., k]) &
                np.isfinite(z) & (z >= smin) & (z < smax)
            )
        else:
            border[..., k] = False  # sem σ válido, não define BZ

    return dict(core=core, nrf=nrf, border=border, ring=ring)


# ============================================================
#                  PLOTS: 2D (fatia) e 3D (volume)
# ============================================================

def plot_fibrosis_slice(masks, slice_onebased, lw=2.0, title=None):
    """
    Plota SOMENTE os contornos da fatia escolhida (sem MRI).
    Cores: verde=NRF, azul=Core, amarelo=Borderzone. Mostra também o ring (preto).
    """
    core = masks["core"][..., slice_onebased - 1]
    nrf  = masks["nrf"][...,  slice_onebased - 1]
    bord = masks["border"][..., slice_onebased - 1]
    ring = masks["ring"][...,   slice_onebased - 1]

    H, W = core.shape
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)   # origem no canto superior esquerdo (convenção imagem)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        plt.title(title, fontsize=11)

    def draw(mask, color, width):
        if mask is None or not np.any(mask):
            return
        for c in find_contours(mask.astype(float), 0.5):
            ax.plot(c[:, 1], c[:, 0], color=color, linewidth=width)

    draw(ring, "black", 1.2)
    #draw(bord, "yellow", lw)
    draw(core, "blue", lw + 0.4)   # Azul = Core
    draw(nrf,  "green", lw + 0.4)  # Verde = NRF
    plt.show()


def plot_fibrosis_3d(masks, z_scale=6.0, voxel_size=(1.0, 1.0, 1.0), elev=25, azim=-60):
    """
    Desenha contornos 3D de todas as fatias.
    - z_scale: fator para "espaçar" as fatias. Ex.: 6.0
    - voxel_size: (dx, dy, dz) em mm (se quiser escala física). Use (1,1,1) se não tiver os metadados.
    - Ajusta box_aspect e limites para evitar o efeito "achatado".
    """
    core, nrf, border = masks["core"], masks["nrf"], masks["border"]
    H, W, S = core.shape
    dx, dy, dz = voxel_size
    zf = z_scale * dz

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")

    def draw(mask, color):
        for z in range(S):
            if np.any(mask[..., z]):
                for c in find_contours(mask[..., z].astype(float), 0.5):
                    x = c[:, 1] * dx              # escala física opcional
                    y = c[:, 0] * dy
                    zcoord = np.full_like(x, z * zf)
                    ax.plot(x, y, zcoord, color=color, linewidth=1)

    #draw(border, "yellow")
    draw(core,   "blue")   # Azul = Core
    draw(nrf,    "green")  # Verde = NRF

    # Aspect ratio consistente com as unidades escolhidas
    ax.set_box_aspect((W * dx, H * dy, max(1, (S - 1) * zf)))
    ax.set_xlim(0, W * dx)
    ax.set_ylim(H * dy, 0)
    ax.set_zlim(0, max(1, (S - 1) * zf))

    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (slice)")
    plt.title("Fibrosis contours (3D) — Green=Core • Blue=BorderZone")
    plt.tight_layout()
    plt.show()


# ============================================================
#                      EXECUÇÃO DIRETA (CLI)
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Plot de contornos de fibrose a partir de .mat do Segment.")
    ap.add_argument("--mat", required=True, help="Caminho do .mat (ex.: Patient_7.mat)")
    ap.add_argument("--mode", choices=["2d", "3d"], default="2d",
                    help="2d = fatia única; 3d = volume completo")
    ap.add_argument("--slice", type=int, default=7,
                    help="Fatia 1-based para o modo 2d (ex.: 6)")
    ap.add_argument("--z-scale", type=float, default=1.0,
                    help="Fator de escala do eixo Z no 3D (ex.: 6.0)")
    ap.add_argument("--dx", type=float, default=1.0, help="pixel spacing X (mm)")
    ap.add_argument("--dy", type=float, default=1.0, help="pixel spacing Y (mm)")
    ap.add_argument("--dz", type=float, default=1.0, help="slice thickness (mm)")
    args = ap.parse_args()

    masks = compute_masks_volume(args.mat, bz_range=(2.0, 5.0))

    if args.mode == "2d":
        plot_fibrosis_slice(
            masks,
            slice_onebased=args.slice,
            lw=2.0,
            title=f"Slice {args.slice} — Green=Core • Blue=BorderZone"
        )
    else:
        plot_fibrosis_3d(
            masks,
            z_scale=args.z_scale,
            voxel_size=(args.dx, args.dy, args.dz),
            elev=25, azim=-60
        )

if __name__ == "__main__":
    main()
