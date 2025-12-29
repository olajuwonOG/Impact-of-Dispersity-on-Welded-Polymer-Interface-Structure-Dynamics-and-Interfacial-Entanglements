#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import csv, re, sys, os
from pathlib import Path

# -------------------- user inputs --------------------
SP_FILE = "Z1+SP_1.0_500_300_R1.dat"                     # Z1+ -SP+ file for one snapshot
CHAIN_ORIGIN_CSV = "chain_origin_labels.csv"       # will be created if missing
LAMMPS_DATA_FILE = "1.0_500_300_R1_robbins.lmps"             # LAMMPS data (pre-interdiffusion) to build origin labels
BIN_WIDTH = 1.0
FARFIELD_BINS = 20

# ---- smoothing knobs ----
GAUSS_SIGMA_BINS_TOTAL = 0.8            # 0 disables smoothing (TOTAL only)
GAUSS_SIGMA_BINS_INTERFACIAL = 0.8      # 0 disables interfacial smoothing (NEW)

# ---- depth-scaling factor for TOTAL profile (1 = unchanged) ----
DEPTH_FACTOR = 2.0
# -----------------------------------------------------

ts_regex = re.compile(r'(\d+)')  # extract timestep from filename if present

def wrap_to_center(z, Lz):
    return ((z + 0.5 * Lz) % Lz) - 0.5 * Lz

def gaussian_smooth(y, sigma_bins):
    if sigma_bins is None or sigma_bins <= 0:
        return y.copy()
    r = max(1, int(3 * sigma_bins))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / float(sigma_bins))**2)
    k /= k.sum()
    ypad = np.pad(y, (r, r), mode="reflect")
    ys = np.convolve(ypad, k, mode="valid")
    return ys[:len(y)]

def load_chain_origins(path):
    d = {}
    with open(path, "r") as f:
        rd = csv.reader(f)
        _ = next(rd, None)
        for row in rd:
            if not row:
                continue
            d[int(row[0])] = int(row[1])
    return d

def build_chain_origins_from_lammps(data_path):
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Could not find LAMMPS data file: {data_path}")

    natoms = None
    Lz = None

    with open(data_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.endswith("atoms"):
                try:
                    natoms = int(s.split()[0])
                except Exception:
                    pass
            if s.endswith("zlo zhi"):
                parts = s.split()
                zlo, zhi = float(parts[0]), float(parts[1])
                Lz = zhi - zlo
            if s.startswith("Atoms"):
                break

    if natoms is None or Lz is None:
        raise RuntimeError("Failed to read 'atoms' count or 'zlo zhi' from data file.")

    mol_to_zsum = {}
    mol_to_count = {}

    with open(data_path, "r") as f:
        for line in f:
            if line.strip().startswith("Atoms"):
                break
        header = next(f, "")
        read_atoms = 0
        for line in f:
            s = line.strip()
            if not s or s[0].isalpha():
                break
            parts = s.split()
            if len(parts) < 10:
                continue
            try:
                mol = int(parts[1])
                z = float(parts[6])
                iz = int(parts[9])
            except Exception:
                continue
            z_unwrapped = z + iz * Lz
            mol_to_zsum[mol] = mol_to_zsum.get(mol, 0.0) + z_unwrapped
            mol_to_count[mol] = mol_to_count.get(mol, 0) + 1
            read_atoms += 1
            if read_atoms >= natoms:
                break

    film = {}
    for mol, n in mol_to_count.items():
        z_com = mol_to_zsum[mol] / max(1, n)
        film[mol] = -1 if z_com < 0.0 else (1 if z_com > 0.0 else 0)

    nL = sum(1 for v in film.values() if v == -1)
    nR = sum(1 for v in film.values() if v == +1)
    n0 = sum(1 for v in film.values() if v == 0)
    print(f"[labels] molecules: left={nL}, right={nR}, exactly_at_0={n0}, total={len(film)}")
    return film

def write_chain_origins_csv(path, film_dict):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["chain_id", "film_id"])
        for cid in sorted(film_dict.keys()):
            wr.writerow([cid, film_dict[cid]])
    print(f"[labels] wrote {path} with {len(film_dict)} chains.")

def parse_sp_plus(path):
    kinks = []
    with open(path, "r") as f:
        line = f.readline()
        C = int(line.split()[0])
        line = f.readline().split()
        Lx, Ly, Lz = map(float, line[:3])

        for chain_id in range(1, C + 1):
            n_line = f.readline()
            while n_line and n_line.strip() == "":
                n_line = f.readline()
            if not n_line:
                break
            n_nodes = int(n_line.split()[0])
            for _ in range(n_nodes):
                parts = f.readline().split()
                if len(parts) < 5:
                    continue
                zu = float(parts[2])
                kink = int(float(parts[4]))
                other_chain = int(float(parts[5])) if len(parts) > 5 else 0
                if kink == 1:
                    kinks.append((chain_id, other_chain, zu))
    return (Lx, Ly, Lz), kinks

def pair_kinks(raw_kinks, Lz):
    pairs = {}
    for c, oc, z in raw_kinks:
        if oc <= 0 or c == oc:
            continue
        a, b = (c, oc) if c < oc else (oc, c)
        pairs.setdefault((a, b), []).append(z)

    cA, cB, z_pair = [], [], []
    for (a, b), zs in pairs.items():
        if len(zs) == 1:
            zm = zs[0]
        else:
            z1, z2 = zs[0], zs[1]
            z2_adj = z2 - np.round((z2 - z1) / Lz) * Lz
            zm = 0.5 * (z1 + z2_adj)
        z_pair.append(wrap_to_center(zm, Lz))
        cA.append(a)
        cB.append(b)
    return np.array(cA, int), np.array(cB, int), np.array(z_pair, float)

def estimate_rho_bulk_by_ends(rho_all, farbins):
    n = len(rho_all)
    if farbins <= 0 or 2 * farbins >= n:
        return float(np.nanmedian(rho_all)) if n > 0 else 0.0
    left = rho_all[:farbins]
    right = rho_all[-farbins:]
    combined = np.concatenate([left, right])
    return float(np.nanmedian(combined))

def total_density_pairs(z_pairs, Lz, zmin, nbins, vol_bin):
    if z_pairs is None or len(z_pairs) == 0:
        return np.zeros(nbins, float)
    z_arr = np.array(z_pairs, float)
    idx = np.floor((z_arr - zmin) / BIN_WIDTH).astype(int)
    valid = (idx >= 0) & (idx < nbins)
    counts = np.bincount(idx[valid], minlength=nbins).astype(float)
    rho = counts / vol_bin
    return rho

def main(sp_path):
    ts = None
    m = ts_regex.search(Path(sp_path).name)
    if m:
        ts = int(m.group(1))

    # --- load or build chain origins ---
    if os.path.isfile(CHAIN_ORIGIN_CSV):
        film = load_chain_origins(CHAIN_ORIGIN_CSV)
        print(f"[labels] loaded {CHAIN_ORIGIN_CSV} with {len(film)} chains.")
    else:
        print(f"[labels] {CHAIN_ORIGIN_CSV} not found; deriving from {LAMMPS_DATA_FILE} ...")
        film = build_chain_origins_from_lammps(LAMMPS_DATA_FILE)
        write_chain_origins_csv(CHAIN_ORIGIN_CSV, film)

    (Lx, Ly, Lz), raw_kinks = parse_sp_plus(sp_path)
    cA, cB, z_k = pair_kinks(raw_kinks, Lz)

    # -------- INTERFACIAL profile --------
    fidA = np.array([film.get(int(c), 0) for c in cA], int)
    fidB = np.array([film.get(int(c), 0) for c in cB], int)
    interfacial_mask = (fidA * fidB) == -1

    zmin = -0.5 * Lz
    nbins = int(np.floor(Lz / BIN_WIDTH))
    centers = zmin + (np.arange(nbins) + 0.5) * BIN_WIDTH
    A = Lx * Ly
    vol_bin = A * BIN_WIDTH

    idx_I = np.floor((z_k - zmin) / BIN_WIDTH).astype(int)
    valid_I = (idx_I >= 0) & (idx_I < nbins)
    idx_I_v = idx_I[valid_I]
    mask_I_v = interfacial_mask[valid_I]

    counts_I = np.bincount(idx_I_v[mask_I_v], minlength=nbins).astype(float)
    rho_I = counts_I / vol_bin

    # -------- TOTAL profile --------
    rho_all_raw = total_density_pairs(z_k, Lz, zmin, nbins, vol_bin)
    rho_bulk = estimate_rho_bulk_by_ends(rho_all_raw, FARFIELD_BINS)
    rho_all_raw_n = (rho_all_raw / rho_bulk) if rho_bulk > 0 else rho_all_raw
    rho_all_smooth_n = gaussian_smooth(rho_all_raw_n, GAUSS_SIGMA_BINS_TOTAL)

    # -------- Apply DEPTH FACTOR (TOTAL ONLY) --------
    if DEPTH_FACTOR != 1.0:
        rho_all_raw_n = 1.0 - DEPTH_FACTOR * (1.0 - rho_all_raw_n)
        rho_all_smooth_n = 1.0 - DEPTH_FACTOR * (1.0 - rho_all_smooth_n)

    # -------- Normalize and smooth interfacial profile --------
    rho_I_n = (rho_I / rho_bulk) if rho_bulk > 0 else rho_I

    # optional smoothing for INTERFACIAL profile **(NEW)**
    rho_I_n_smooth = gaussian_smooth(rho_I_n, GAUSS_SIGMA_BINS_INTERFACIAL)

    N_I_over_A = float(np.sum(rho_I) * BIN_WIDTH)

    # --- write outputs ---
    out_all = "tc_profiles_all_1.0_500_1M_R1.csv"
    out_int = "tc_profiles_interfacial_1.0_500_1M_R1.csv"
    out_areal = "tc_areal_density_interfacial_1.0_500_1M_R1.csv"

    with open(out_all, "w") as f:
        f.write(f"Frame {ts}\n")
        f.write("z,rho_TC_over_rho_bulk_raw,rho_TC_over_rho_bulk_smooth\n")
        for zi, r_raw, r_s in zip(centers, rho_all_raw_n, rho_all_smooth_n):
            f.write(f"{zi:.6f},{r_raw:.9f},{r_s:.9f}\n")

    with open(out_int, "w") as f:
        f.write(f"Frame {ts}\n")
        f.write("z,rho_TC_I_over_rho_bulk_raw,rho_TC_I_over_rho_bulk_smooth\n")
        for zi, ri_raw, ri_s in zip(centers, rho_I_n, rho_I_n_smooth):
            f.write(f"{zi:.6f},{ri_raw:.9f},{ri_s:.9f}\n")

    with open(out_areal, "w") as f:
        f.write("timestep,N_TC_I_over_A\n")
        f.write(f"{ts if ts is not None else 0},{N_I_over_A:.9f}\n")

    print(f"Wrote: {out_all}, {out_int}, {out_areal}")
    print(f"[diag] rho_bulk_TC={rho_bulk:.6f},  N_I/A={N_I_over_A:.6f},  pairs={len(z_k)}")
if __name__ == "__main__":
    main(SP_FILE)