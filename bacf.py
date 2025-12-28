#!/usr/bin/env python3
import numpy as np
import time as tm

start_time = tm.time()

# ----------------- User params -----------------
npoly = 4000
fname = 'Pre_equilibration.dump'
outfile = 'bond_autocorr_regions_long.csv'

FRAME_LIMIT = 200   # or None to read all frames

# Chain-length filter for test chains
min_chain_length = 420 # 0 #345
max_chain_length = 2000 # 140 # 375

# Surface thickness (sigma) measured inward from polymer free surface
surface_thickness = 2.0

# Optional: thickness (sigma) for center slab
center_width = 5.0
# ------------------------------------------------


def read_header(f):
    """Read one header from a (orthogonal) LAMMPS dump."""
    f.readline()  # ITEM: TIMESTEP
    timestep = int(f.readline())

    f.readline()  # ITEM: NUMBER OF ATOMS
    num_atoms = int(f.readline())

    f.readline()  # ITEM: BOX BOUNDS xx yy zz
    xlo, xhi = map(float, f.readline().split()[:2])
    ylo, yhi = map(float, f.readline().split()[:2])
    zlo, zhi = map(float, f.readline().split()[:2])
    return timestep, num_atoms, xlo, xhi, ylo, yhi, zlo, zhi


# ------------ Read dump (up to FRAME_LIMIT) ------------
with open(fname, 'r') as f:
    frame = 0
    init_timestep, num_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_header(f)

    timestep = []
    box_bounds = []
    r = []

    id2mol = np.zeros(num_atoms + 1, dtype=int)
    id2type = np.zeros(num_atoms + 1, dtype=int)

    # Parse ATOMS header (columns)
    cols = f.readline().split()
    id_index   = cols.index("id") - 2
    mol_index  = cols.index("mol") - 2 if "mol" in cols else None
    type_index = cols.index("type") - 2 if "type" in cols else None

    scaled = ("xs" in cols)
    x_index = cols.index("x" if not scaled else "xs") - 2
    y_index = cols.index("y" if not scaled else "ys") - 2
    z_index = cols.index("z" if not scaled else "zs") - 2

    # Store first frame
    timestep.append(init_timestep)
    box_bounds.append([[xlo, xhi], [ylo, yhi], [zlo, zhi]])

    r_frame = np.zeros((num_atoms + 1, 3), dtype=float)
    for _ in range(num_atoms):
        line = f.readline().split()
        my_id = int(line[id_index])
        rx = float(line[x_index]); ry = float(line[y_index]); rz = float(line[z_index])

        if scaled:
            rx = rx * (xhi - xlo) + xlo
            ry = ry * (yhi - ylo) + ylo
            rz = rz * (zhi - zlo) + zlo

        r_frame[my_id] = [rx, ry, rz]

        if mol_index is not None:
            id2mol[my_id] = int(line[mol_index])
        if type_index is not None:
            id2type[my_id] = int(line[type_index])

    r.append(r_frame)
    frame = 1

    # Build molecule to ids map from first frame
    num_mols = id2mol.max() if mol_index is not None else 0
    mol2ids = [np.where(id2mol == i)[0] for i in range(1, num_mols + 1)]

    # Filter molecules by chain length for test chains
    filtered_mol2ids = [mol for mol in mol2ids
                        if min_chain_length <= len(mol) <= max_chain_length]

    print(f"Filtered test chains: {len(filtered_mol2ids)} (length {min_chain_length}-{max_chain_length})")
    if len(filtered_mol2ids) == 0:
        raise RuntimeError(
            "No molecules passed the chain-length filter. "
            "Fix min_chain_length/max_chain_length to match your system."
        )

    # Read subsequent frames
    while True:
        if FRAME_LIMIT is not None and frame >= FRAME_LIMIT:
            break
        try:
            my_timestep, _, my_xlo, my_xhi, my_ylo, my_yhi, my_zlo, my_zhi = read_header(f)
        except Exception:
            break  # EOF

        timestep.append(my_timestep)
        box_bounds.append([[my_xlo, my_xhi], [my_ylo, my_yhi], [my_zlo, my_zhi]])

        f.readline()  # ITEM: ATOMS ...
        r_frame = np.zeros((num_atoms + 1, 3), dtype=float)
        for _ in range(num_atoms):
            line = f.readline().split()
            my_id = int(line[id_index])
            rx = float(line[x_index]); ry = float(line[y_index]); rz = float(line[z_index])

            if scaled:
                rx = rx * (my_xhi - my_xlo) + my_xlo
                ry = ry * (my_yhi - my_ylo) + my_ylo
                rz = rz * (my_zhi - my_zlo) + my_zlo

            r_frame[my_id] = [rx, ry, rz]

        r.append(r_frame)
        frame += 1

frames = len(r)
timestep = np.array(timestep, dtype=int)

print(f"Frames read: {frames} (FRAME_LIMIT={FRAME_LIMIT})")

# ------------------------------------------------------------
# NEW: Define polymer free-surface positions from the film itself
# We use ONLY atoms that belong to the filtered test chains (consistent).
# ------------------------------------------------------------
test_atom_ids = np.concatenate([mol for mol in filtered_mol2ids]).astype(int)
test_atom_ids = test_atom_ids[test_atom_ids > 0]  # remove 0 if present

z_test0 = r[0][test_atom_ids, 2]
z_poly_min0 = float(np.min(z_test0))
z_poly_max0 = float(np.max(z_test0))
z_poly_center0 = 0.5 * (z_poly_min0 + z_poly_max0)

print("\nDetected polymer slab from TEST-CHAIN atoms at t=0:")
print(f"  z_poly_min0 = {z_poly_min0:.3f}")
print(f"  z_poly_max0 = {z_poly_max0:.3f}")
print(f"  z_poly_center0 = {z_poly_center0:.3f}")
print(f"  surface_thickness = {surface_thickness:.3f} sigma")

# ------------- Define z ranges for each region (BASED ON POLYMER, NOT BOX) -------------
regions = {
    "left":  [(z_poly_min0, z_poly_min0 + surface_thickness)],
    "right": [(z_poly_max0 - surface_thickness, z_poly_max0)],
    "both":  [(z_poly_min0, z_poly_min0 + surface_thickness),
              (z_poly_max0 - surface_thickness, z_poly_max0)],
    "center": [(z_poly_center0 - 0.5 * center_width,
                z_poly_center0 + 0.5 * center_width)]
}

print("\nRegion z ranges (based on polymer at t=0):")
for name, ranges in regions.items():
    for zmin, zmax in ranges:
        print(f"  {name}: [{zmin:.3f}, {zmax:.3f}]")

# ------------- Select bonds per region at t = 0 -------------
selected_bonds = {name: [] for name in regions.keys()}

# NOTE: This keeps your original intent:
# - bonds are taken from filtered test chains
# - and only include bonds where id2type[id1] == 1 (your sticker/bond-type choice)
for mol_ids in filtered_mol2ids:
    ids = np.sort(mol_ids)  # assumes consecutive along chain
    for k in range(len(ids) - 1):
        id1 = int(ids[k])
        id2 = int(ids[k + 1])

        if type_index is not None:
            if id2type[id1] != 1:
                continue

        midz = 0.5 * (r[0][id1, 2] + r[0][id2, 2])

        for region_name, z_ranges in regions.items():
            for zmin, zmax in z_ranges:
                if zmin <= midz < zmax:
                    selected_bonds[region_name].append((id1, id2))
                    break

# Make bonds unique per region
for region_name in selected_bonds:
    selected_bonds[region_name] = list(set(selected_bonds[region_name]))
    print(f"Region {region_name}: {len(selected_bonds[region_name])} bonds selected")

if len(selected_bonds["left"]) == 0 and len(selected_bonds["right"]) == 0:
    print("\nWARNING: No surface bonds selected.")
    print("Most common causes:")
    print("  1) surface_thickness too small (try 5.0)")
    print("  2) id2type filter (id2type[id1] == 1) removes almost all surface bonds")
    print("  3) atom-id sorting is not true contour order (needs topology-based ordering)")

# ------------- Compute bond-vector autocorrelation per region -------------
corr = {name: np.zeros(frames, dtype=float) for name in regions.keys()}
counts = {name: np.zeros(frames, dtype=float) for name in regions.keys()}

for region_name, bonds in selected_bonds.items():
    if len(bonds) == 0:
        corr[region_name][:] = np.nan
        continue

    for (id1, id2) in bonds:
        v0 = r[0][id2] - r[0][id1]
        mag0 = np.linalg.norm(v0)
        if mag0 == 0.0:
            continue
        u0 = v0 / mag0

        for t in range(frames):
            vt = r[t][id2] - r[t][id1]
            magt = np.linalg.norm(vt)
            if magt == 0.0:
                continue
            ut = vt / magt
            corr[region_name][t] += np.dot(u0, ut)
            counts[region_name][t] += 1

    valid = counts[region_name] > 0
    corr[region_name][valid] /= counts[region_name][valid]
    corr[region_name][~valid] = np.nan

# ------------- Write output -------------
with open(outfile, 'w') as OUT:
    OUT.write("timestep, "
              "C_left, count_left, "
              "C_right, count_right, "
              "C_both, count_both, "
              "C_center, count_center\n")
    for t in range(frames):
        Cl = corr["left"][t]
        Cr = corr["right"][t]
        Cb = corr["both"][t]
        Cc = corr["center"][t]
        nl = int(counts["left"][t])
        nr = int(counts["right"][t])
        nb = int(counts["both"][t])
        nc = int(counts["center"][t])
        OUT.write(f"{timestep[t]}, "
                  f"{Cl:.7f}, {nl}, "
                  f"{Cr:.7f}, {nr}, "
                  f"{Cb:.7f}, {nb}, "
                  f"{Cc:.7f}, {nc}\n")

print("*************** CODE COMPLETE *********************")
elapsed_time_minutes = (tm.time() - start_time) / 60.0
print(f"Frames processed: {frames} (FRAME_LIMIT={FRAME_LIMIT})")
print(f"Execution time: {elapsed_time_minutes:.2f} minutes")
