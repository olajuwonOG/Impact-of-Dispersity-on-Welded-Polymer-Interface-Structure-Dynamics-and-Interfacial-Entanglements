# Concentration profile for two welded films across z=0
# Step at tW=0; broadens with time by diffusion & roughening.
# We DO NOT shift coordinates; we analyze in z' = z - z_int(t) where z_int(t) is the 0.5 crossing of phi_left.

import sys
import numpy as np
import math as m
import time as tm

start_time = tm.time()
np.set_printoptions(threshold=np.inf)

# -------------------- IO helpers --------------------
def read_header(f):
    f.readline()  # ITEM: TIMESTEP
    timestep = int(f.readline())
    f.readline()  # ITEM: NUMBER OF ATOMS
    num_atoms = int(f.readline())
    f.readline()  # ITEM: BOX BOUNDS xx yy zz
    line = f.readline().split()
    xlo, xhi = float(line[0]), float(line[1])
    line = f.readline().split()
    ylo, yhi = float(line[0]), float(line[1])
    line = f.readline().split()
    zlo, zhi = float(line[0]), float(line[1])
    return timestep, num_atoms, xlo, xhi, ylo, yhi, zlo, zhi

def wrap(frame, r, box_bounds):
    t = frame
    bound_lo = np.array([box_bounds[t][0][0], box_bounds[t][1][0], box_bounds[t][2][0]])
    bound_hi = np.array([box_bounds[t][0][1], box_bounds[t][1][1], box_bounds[t][2][1]])
    boxsize = bound_hi - bound_lo
    for atom in range(1, len(r[t])):
        shift = np.zeros(3, float)
        for axis, coord in enumerate(r[t][atom]):
            if coord > bound_hi[axis]:
                shift[axis] = -boxsize[axis]
            elif coord < bound_lo[axis]:
                shift[axis] =  boxsize[axis]
        if np.sum(np.abs(shift)) != 0:
            r[t][atom] += shift
    return r

def find_interface_z(zcenters, cnt_left, cnt_right):
    """Find z where phi_left crosses 0.5 using linear interpolation."""
    total = cnt_left + cnt_right
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = np.where(total > 0, cnt_left / total, np.nan)
    # Look for a sign change of (phi - 0.5)
    s = phi - 0.5
    # Remove nans for robust crossing detection
    valid = ~np.isnan(s)
    zc = np.array(zcenters)[valid]
    sv = s[valid]
    if len(zc) < 2:
        return 0.0
    for i in range(len(zc) - 1):
        if sv[i] == 0:
            return float(zc[i])
        if (sv[i] > 0 and sv[i+1] < 0) or (sv[i] < 0 and sv[i+1] > 0):
            # linear interpolate between zc[i] and zc[i+1]
            z1, z2 = zc[i], zc[i+1]
            s1, s2 = sv[i], sv[i+1]
            return float(z1 + (z2 - z1) * (0 - s1) / (s2 - s1))
    # Fallbacks: fully one-sided → choose mid-plane
    return 0.0

# -------------------- User settings --------------------
print("Reading input file...")
fname = 'production.dump'
f = open(fname, 'r')

bin_width = 1.0  # LJ units

# -------------------- Read initial header --------------------
frame = 0
init_timestep, num_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_header(f)

# Unknown number of frames → allocate small and grow
num_frames = float('inf')
alloc = 1
inf_frames = True

timestep       = np.zeros(alloc, int)
box_bounds     = np.zeros([alloc, 3, 2], float)
box_dimensions = np.zeros([alloc, 3], float)
bin_volume     = np.zeros(alloc, float)

timestep[frame] = init_timestep
box_bounds[frame][0][0], box_bounds[frame][0][1] = xlo, xhi
box_bounds[frame][1][0], box_bounds[frame][1][1] = ylo, yhi
box_bounds[frame][2][0], box_bounds[frame][2][1] = zlo, zhi
box_dimensions[frame][0] = xhi - xlo
box_dimensions[frame][1] = yhi - ylo
box_dimensions[frame][2] = zhi - zlo
bin_volume[frame] = box_dimensions[frame][0] * box_dimensions[frame][1] * bin_width

# Coordinates/images
r   = np.zeros([alloc, num_atoms + 1, 3], float)
r_0 = np.zeros([num_atoms + 1, 3], float)
ir  = np.zeros([alloc, num_atoms + 1, 3], int)

# ID maps
id2mol  = np.zeros(num_atoms + 1, int)
id2type = np.zeros(num_atoms + 1, int)

# -------------------- Parse "ITEM: ATOMS ..." header for columns --------------------
line = f.readline().split()
id_index = line.index("id") - 2
mol_index  = line.index("mol")  - 2 if "mol"  in line else None
type_index = line.index("type") - 2 if "type" in line else None

if "x" in line:
    scaled = False
    x_index = line.index("x")  - 2
    y_index = line.index("y")  - 2
    z_index = line.index("z")  - 2
elif "xs" in line:
    scaled = True
    x_index = line.index("xs") - 2
    y_index = line.index("ys") - 2
    z_index = line.index("zs") - 2
else:
    raise RuntimeError("Dump must contain x/y/z or xs/ys/zs columns.")

ix_index = line.index("ix") - 2 if "ix" in line else None
iy_index = line.index("iy") - 2 if "iy" in line else None
iz_index = line.index("iz") - 2 if "iz" in line else None
if ix_index is None or iy_index is None or iz_index is None:
    raise RuntimeError("Dump must include ix/iy/iz image flags.")

# -------------------- Read first frame --------------------
for _ in range(num_atoms):
    rec = f.readline().split()
    my_id = int(rec[id_index])
    r[frame][my_id][0] = float(rec[x_index])
    r[frame][my_id][1] = float(rec[y_index])
    r[frame][my_id][2] = float(rec[z_index])
    ir[frame][my_id][0] = int(rec[ix_index])
    ir[frame][my_id][1] = int(rec[iy_index])
    ir[frame][my_id][2] = int(rec[iz_index])

    if scaled:
        r[frame][my_id][0] = r[frame][my_id][0]*box_dimensions[frame][0] - box_dimensions[frame][0]/2
        r[frame][my_id][1] = r[frame][my_id][1]*box_dimensions[frame][1] - box_dimensions[frame][1]/2
        r[frame][my_id][2] = r[frame][my_id][2]*box_dimensions[frame][2] - box_dimensions[frame][2]/2

    r_0[my_id][0] = r[frame][my_id][0] + ir[frame][my_id][0]*box_dimensions[frame][0]
    r_0[my_id][1] = r[frame][my_id][1] + ir[frame][my_id][1]*box_dimensions[frame][1]
    r_0[my_id][2] = r[frame][my_id][2] + ir[frame][my_id][2]*box_dimensions[frame][2]

    if mol_index is not None:  id2mol[my_id]  = int(rec[mol_index])
    if type_index is not None: id2type[my_id] = int(rec[type_index])

# provisional mol map
if mol_index is not None:
    num_mols = id2mol.max()
    mol2ids = [[]]
    for molid in range(1, num_mols+1):
        mol2ids.append(np.where(id2mol == molid)[0])
else:
    num_mols = 0
    mol2ids = [[]]

r = wrap(frame, r, box_bounds)

# -------------------- Read remaining frames --------------------
frame = 1
while frame < num_frames:
    try:
        my_timestep, my_num_atoms, my_xlo, my_xhi, my_ylo, my_yhi, my_zlo, my_zhi = read_header(f)
    except Exception:
        break  # EOF

    if inf_frames:
        timestep       = np.append(timestep, 0)
        box_bounds     = np.concatenate((box_bounds,     np.zeros([1, 3, 2], float)))
        box_dimensions = np.concatenate((box_dimensions, np.zeros([1, 3],     float)))
        bin_volume     = np.append(bin_volume, 0.)
        r  = np.concatenate((r,  np.zeros([1, num_atoms + 1, 3], float)))
        ir = np.concatenate((ir, np.zeros([1, num_atoms + 1, 3], int)))

    timestep[frame] = my_timestep
    box_bounds[frame][0][0], box_bounds[frame][0][1] = my_xlo, my_xhi
    box_bounds[frame][1][0], box_bounds[frame][1][1] = my_ylo, my_yhi
    box_bounds[frame][2][0], box_bounds[frame][2][1] = my_zlo, my_zhi
    box_dimensions[frame][0] = my_xhi - my_xlo
    box_dimensions[frame][1] = my_yhi - my_ylo
    box_dimensions[frame][2] = my_zhi - my_zlo
    bin_volume[frame] = box_dimensions[frame][0]*box_dimensions[frame][1]*bin_width

    f.readline()  # ITEM: ATOMS ...
    for _ in range(num_atoms):
        rec = f.readline().split()
        my_id = int(rec[id_index])
        r[frame][my_id][0] = float(rec[x_index])
        r[frame][my_id][1] = float(rec[y_index])
        r[frame][my_id][2] = float(rec[z_index])
        ir[frame][my_id][0] = int(rec[ix_index])
        ir[frame][my_id][1] = int(rec[iy_index])
        ir[frame][my_id][2] = int(rec[iz_index])

        if scaled:
            r[frame][my_id][0] = r[frame][my_id][0]*box_dimensions[frame][0] - box_dimensions[frame][0]/2
            r[frame][my_id][1] = r[frame][my_id][1]*box_dimensions[frame][1] - box_dimensions[frame][1]/2
            r[frame][my_id][2] = r[frame][my_id][2]*box_dimensions[frame][2] - box_dimensions[frame][2]/2

    r = wrap(frame, r, box_bounds)
    frame += 1

num_frames = frame

# -------------------- Recenter COM (no folding) --------------------
shift_cm = np.zeros([num_frames, 3], float)
shift_cm[0] = -np.sum(r_0, axis=0) / num_atoms

def safe_shift_ok(zshift, Lz, zvals):
    if zshift < 0:
        return abs(zshift) <= abs(-Lz/2.0 - np.min(zvals))
    else:
        return abs(zshift) <= abs( Lz/2.0 - np.max(zvals))

if safe_shift_ok(shift_cm[0,2], box_dimensions[0,2], r[0,:,2]):
    for atom in range(1, num_atoms+1):
        r[0][atom] += shift_cm[0]
else:
    print("Warning: COM shift would move atoms outside frame 0.")

for t in range(1, num_frames):
    shift_cm[t] = (shift_cm[0]/box_dimensions[0]) * box_dimensions[t]
    if safe_shift_ok(shift_cm[t,2], box_dimensions[t,2], r[t,:,2]):
        for atom in range(1, num_atoms+1):
            r[t][atom] += shift_cm[t]
    else:
        print(f"Warning: COM shift would move atoms outside frame {t}.")

# -------------------- Film-of-origin labels at t=0 --------------------
film_id = np.zeros(num_atoms + 1, dtype=np.int8)   # -1 left, +1 right, 0 exactly 0
for atom in range(1, num_atoms+1):
    z0 = r[0][atom][2]
    if z0 < 0.0:   film_id[atom] = -1
    elif z0 > 0.0: film_id[atom] = +1
    else:          film_id[atom] = 0
nleft0  = int((film_id == -1).sum())
nright0 = int((film_id == +1).sum())
print(f"Origin labels → left={nleft0}, right={nright0}, at0={(film_id==0).sum()}")

# -------------------- Chain-length filter (optional) --------------------
if id2mol.max() > 0:
    num_mols = id2mol.max()
    mol2ids = [[] for _ in range(num_mols + 1)]
    for molid in range(1, num_mols+1):
        ids = np.where(id2mol == molid)[0]
        if ids.size: mol2ids[molid] = ids.tolist()
else:
    num_mols = 0
    mol2ids = [[]]

min_chain_length = 345 # 0 #345 # 0 #355 #135
max_chain_length = 377 #2000 #377 # 2000 #365 #145
valid_mol_ids = [
    molid for molid in range(1, num_mols+1)
    if min_chain_length <= len(mol2ids[molid]) <= max_chain_length
]
use_molecules = (num_mols > 0) and (len(valid_mol_ids) > 0)
print(f"Num molecules: {num_mols}; valid (by length): {len(valid_mol_ids)}")

# -------------------- Profiles (two-pass per frame) --------------------
print("Calculating concentration profiles...")

# Output containers
bin_centers_rel = []     # centers in z' (relative to z_int(t))
conc_left       = []     # concentrations per frame in z'
conc_right      = []
phi_left        = []
phi_right       = []

for t in range(num_frames):
    # Pass 1: absolute bins to find z_int(t)
    Lz  = box_dimensions[t][2]
    Axy = box_dimensions[t][0]*box_dimensions[t][1]
    zmin_abs = -0.5*Lz
    zmax_abs =  0.5*Lz
    nbins = int(m.floor((zmax_abs - zmin_abs)/bin_width))
    edges_abs   = [zmin_abs + i*bin_width for i in range(nbins+1)]
    centers_abs = [zmin_abs + (i+0.5)*bin_width for i in range(nbins)]
    cntL_abs = np.zeros(nbins)
    cntR_abs = np.zeros(nbins)

    if use_molecules:
        for molid in valid_mol_ids:
            for atom in mol2ids[molid]:
                z = r[t][atom][2]
                idx = int(m.floor((z - zmin_abs)/bin_width))
                if 0 <= idx < nbins:
                    if   film_id[atom] == -1: cntL_abs[idx] += 1
                    elif film_id[atom] == +1: cntR_abs[idx] += 1
    else:
        for atom in range(1, num_atoms+1):
            z = r[t][atom][2]
            idx = int(m.floor((z - zmin_abs)/bin_width))
            if 0 <= idx < nbins:
                if   film_id[atom] == -1: cntL_abs[idx] += 1
                elif film_id[atom] == +1: cntR_abs[idx] += 1

    z_int = find_interface_z(centers_abs, cntL_abs, cntR_abs)

    # Pass 2: relative bins around z' = z - z_int (symmetric about 0)
    zmin_rel = -0.5*Lz
    nbins_rel = nbins
    centers_rel = [zmin_rel + (i+0.5)*bin_width for i in range(nbins_rel)]
    cntL_rel = np.zeros(nbins_rel)
    cntR_rel = np.zeros(nbins_rel)

    if use_molecules:
        for molid in valid_mol_ids:
            for atom in mol2ids[molid]:
                z_rel = r[t][atom][2] - z_int
                idx = int(m.floor((z_rel - zmin_rel)/bin_width))
                if 0 <= idx < nbins_rel:
                    if   film_id[atom] == -1: cntL_rel[idx] += 1
                    elif film_id[atom] == +1: cntR_rel[idx] += 1
    else:
        for atom in range(1, num_atoms+1):
            z_rel = r[t][atom][2] - z_int
            idx = int(m.floor((z_rel - zmin_rel)/bin_width))
            if 0 <= idx < nbins_rel:
                if   film_id[atom] == -1: cntL_rel[idx] += 1
                elif film_id[atom] == +1: cntR_rel[idx] += 1

    # Normalize → concentrations; compute fractions
    vol_bin = Axy*bin_width
    cL = cntL_rel/vol_bin
    cR = cntR_rel/vol_bin
    with np.errstate(divide='ignore', invalid='ignore'):
        tot = cL + cR
        phiL = np.where(tot > 0, cL/tot, 0.0)
        phiR = np.where(tot > 0, cR/tot, 0.0)

    bin_centers_rel.append(centers_rel)
    conc_left.append(cL)
    conc_right.append(cR)
    phi_left.append(phiL)
    phi_right.append(phiR)

# Diagnostics: crossings vs time
cross_LR = np.zeros(num_frames, dtype=int)
cross_RL = np.zeros(num_frames, dtype=int)
for t in range(num_frames):
    for atom in range(1, num_atoms+1):
        if film_id[atom] == -1 and r[t][atom][2] > 0.0: cross_LR[t] += 1
        elif film_id[atom] == +1 and r[t][atom][2] < 0.0: cross_RL[t] += 1

# -------------------- Write outputs --------------------
file_conc  = "concentration_profile_twofilms_test_N.csv"
file_frac  = "fraction_profile_twofilms_test_N.csv"
file_cross = "crossing_counts_twofilms_test_N.csv"

with open(file_conc, "w") as OUT:
    for t in range(num_frames):
        OUT.write(f"Frame {timestep[t]}\n")
        OUT.write("z_rel,conc_left,conc_right\n")
        for i, zc in enumerate(bin_centers_rel[t]):
            OUT.write(f"{zc:.6f},{conc_left[t][i]:.9f},{conc_right[t][i]:.9f}\n")

with open(file_frac, "w") as OUT:
    for t in range(num_frames):
        OUT.write(f"Frame {timestep[t]}\n")
        OUT.write("z_rel,phi_left,phi_right\n")
        for i, zc in enumerate(bin_centers_rel[t]):
            OUT.write(f"{zc:.6f},{phi_left[t][i]:.9f},{phi_right[t][i]:.9f}\n")

with open(file_cross, "w") as OUT:
    OUT.write("timestep,cross_left_to_right,cross_right_to_left,frac_LtoR,frac_RtoL\n")
    for t in range(num_frames):
        fL = cross_LR[t] / max(1, nleft0)
        fR = cross_RL[t] / max(1, nright0)
        OUT.write(f"{timestep[t]},{cross_LR[t]},{cross_RL[t]},{fL:.6f},{fR:.6f}\n")

print(f"Wrote: {file_conc}, {file_frac}, {file_cross}")
print(f"Execution time: {(tm.time()-start_time)/60:.2f} minutes")