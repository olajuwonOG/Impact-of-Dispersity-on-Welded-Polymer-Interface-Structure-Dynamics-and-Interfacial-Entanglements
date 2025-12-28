# Distribution of chain segments n_end/N that have crossed the interface
# (analogue of Fig. 5 in "Dynamics of polymers across an interface")

import sys
import numpy as np
import math as m
import time as tm

start_time = tm.time()
np.set_printoptions(threshold=np.inf)

# -------------------- IO helpers (UNCHANGED) --------------------
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

# -------------------- User settings --------------------
print("Reading input file...")
fname = 'production.dump'
f = open(fname, 'r')

bin_width_z = 1.0     # (used only for COM check etc., kept from original)
penetration_cut = 3.0 # σ: only monomers that have crossed by > 3σ
s_max = 0.5           # n_end / N ranges from 0 to 0.5
ds = 0.01             # bin width in contour coordinate

# -------------------- Read initial header (UNCHANGED) --------------------
frame = 0
init_timestep, num_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_header(f)

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
bin_volume[frame] = box_dimensions[frame][0] * box_dimensions[frame][1] * bin_width_z

# Coordinates/images
r   = np.zeros([alloc, num_atoms + 1, 3], float)
r_0 = np.zeros([num_atoms + 1, 3], float)
ir  = np.zeros([alloc, num_atoms + 1, 3], int)

# ID maps
id2mol  = np.zeros(num_atoms + 1, int)
id2type = np.zeros(num_atoms + 1, int)

# -------------------- Parse "ITEM: ATOMS ..." header (UNCHANGED) --------------------
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

# -------------------- Read first frame (UNCHANGED) --------------------
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

# provisional mol map (UNCHANGED)
if mol_index is not None:
    num_mols = id2mol.max()
    mol2ids = [[]]
    for molid in range(1, num_mols+1):
        mol2ids.append(np.where(id2mol == molid)[0])
else:
    num_mols = 0
    mol2ids = [[]]

r = wrap(frame, r, box_bounds)

# -------------------- Read remaining frames (UNCHANGED) --------------------
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
    bin_volume[frame] = box_dimensions[frame][0]*box_dimensions[frame][1]*bin_width_z

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

# -------------------- Recenter COM (UNCHANGED) --------------------
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

# -------------------- Film-of-origin labels at t=0 (UNCHANGED) --------------------
film_id = np.zeros(num_atoms + 1, dtype=np.int8)   # -1 left, +1 right, 0 exactly 0
for atom in range(1, num_atoms+1):
    z0 = r[0][atom][2]
    if z0 < 0.0:   film_id[atom] = -1
    elif z0 > 0.0: film_id[atom] = +1
    else:          film_id[atom] = 0
nleft0  = int((film_id == -1).sum())
nright0 = int((film_id == +1).sum())
print(f"Origin labels → left={nleft0}, right={nright0}, at0={(film_id==0).sum()}")

# -------------------- Chain-length filter (UNCHANGED) --------------------
if id2mol.max() > 0:
    num_mols = id2mol.max()
    mol2ids = [[] for _ in range(num_mols + 1)]
    for molid in range(1, num_mols+1):
        ids = np.where(id2mol == molid)[0]
        if ids.size: mol2ids[molid] = ids.tolist()
else:
    num_mols = 0
    mol2ids = [[]]

min_chain_length = 0
max_chain_length = 2000
valid_mol_ids = [
    molid for molid in range(1, num_mols+1)
    if min_chain_length <= len(mol2ids[molid]) <= max_chain_length
]
use_molecules = (num_mols > 0) and (len(valid_mol_ids) > 0)
print(f"Num molecules: {num_mols}; valid (by length): {len(valid_mol_ids)}")

# -------------------- NEW: build contour index per atom --------------------
# For each molecule, we assume atoms are ordered along the chain by atom id.
# We create:
#   pos_in_chain[atom] = index along backbone (0 .. N-1)
#   chain_len[atom]    = N for that chain
pos_in_chain = -np.ones(num_atoms + 1, dtype=int)
chain_len    = np.zeros(num_atoms + 1, dtype=int)

if use_molecules:
    for molid in valid_mol_ids:
        ids = np.array(mol2ids[molid], dtype=int)
        ids_sorted = np.sort(ids)   # assumes id order = contour order
        N = len(ids_sorted)
        for j, atom in enumerate(ids_sorted):
            pos_in_chain[atom] = j
            chain_len[atom]    = N

# -------------------- NEW: f_uptake(n_end/N) histograms --------------------
print("Calculating f_uptake(n_end/N) distributions...")

nbins_s = int(s_max / ds)
s_edges  = np.linspace(0.0, s_max, nbins_s + 1)
s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

# f_uptake[t, i] = histogram for frame t at contour bin i
f_uptake = np.zeros((num_frames, nbins_s), float)

for t in range(num_frames):
    # Loop over all atoms in valid chains
    for atom in range(1, num_atoms + 1):
        if pos_in_chain[atom] < 0:
            continue  # atom not in a valid chain

        z = r[t][atom][2]

        # Check if this monomer has crossed the interface by more than "penetration_cut"
        crossed = False
        if film_id[atom] == -1 and z >  penetration_cut:
            crossed = True   # came from left, now deep in right
        elif film_id[atom] == +1 and z < -penetration_cut:
            crossed = True   # came from right, now deep in left

        if not crossed:
            continue

        # Contour distance from nearest chain end
        N = chain_len[atom]
        j = pos_in_chain[atom]
        n_end = min(j, N - 1 - j)      # distance (in monomers) from closest end
        s = n_end / float(N)          # n_end / N in [0, 0.5]

        if 0.0 <= s <= s_max:
            ibin = int(s / ds)
            if ibin >= nbins_s:
                ibin = nbins_s - 1
            f_uptake[t, ibin] += 1.0

    # Normalize f_uptake for this frame to a probability density
    total_crossed = f_uptake[t, :].sum()
    if total_crossed > 0:
        f_uptake[t, :] /= (total_crossed * ds)

# -------------------- Write output: uptake distribution --------------------
file_uptake = "uptake_distribution_nend_over_N.csv"
with open(file_uptake, "w") as OUT:
    OUT.write("Frame,timestep\n")
    for t in range(num_frames):
        OUT.write(f"# Frame {t}, timestep {timestep[t]}\n")
        OUT.write("nend_over_N,f_uptake\n")
        for i, sc in enumerate(s_centers):
            OUT.write(f"{sc:.6f},{f_uptake[t, i]:.9f}\n")

print(f"Wrote: {file_uptake}")
print(f"Execution time: {(tm.time()-start_time)/60:.2f} minutes")