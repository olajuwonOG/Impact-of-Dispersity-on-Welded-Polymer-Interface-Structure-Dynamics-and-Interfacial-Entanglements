#!/usr/bin/env python3
import numpy as np

def read_header(f):
    """Reads one header from the LAMMPS dump file."""
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

def wrap_coord(val, lo, hi):
    """Wrap a coordinate into [lo, hi) using periodic boundary conditions."""
    L = hi - lo
    return lo + ((val - lo) % L)

# Input and output filenames
fname = 'Pre_equilibration.dump'
outfile = 'density_1.4_360_R1.csv'

with open(fname, 'r') as f:
    frame = 0
    init_timestep, num_atoms, xlo, xhi, ylo, yhi, zlo, zhi = read_header(f)
    
    # arrays for time + box
    timestep = np.zeros(1, dtype=int)
    box_bounds = np.zeros((1, 3, 2), dtype=float)  # [frame, (x,y,z), (lo,hi)]
    timestep[frame] = init_timestep
    box_bounds[frame, 0, :] = [xlo, xhi]
    box_bounds[frame, 1, :] = [ylo, yhi]
    box_bounds[frame, 2, :] = [zlo, zhi]
    
    # positions and image flags (atoms are 1-indexed)
    r = np.zeros((1, num_atoms + 1, 3), dtype=float)
    ir = np.zeros((1, num_atoms + 1, 3), dtype=int)
    
    # --------- Parse ATOMS header (first frame) ----------
    line = f.readline().split()  # ITEM: ATOMS id type x y z ix iy iz ...

    id_index = line.index("id") - 2
    mol_index = line.index("mol") - 2 if "mol" in line else None
    type_index = line.index("type") - 2 if "type" in line else None

    # coordinates: scaled (xs,ys,zs) or unscaled (x,y,z)
    if "xs" in line:
        scaled = True
        x_index = line.index("xs") - 2
        y_index = line.index("ys") - 2
        z_index = line.index("zs") - 2
    elif "x" in line:
        scaled = False
        x_index = line.index("x") - 2
        y_index = line.index("y") - 2
        z_index = line.index("z") - 2
    else:
        raise RuntimeError("No x/xs coordinate column found in ATOMS header.")
    
    # image flags present?
    has_images = "ix" in line
    if has_images:
        ix_index = line.index("ix") - 2
        iy_index = line.index("iy") - 2
        iz_index = line.index("iz") - 2

    # --------- Read atom data for the first frame ----------
    Lx = box_bounds[frame, 0, 1] - box_bounds[frame, 0, 0]
    Ly = box_bounds[frame, 1, 1] - box_bounds[frame, 1, 0]
    Lz = box_bounds[frame, 2, 1] - box_bounds[frame, 2, 0]
    xlo = box_bounds[frame, 0, 0]
    ylo = box_bounds[frame, 1, 0]
    zlo = box_bounds[frame, 2, 0]
    xhi = box_bounds[frame, 0, 1]
    yhi = box_bounds[frame, 1, 1]
    zhi = box_bounds[frame, 2, 1]

    for _ in range(num_atoms):
        line = f.readline().split()
        my_id = int(line[id_index])

        x = float(line[x_index])
        y = float(line[y_index])
        z = float(line[z_index])

        if has_images:
            ix = int(line[ix_index])
            iy = int(line[iy_index])
            iz = int(line[iz_index])
        else:
            ix = iy = iz = 0

        # Convert to real coordinates
        if scaled:
            x = x * Lx + xlo
            y = y * Ly + ylo
            z = z * Lz + zlo

        # Wrap into primary box using PBC (handles unwrapped coords)
        x = wrap_coord(x, xlo, xhi)
        y = wrap_coord(y, ylo, yhi)
        z = wrap_coord(z, zlo, zhi)

        r[frame, my_id, 0] = x
        r[frame, my_id, 1] = y
        r[frame, my_id, 2] = z
        ir[frame, my_id, 0] = ix
        ir[frame, my_id, 1] = iy
        ir[frame, my_id, 2] = iz

    # --------- Read subsequent frames ----------
    frame = 1
    while True:
        try:
            my_timestep, my_num_atoms, my_xlo, my_xhi, my_ylo, my_yhi, my_zlo, my_zhi = read_header(f)
        except Exception:
            break  # End of file reached
        
        if my_num_atoms != num_atoms:
            raise RuntimeError("Number of atoms changed between frames; script assumes constant num_atoms.")
        
        # extend arrays
        timestep = np.append(timestep, my_timestep)
        box_bounds = np.concatenate((box_bounds, np.zeros((1, 3, 2), dtype=float)))
        r = np.concatenate((r, np.zeros((1, num_atoms + 1, 3), dtype=float)))
        ir = np.concatenate((ir, np.zeros((1, num_atoms + 1, 3), dtype=int)))

        box_bounds[frame, 0, :] = [my_xlo, my_xhi]
        box_bounds[frame, 1, :] = [my_ylo, my_yhi]
        box_bounds[frame, 2, :] = [my_zlo, my_zhi]

        # skip "ITEM: ATOMS ..." header line
        f.readline()

        Lx = box_bounds[frame, 0, 1] - box_bounds[frame, 0, 0]
        Ly = box_bounds[frame, 1, 1] - box_bounds[frame, 1, 0]
        Lz = box_bounds[frame, 2, 1] - box_bounds[frame, 2, 0]
        xlo = box_bounds[frame, 0, 0]
        ylo = box_bounds[frame, 1, 0]
        zlo = box_bounds[frame, 2, 0]
        xhi = box_bounds[frame, 0, 1]
        yhi = box_bounds[frame, 1, 1]
        zhi = box_bounds[frame, 2, 1]

        for _ in range(num_atoms):
            line = f.readline().split()
            my_id = int(line[id_index])

            x = float(line[x_index])
            y = float(line[y_index])
            z = float(line[z_index])

            if has_images:
                ix = int(line[ix_index])
                iy = int(line[iy_index])
                iz = int(line[iz_index])
            else:
                ix = iy = iz = 0

            if scaled:
                x = x * Lx + xlo
                y = y * Ly + ylo
                z = z * Lz + zlo

            # Always wrap into primary box
            x = wrap_coord(x, xlo, xhi)
            y = wrap_coord(y, ylo, yhi)
            z = wrap_coord(z, zlo, zhi)

            r[frame, my_id, 0] = x
            r[frame, my_id, 1] = y
            r[frame, my_id, 2] = z
            ir[frame, my_id, 0] = ix
            ir[frame, my_id, 1] = iy
            ir[frame, my_id, 2] = iz

        frame += 1

frames = frame  # total number of frames read

# ----- Calculate number density as a function of z position -----

shell_width = 1.0  # thickness of each z-bin
zlo_global = box_bounds[0, 2, 0]
zhi_global = box_bounds[0, 2, 1]
shell_edges = np.arange(zlo_global, zhi_global + shell_width, shell_width)
num_bins = len(shell_edges) - 1

density_counts = np.zeros((num_bins, frames), dtype=float)

for t in range(frames):
    zs = r[t, 1:, 2]  # atoms are 1-indexed
    counts, _ = np.histogram(zs, bins=shell_edges)
    density_counts[:, t] = counts

# average over frames
avg_counts = np.mean(density_counts, axis=1)

# box cross-sectional area
area = (box_bounds[0, 0, 1] - box_bounds[0, 0, 0]) * (box_bounds[0, 1, 1] - box_bounds[0, 1, 0])
Lz_global = zhi_global - zlo_global

number_density = avg_counts / (area * shell_width)

# Sanity check
rho_box = num_atoms / (area * Lz_global)
print("Density from box        =", rho_box)
print("Mean density from profile =", number_density.mean())
print("Sum of avg_counts (≈ num_atoms):", avg_counts.sum())

# Write out CSV: z_center, number_density
with open(outfile, 'w') as out:
    out.write("z_center, number_density\n")
    for i in range(num_bins):
        z_center = 0.5 * (shell_edges[i] + shell_edges[i+1])
        out.write(f"{z_center:.7f}, {number_density[i]:.7f}\n")

print("DONE")
