# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# February 2025
# Author: Sohil Kollipara
# Modified by Taofeek Tejuosho

# lammps types
# 1 end cap
# 2 normal bead
# 3 side chain start
# 4 side chain
# 5 side chain end cap

import math
import random
import pandas as pd
import numpy as np
import scipy.stats as st
from scipy.special import gamma, factorial
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
from datetime import datetime

# Function: generateInputFile
# inputs nbeadInput - number of beads (ignored), monomers per polymer (only used if monodisperesed) - number of monomers in each polymer, npoly

def get_random_direction(bond):
    dx = random.random()-0.5
    dy = random.random()-0.5
    dz = random.random()-0.5
    r = np.sqrt(dx*dx+dy*dy+dz*dz)
    scale = bond/r
    dx = scale*dx
    dy = scale*dy
    dz = scale*dz
    return dx,dy,dz

def generateInputFile(nbeadsInput, monomers_per_polymer_input, npoly_input, minsep_input, dens_input, bond_length_input, filename, nmin, nmax):
    class Schulz_Zimm(rv_continuous):
        def _pdf(self, x, z, Mn):
            #return (x**(alpha - 1) * np.exp(-x / beta)) / (beta**alpha * np.math.gamma(alpha))
            return (z**(z+1)/np.math.gamma(z+1))*(x**(z-1)/Mn**(z))*np.exp(-z*x/Mn)

    # Parameters for the distribution
    z = 1 #2.5 #2.5 #1 #2.5 # 2.5 #2.5 #1.25# 6.25   # Shape parameter to control dispersity
    Mn = 180 #257 #257 #357 #257 #33.3  # Scale parameter - Number average molecular weight
    num_samples = 6000 #Number of chains
    nmin = 0 #Minimum chain length
    nmax = 1600 #Maximum chain length
    # Create an instance of the custom gamma distribution to generate random samples from the distribution
    np.random.seed(23) #************ used to change - R1 23 and R2 is 24, except for D = 2.0 which is 21
    #np.random.seed(23) #************ used to change - R2
    SZ = Schulz_Zimm(momtype=1, a = nmin, b = nmax, name='SZ')
    monomers_per_polymer = SZ.rvs(z=z, Mn=Mn,  size=num_samples)
        #monomers_per_polymer = brentq(SZ.ppf, 0, 1000, args=(random.random(), z, Mn), maxiter=1000)
        # Create a histogram to visualize the distribution
    plt.hist(monomers_per_polymer, bins=30, edgecolor = "black", density=True, alpha=0.6, color='blue')

        # Calculate the theoretical distribution for visualization
    #x_vals = np.linspace(0, max(monomers_per_polymer), num_samples)
    #y_vals = SZ.pdf(x_vals, z = z, Mn=Mn)
    #plt.plot(x_vals, y_vals / np.trapz(y_vals, x_vals), color='red', label='Schulz-Zimm Distribution')
    #plt.ylim([0, 0.07])
    plt.xlabel('Chain Length')
    plt.ylabel('Probability Density')
    plt.title('Schulz Zimm Distribution')
    plt.legend()
    plt.show()
    #print(X)
    print("what type is this",type(monomers_per_polymer))
    monomers_per_polymer = monomers_per_polymer.astype(int)

    def remove_and_add(arr):
    # Find the minimum value in the array

    # Count the occurrences of 1 and 2 in the array
        count_zeros = np.count_nonzero(arr == 0)
        count_ones = np.count_nonzero(arr == 1)
        count_twos = np.count_nonzero(arr == 2)
        #count_threes = np.count_nonzero(arr == 3)

    # Create a new array with 0s and 1s, 2s and/or 3s removed
        #new_arr = arr[(arr != 0) & (arr != 1) & (arr !=2) & (arr !=3) ]
        new_arr = arr[(arr != 0) & (arr != 1) & (arr !=2) ]
        #new_arr = arr[(arr != 0) & (arr != 1)]
        min_val = np.min(new_arr)
    # Add the next lowest number to the array
        #new_arr = np.append(new_arr, [min_val] * (count_ones + count_zeros + count_twos + count_threes))
        new_arr = np.append(new_arr, [min_val] * (count_ones + count_zeros + count_twos))
        return new_arr
    original_array = monomers_per_polymer
    result_array = remove_and_add(monomers_per_polymer)

    monomers_per_polymer = result_array
    monomers_per_polymer = monomers_per_polymer.astype(int)
    list_mon = monomers_per_polymer.tolist()
    print("Number of the polymer chains", len(list_mon))
    print(sum(monomers_per_polymer))
    print(f"shortest chain is {min(monomers_per_polymer)}")
    print(f"longest chain is {max(monomers_per_polymer)}")
    print("Number of the polymer chains", len(list_mon))
    print(sum(monomers_per_polymer))
    """Calculating the dispersity"""
    z = Counter(list_mon)
    data = pd.DataFrame({"No_chains":z})
    data.index
    data.reset_index(inplace = True)
    data = data.rename(columns = {'index': "No_of_monomers"})
    data['weight_fraction'] = data["No_of_monomers"]*data["No_chains"]/sum(data["No_of_monomers"]* data["No_chains"])
    Mw = sum((data['No_of_monomers']*data['weight_fraction']))
    print("Average molecular weight is ", Mw)
    Mn = sum((data['No_of_monomers']*data['No_chains']))/sum(data['No_chains'])
    print("number average molecular weight is ", Mn)
    D = Mw/Mn
    print("Dispersity is ", D)
    a = np.array(list_mon)
    # Creating histogram
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(a, bins = 100, edgecolor = "black", density = True)
    ax.axvline(monomers_per_polymer.mean(), color='r', linestyle='dashed', linewidth=3)
    ax.axvline(Mw, color='g', linestyle='dashed', linewidth=3)
    #plt.title('SZ (MW = , D = 41.4)', fontsize = 30, fontweight = 'bold', fontname="Arial")

    plt.title(f'Schulz - Zimm Distribution, D = {round(D,1)}, Mw = {round(Mw,0)} & N = {num_samples}', fontsize = 18, fontweight = 'bold', fontname="Arial")
    plt.xlabel("Number of Beads/Molecular Weights of chains", fontsize = 20, fontname="Arial")
    plt.ylabel("Probability (Weight fraction)", fontsize = 20, fontname="Arial")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.ylim([0, 0.0175])
    #plt.xlim([0,1200])
    plt.show()
    pass
    #random_seed = 12
    polydispersity = True

    if nbeadsInput > 0:
        print("input params detected")
        nbeads = nbeadsInput
        if not polydispersity:
            monomers_per_polymer = monomers_per_polymer_input  # monomers in polymer (including x)
        npoly = npoly_input  # number of polymers
        minsep = minsep_input  # minimum seperation to prevent overlap
        # pendant_size = 1.0  # pendant group diameter/bead diameter
        dens = dens_input  # bead density
        bond_length = bond_length_input  # bond length
        #p_branch_end = p_branch_end_input
    else:
        print("no input params or incorrect params (7 fields)")
        print(
            "fields are:\nbeads per monomer (mainchain only)\nmonomers per polymer\nnumber of polymers\nminimum "
            "seperation to prevent overlap\nbead density\nbond length\nprobability of end of side branch")
        nbeads = 5  # beads per monomer mainchain
        #monomers_per_polymer = 3  # monomers in polymer (including x)
        npoly = 3  # number of polymers
        minsep = 1.0  # minimum seperation to prevent overlap
        # pendant_size = 1.0  # pendant group diameter/bead diameter
        dens = 0.85  # bead density
        bond_length = 0.97  # bond length
        p_branch_end = 0.5
    minsep2 = minsep ** 2
    print(minsep2)
    print("beads per monomer: ", nbeads)
    # print("number of beads in x segement of only backbone: ", nxbeads)
    #print("monomers per polymer: ", monomers_per_polymer)
    print("Maximum chain length is", max(monomers_per_polymer))
    print("Minimum chain length is", min(monomers_per_polymer))
    print("number of polymers: ", npoly)
    print("minimum seperation ", minsep)
    # print("pendant size: ", pendant_size)
    print("bead density: ", dens)
    bead_count = 0 #accumulator
    side_chain_total_length = 0
    polymers = []
    ntypes=1
    for i in range(npoly):
        monomers = []
        for j in range(monomers_per_polymer[i]):
            monomer = []
            # starting bead here (0)
            if j == 0:
                #TODO change starting bead back to endcap
                monomer.append(0)
                bead_count = bead_count + 1
            elif j == monomers_per_polymer[i] - 1:
                #TODO change back to endcap
                monomer.append(0)
                bead_count = bead_count + 1
            # regular main chain bead if not starting bead
            else:
                monomer.append(1)
                bead_count = bead_count + 1
            # append individual monomer to monomers list
            monomers.append(monomer)
        # append monomers that constitute polymer into polymer list
        polymers.append(monomers)

    INPUT_LAMMPS = open(filename, 'w')
    if polydispersity:
        nmonomers = np.sum(monomers_per_polymer)
    else:
        nmonomers = monomers_per_polymer * npoly
    #ntypes = len(type_names)
    # is this right?
    nbranches = nbeads * nmonomers
    ntot = bead_count
    vol = ntot / dens
    side = math.pow(vol/2, 1 / 3)
    dim = ntot
    hx, hy, hz = (2*side, side, side)
    hx2, hy2, hz2 = (hx / 2, hy / 2, hz / 2)
    if not polydispersity:
        nbonds = side_chain_total_length + (((monomers_per_polymer * nbeads) - 1) * npoly)
    else:
        ## TODO get working for branches
        nbonds = np.sum(monomers_per_polymer) - monomers_per_polymer.size
        nangles = nbonds - monomers_per_polymer.size
        ndihedrals = nangles - monomers_per_polymer.size
    print()
    print("Total number of particles:", ntot)
    print("Number of chains =", npoly)
    print("actual density", ntot / vol)
    print("Average beads in main chain =", np.average(monomers_per_polymer) * nbeads)
    print("beads in side chain =", side_chain_total_length)
    print("average side chain length =", side_chain_total_length / nmonomers)
    print("monomers total =", nmonomers)
    print("number of bonds =", nbonds)
    #print("Number of atoms types = ", ntypes)
    print("seed = ", random.seed)

    print()
    print("Geometry:")
    print("dens = ", dens)

    print("vol = ", vol)

    print()
    print("metric: %10.4f %10.4f %10.4f\n\n" % (hx, hy, hz))

    # position variables - initializes the position
    xc = np.zeros(dim)
    yc = np.zeros(dim)
    zc = np.zeros(dim)
    # velocity variables - initializes the velocity
    cx = np.zeros(dim)
    print("what is cx: ",cx.size)
    cy = np.zeros(dim)
    cz = np.zeros(dim)
    # Building polymers

    Rg2Ave = 0.0
    RgAve = 0.0
    Rend2Ave = 0.0
    types = np.zeros(dim)
    molnum = np.zeros(dim)
    q = np.zeros(dim)
    k = 0
    bonds = []

    for polymerNum in range(len(polymers)):
        lengthcurrentpoly = 0
        firstInPolymer = True
        for monomerNum in range(len(polymers[polymerNum])):
            inSideChain = False
            sideChainStart = (0, 0, 0)
            for unitNum in range(len(polymers[polymerNum][monomerNum])):
                unit = polymers[polymerNum][monomerNum][unitNum]
                types[k] = unit
                molnum[k] = polymerNum + 1
                if firstInPolymer:
                    xc[k] = random.random() * hx #- (0.5 * hx) #returns a random floating number between 0 and 1
                    yc[k] = random.random() * hy #- (0.5 * hy)
                    #zc[k] = random.random() * hz #- (hz/2) #- (hz/2) #- (0.5 * hz)
                    zc[k] = (random.random()*hz)  - (hz/2) #- (hz/2) # - (hz/2)
                    firstInPolymer = False
                else:
                    if unitNum > 0:
                        prevUnit = polymers[polymerNum][monomerNum][unitNum - 1]
                    else:
                        prevUnit = polymers[polymerNum][monomerNum - 1][-1]
                        # pick random direction; scale to be bond length
                    dx = random.random() - 0.5
                    dy = random.random() - 0.5
                    dz = random.random() - 0.5
                    r = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                    scale = bond_length / r
                    dx = scale * dx
                    dy = scale * dy
                    dz = scale * dz
                    if types[k] == 2:
                        inSideChain = True
                        xc[k] = xc[k - 1] + dx
                        yc[k] = yc[k - 1] + dy
                        zc[k] = zc[k - 1] + dz
                        sideChainStart = (xc[k], yc[k], zc[k])
                    elif types[k] == 4 or types[k] == 3:
                        xc[k] = xc[k - 1] + dx
                        yc[k] = yc[k - 1] + dy
                        zc[k] = zc[k - 1] + dz
                    elif (types[k] == 1 or types[k] == 0) and inSideChain:
                        xc[k] = sideChainStart[0] + dx
                        yc[k] = sideChainStart[1] + dy
                        zc[k] = sideChainStart[2] + dz
                        inSideChain = False
                    else:
                        # pick random direction; scale to be bond length
                        dx,dy,dz = get_random_direction(bond_length)
                        #if random direction would cause it to be outside z box direction, get another direction

                        #print("bond_length", math.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
                        #print("box", hx, hy, hz)
                        #while dz + zc[k-1] > hz or  zc[k-1] + dz < 0: #### ---- used before
                        while dz + zc[k-1]  > hz/2 - 0.97 or  zc[k-1] + dz  < -hz/2 + 0.97:
                        #while dz + zc[k-1]  > hz/2 - 0.97 or  zc[k-1] + dz  < -hz/2 + 0.97:
                            dx,dy,dz = get_random_direction(bond_length)
                        # xc[k] = xc[k-1] + dx
                        # yc[k] = yc[k-1] + dy
                        # zc[k] = zc[k-1] + dz

                        xc[k] = xc[k-1] + dx
                        yc[k] = yc[k-1] + dy
                        zc[k] = zc[k-1] + dz
                k = k + 1
    
    # #no charges
    # for k in range(types.size):
    #     if xc[k] > hx:
    #         cx[k] = int(xc[k] / hx)
    #         xc[k] = xc[k] - cx[k] * hx - hx2
    #     elif (xc[k] < 0.0):
    #         cx[k] = -int((-xc[k] + hx) / hx)
    #         xc[k] = xc[k] - cx[k] * hx - hx2
    #     else:
    #         cx[k] = 0
    #         xc[k] = xc[k] - hx2
    #     if yc[k] > hy:
    #         cy[k] = int(yc[k] / hy)
    #         yc[k] = yc[k] - cy[k] * hy - hy2
    #     elif yc[k] < 0.0:
    #         cy[k] = -int((-yc[k] + hy) / hy)
    #         yc[k] = yc[k] - cy[k] * hy - hy2
    #     else:
    #         cy[k] = 0
    #         yc[k] = yc[k] - hy2
    #     if zc[k] > hz:
    #         cz[k] = int(zc[k] / hz)
    #         zc[k] = zc[k] - cz[k] * hz - hz2
    #     elif zc[k] < 0.0:
    #         cz[k] = -int((-zc[k] + hz) / hz)
    #         zc[k] = zc[k] - cz[k] * hz - hz2
    #     else:
    #         cz[k] = 0
    #         zc[k] = zc[k] - hz2

    print("Polymers built.")
    x_date = datetime.now()

    INPUT_LAMMPS.write("#")
    INPUT_LAMMPS.write(x_date.strftime("%d-%B-%Y"))
    INPUT_LAMMPS.write("\n")
    INPUT_LAMMPS.write("%10i    atoms\n" % ntot)
    INPUT_LAMMPS.write("%10i    bonds\n" % nbonds)
    #INPUT_LAMMPS.write("%10i    angles\n" %     0)
    INPUT_LAMMPS.write("%10i    angles\n" % nangles)
    INPUT_LAMMPS.write("%10i    dihedrals\n" % 0)
    # INPUT_LAMMPS.write("%10i    impropers\n" % npendant)
    INPUT_LAMMPS.write("%10i    impropers\n" % 0)
    INPUT_LAMMPS.write("\n")
    #TODO change back to full types
    INPUT_LAMMPS.write("%10i    atom types\n" % 1)
    INPUT_LAMMPS.write("%10i    bond types\n" % 1)
    INPUT_LAMMPS.write("%10i    angle types\n" % 1)
    INPUT_LAMMPS.write("%10i    improper types\n" % 0)

    INPUT_LAMMPS.write("\n")
    INPUT_LAMMPS.write(" %16.8f %16.8f   xlo xhi\n" % (-hx2-0.8,hx2+0.8))
    INPUT_LAMMPS.write(" %16.8f %16.8f   ylo yhi\n" % (-hy2-0.8,hy2+0.8))
    INPUT_LAMMPS.write(" %16.8f %16.8f   zlo zhi\n" % (-hz2-0.8,hz2+0.8))
    INPUT_LAMMPS.write("\n")
    INPUT_LAMMPS.write("Atoms\n")
    INPUT_LAMMPS.write("\n")
    INPUT_LAMMPS.flush()
    mass = 1.0
    #chains
    for i in range(dim):
        unit = types[i]
        # could use a dictionary here between type and segname
        imol = molnum[i]  # this implies the polymers must be placed in before the counterions
        #TODO changed the default type to 1 (third input)

        INPUT_LAMMPS.write("%6i %6i %2i %6.2f %9.4f %9.4f %9.4f %6i %6i %6i\n" % (
            i+1, imol, 1, q[i], xc[i], yc[i], zc[i], cx[i], cy[i], 0))
    # Bonds
    INPUT_LAMMPS.write("\n")
    INPUT_LAMMPS.write("Bonds\n")
    INPUT_LAMMPS.write("\n")
    bonds = np.zeros(nbonds)

    inSideChain = False
    bond_count = 0
    sideChainStart = 0
    nbonds_actual = sum(monomers_per_polymer) - len(monomers_per_polymer)
    #for i in range(cx.size - 1):
    print("what is cx size:", cx.size)
    n = 0
    for i in range(cx.size - 1):
        if molnum[i + 1] == molnum[i]:
            bond_count = bond_count + 1
            n+=1
            if (types[i + 1] == 1 and types[i] == 0) or (types[i + 1] == 0 and types[i] == 1):
                INPUT_LAMMPS.write("%8i  1 %8i %8i\n" % (bond_count, i+1, i + 2))
            elif (types[i+1]==0 and types[i]==0):
                bond_count = bond_count-1
            elif (types[i + 1] == 2 and types[i] == 0):
                inSideChain = True
                sideChainStart = i + 1
                INPUT_LAMMPS.write("%8i  2 %8i %8i\n" % (bond_count, i+1, i + 2))
            elif (types[i + 1] == 0) and inSideChain:
                inSideChain = False
                INPUT_LAMMPS.write("%8i  2 %8i %8i\n" % (bond_count, sideChainStart+1, i + 2))
            elif (types[i + 1] == 1 and types[i] == 1):
                #TODO change back to 3
                INPUT_LAMMPS.write("%8i  1 %8i %8i\n" % (bond_count, i+1, i + 2))
            elif types[i + 1] == 2 and types[i] == 1:
                inSideChain = True
                sideChainStart = i + 1
                INPUT_LAMMPS.write("%8i  4 %8i %8i\n" % (bond_count, i+1, i + 2))
            elif (types[i + 1] == 1) and inSideChain:
                inSideChain = False
                INPUT_LAMMPS.write("%8i  4 %8i %8i\n" % (bond_count, sideChainStart+1, i + 2))
            elif types[i + 1] == 3 and types[i] == 2:
                INPUT_LAMMPS.write("%8i  5 %8i %8i\n" % (bond_count, i+1, i + 2))
            elif types[i + 1] == 4 and types[i] == 2:
                INPUT_LAMMPS.write("%8i  6 %8i %8i\n" % (bond_count, i+1, i + 2))
            elif (types[i + 1] == 3 and types[i] == 3):
                INPUT_LAMMPS.write("%8i  7 %8i %8i\n" % (bond_count, i+1, i + 2))
            elif types[i + 1] == 4 and types[i] == 3:
                INPUT_LAMMPS.write("%8i  8 %8i %8i\n" % (bond_count, i+1, i + 2))
                
    #Angles
    INPUT_LAMMPS.write("\n")
    INPUT_LAMMPS.write("Angles\n")
    INPUT_LAMMPS.write("\n")
    angles = np.zeros(nangles)
    n=0
    angle_count = 0 #building angles
    for i in range(1, cx.size-1):
        if molnum[i-1] == molnum[i] == molnum[i+1]:
            n+=1
            angle_count = angle_count + 1 #angle number
            jangle2 = i + 1
            jangle3 = i + 2
                #INPUT_LAMMPS.write("%8i  1 %8i %8i\n" % (bond_count, i+1, i + 2))
            INPUT_LAMMPS.write("%8i 1 %8i %8i %8i\n" % (angle_count, i-0, jangle2, jangle3))
            #if (types[i + 1] == 1 and types[i] == 0) or (types[i + 1] == 0 and types[i] == 1):
    print(f"what is n:", n)
    INPUT_LAMMPS.write("\n")
    INPUT_LAMMPS.write("Masses\n")
    INPUT_LAMMPS.write("\n")
    INPUT_LAMMPS.write("1  1.0\n")
    INPUT_LAMMPS.close()
    print("LAMMPS output complete.")

generateInputFile(1,0,6000,1,0.85,0.97,'input_Disperse_semiflexible.lammps',nmin = 0, nmax = 1600) 
#1200 #1600 - 1.4 and 2.0 respectively = Mw = 360
#300 #600 - 1.4 and 2.0 respectively = Mw = 140