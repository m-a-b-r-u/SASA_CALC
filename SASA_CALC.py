import os
import re
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

RADII = {'H': 1.20, 'C': 1.85, 'N': 1.89, 'O': 1.52, 'F': 1.73, 'S': 2.49, 'Cl': 2.38, 'Br': 3.06, 'I': 2.74, 'P': 2.26}
COLORS = {'H': 'lightgrey', 'C': 'black', 'N': 'blue', 'O': 'red', 'F': 'green', 'S': 'yellow', 'Cl': 'green', 'Br': 'darkred'}

def parse_gamess(fp):
    with open(fp, 'r') as f:
        l = f.readlines()
    atom_l, coord_sec = [], False
    for ln in l:
        if '----- RESULTS FROM SUCCESSFUL RHF      GEOMETRY SEARCH -----' in ln:
            coord_sec = True
        elif coord_sec and re.match(r'^\s*-+\s*$', ln):
            continue
        elif coord_sec and '--- OPTIMIZED RHF      MO-S ---' in ln:
            coord_sec = False
        elif coord_sec:
            p = re.split(r'\s+', ln.strip())
            if len(p) >= 5 and p[0].isalpha() and p[1].replace('.', '', 1).isdigit():
                atom_l.append(p)
    atoms = [(p[0], float(p[2]), float(p[3]), float(p[4])) for p in atom_l if len(p) >= 5]
    return atoms

def parse_inp(fp):
    with open(fp, 'r') as f:
        l = f.readlines()
    atom_l, coord_sec = [], False
    for ln in l:
        if '$DATA' in ln:
            coord_sec = True
        elif coord_sec and re.match(r'^\s*$', ln):
            coord_sec = False
        elif coord_sec:
            p = re.split(r'\s+', ln.strip())
            if len(p) >= 4 and p[1].replace('.', '', 1).isdigit():
                atom_l.append(p)
    atoms = [(p[0], float(p[2]), float(p[3]), float(p[4])) for p in atom_l if len(p) >= 4]
    return atoms

def calc_sasa(atoms, probe_r=1.4, n_pts=5000):
    atom_a, total_a, atom_a_tot, surf_pts = np.zeros(len(atoms)), 0.0, {}, []
    for i, (a_type, x, y, z) in enumerate(atoms):
        if a_type not in RADII:
            continue
        radius = RADII[a_type] + probe_r
        points = gen_sphere_pts(n_pts) * radius
        access_pts = 0
        for pt in points:
            px, py, pz = pt + np.array([x, y, z])
            access = True
            for j, (o_type, x2, y2, z2) in enumerate(atoms):
                if i != j and np.linalg.norm([px - x2, py - y2, pz - z2]) < (RADII.get(o_type, 0) + probe_r):
                    access = False
                    break
            if access:
                access_pts += 1
                surf_pts.append((px, py, pz, a_type))
        area = (access_pts / n_pts) * 4 * np.pi * (RADII[a_type] ** 2)
        atom_a[i] = area
        total_a += area
        atom_a_tot[a_type] = atom_a_tot.get(a_type, 0) + area
    return atom_a_tot, total_a, surf_pts

def gen_sphere_pts(n_pts):
    indices = np.arange(0, n_pts, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y, z = np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)
    return np.stack((x, y, z), axis=-1)

def plot_surf_pts(surf_pts, mol_atoms):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    points = np.array([(x, y, z) for x, y, z, _ in surf_pts])
    atom_types = [atom_type for _, _, _, atom_type in surf_pts]
    for a_type in set(atom_types):
        mask = np.array([at == a_type for at in atom_types])
        ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], color=COLORS[a_type], label=a_type, s=1, alpha=0.5)
    mol_points = np.array([(x, y, z) for _, x, y, z in mol_atoms])
    mol_atom_types = [atom_type for atom_type, _, _, _ in mol_atoms]
    for a_type in set(mol_atom_types):
        mask = np.array([at == a_type for at in mol_atom_types])
        ax.scatter(mol_points[mask, 0], mol_points[mask, 1], mol_points[mask, 2], color=COLORS[a_type], edgecolor='k', s=50, label=f'{a_type} (molecule)')
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    #plt.legend()
    x_lims, y_lims, z_lims = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    plot_rad = 0.5 * max([x_lims[1] - x_lims[0], y_lims[1] - y_lims[0], z_lims[1] - z_lims[0]])
    x_mid, y_mid, z_mid = 0.5 * (x_lims[1] + x_lims[0]), 0.5 * (y_lims[1] + y_lims[0]), 0.5 * (z_lims[1] + z_lims[0])
    ax.set_xlim3d([x_mid - plot_rad, x_mid + plot_rad])
    ax.set_ylim3d([y_mid - plot_rad, y_mid + plot_rad])
    ax.set_zlim3d([z_mid - plot_rad, z_mid + plot_rad])
    plt.show()

def main():
    root = tk.Tk()
    root.withdraw()
    fp = filedialog.askopenfilename(title="Select a GAMESS .dat or .inp file", filetypes=[("DAT files", "*.dat"), ("INP files", "*.inp")])
    if not fp:
        print("No file selected.")
        return
    atoms = []
    if fp.endswith('.dat'):
        atoms = parse_gamess(fp)
    elif fp.endswith('.inp'):
        atoms = parse_inp(fp)
    if not atoms:
        print("No atomic coordinates found.")
        return
    atom_areas, total_area, surf_pts = calc_sasa(atoms)
    print("SASA for each atom type (in Å²):")
    for a_type, area in atom_areas.items():
        print(f"{a_type}: {area:.2f}")
    print(f"\nTotal SASA of the molecule: {total_area:.2f} Å²")
    plot_surf_pts(surf_pts, atoms)

if __name__ == "__main__":
    main()
