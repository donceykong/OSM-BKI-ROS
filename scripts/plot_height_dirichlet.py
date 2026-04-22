#!/usr/bin/env python3
"""Plot per-class height kernel results from height_dirichlet_mcd.yaml.

Shows:
  - Top: phi(h) for Dirichlet-derived params (mu, tau, dead_zone) vs old Gaussian (no dead zone)
  - Bottom: dead zone width and tau comparison across classes
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DIRICHLET_YAML = os.path.join(SCRIPT_DIR, "height_dirichlet_kitti360.yaml")
GAUSSIAN_YAML  = os.path.join(SCRIPT_DIR, "OSM-BKI-ROS/config/datasets/height_gaussians_kitti360.yaml")

with open(DIRICHLET_YAML) as f:
    diri = yaml.safe_load(f)
with open(GAUSSIAN_YAML) as f:
    gaus = yaml.safe_load(f)

class_names = diri["class_names"]
N = len(class_names)

mu_d   = diri["height_kernel_mu"]
tau_d  = diri["height_kernel_tau"]
dz_d   = diri["height_kernel_dead_zone"]

mu_g   = gaus["height_kernel_mu"]
tau_g  = gaus["height_kernel_tau"]

lam = 0.5  # from mcd.yaml

h = np.linspace(-10, 15, 2000)


def phi_kernel(h, mu, tau, dz):
    out = np.ones_like(h)
    for i, hi in enumerate(h):
        excess = max(0.0, abs(hi - mu) - dz)
        out[i] = np.exp(-(excess ** 2) / (2 * tau ** 2))
    return out


def scale(phi, lam):
    return (1 - lam) + lam * phi


# ── Grid: one subplot per class ──────────────────────────────────────────────
ncols = 3
nrows = int(np.ceil(N / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2), sharex=True)
axes = axes.flatten()

for c, name in enumerate(class_names):
    ax = axes[c]

    phi_new = phi_kernel(h, mu_d[c], tau_d[c], dz_d[c])
    phi_old = phi_kernel(h, mu_g[c], tau_g[c], 0.0)

    ax.plot(h, scale(phi_new, lam), color='steelblue',   linewidth=2,
            label=f'Dirichlet  (dz=±{dz_d[c]:.2f} m, τ={tau_d[c]:.2f})')
    ax.plot(h, scale(phi_old, lam), color='darkorange',  linewidth=1.5,
            linestyle='--', label=f'Gaussian   (dz=0, τ={tau_g[c]:.2f})')

    ax.axvspan(mu_d[c] - dz_d[c], mu_d[c] + dz_d[c],
               alpha=0.12, color='steelblue', label='dead zone')
    ax.axvline(mu_d[c], color='steelblue', linestyle=':', linewidth=1)
    ax.axvline(mu_g[c], color='darkorange', linestyle=':', linewidth=1)

    ax.set_title(name, fontweight='bold')
    ax.set_ylim(0.4, 1.05)
    ax.set_xlim(-10, 15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6.5, loc='lower right')
    ax.set_ylabel('ȳ scale')

for ax in axes[N:]:
    ax.set_visible(False)

for ax in axes[(nrows - 1) * ncols: N]:
    ax.set_xlabel('Height above ground (m)')

fig.suptitle('Per-class Height Kernel: Dirichlet (with dead zone) vs Gaussian (no dead zone)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
out1 = os.path.join(SCRIPT_DIR, "height_dirichlet_per_class.png")
plt.savefig(out1, dpi=150)
plt.show()
print(f"Saved {out1}")

# ── Summary bar chart: dead zone and tau per class ────────────────────────────
x = np.arange(N)
w = 0.35

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))

axes2[0].bar(x - w/2, dz_d,  width=w, color='steelblue',  label='Dirichlet dead zone (m)')
axes2[0].bar(x + w/2, [0]*N, width=w, color='darkorange', alpha=0.4, label='Gaussian dead zone (0)')
axes2[0].set_xticks(x)
axes2[0].set_xticklabels(class_names, rotation=30, ha='right')
axes2[0].set_ylabel('Dead zone half-width (m)')
axes2[0].set_title('Per-class Dead Zone')
axes2[0].legend()
axes2[0].grid(True, alpha=0.3, axis='y')

tau_d_plot = [min(t, 10) for t in tau_d]  # cap unlabeled=100 for visibility
tau_g_plot = [min(t, 10) for t in tau_g]
axes2[1].bar(x - w/2, tau_d_plot, width=w, color='steelblue',  label='Dirichlet τ')
axes2[1].bar(x + w/2, tau_g_plot, width=w, color='darkorange', label='Gaussian τ (MAD)', alpha=0.8)
axes2[1].set_xticks(x)
axes2[1].set_xticklabels(class_names, rotation=30, ha='right')
axes2[1].set_ylabel('τ (m)  [capped at 10]')
axes2[1].set_title('Per-class Tau Comparison')
axes2[1].legend()
axes2[1].grid(True, alpha=0.3, axis='y')

fig2.suptitle('Dirichlet vs Gaussian Parameter Summary', fontsize=12, fontweight='bold')
plt.tight_layout()
out2 = os.path.join(SCRIPT_DIR, "height_dirichlet_summary.png")
plt.savefig(out2, dpi=150)
plt.show()
print(f"Saved {out2}")
