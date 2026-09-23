"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================

Wannier-weight ("fat band") plot along the interpolated k-path.

``band_interpolation`` stores the Wannier-interpolated Hamiltonian ``H_skab`` along the
k-path. Its eigenvectors ``U_an(k)`` give the composition of every interpolated band
``n`` in terms of the Wannier orbitals ``a``; ``|U_an(k)|^2`` summed over a chosen set of
orbitals is the weight this module plots as the colour of the band.
"""

from dataclasses import dataclass
import numpy as np
from scipy.constants import physical_constants
Hartree_eV = physical_constants['Hartree energy in eV'][0]
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from h5 import HDFArchive
from coqui import app_log

# Default colour scale for the weight: a diverging blue <-> red pair around a mid gray, so that
# a 50/50 hybrid stays visible on a white background and both poles are saturated.
DEFAULT_WEIGHT_CMAP = LinearSegmentedColormap.from_list(
  "wannier_weight", ["#1f5fbf", "#8c8c8c", "#c8102e"])


@dataclass
class WannierWeightsOnKpath:
  """
  Interpolated bands along the k-path and their Wannier-orbital composition.

  Attributes
  ----------
  E_skn : ndarray, shape (ns, nkpts, nbnd)
      Band energies in eV, relative to the chemical potential, ascending in ``n``.
  w_skan : ndarray, shape (ns, nkpts, nImpOrbs, nbnd)
      ``|U_an(k)|^2``: weight of Wannier orbital ``a`` in interpolated band ``n``.
      Sums to one over ``a`` and over ``n``.
  kpt_label_idx : ndarray
      Indices (1-based, as stored) of the high-symmetry points along the path.
  kpt_labels : list of str
      Labels of the high-symmetry points, with ``G`` rendered as Γ.
  mu : float
      Chemical potential in eV.
  """
  E_skn: np.ndarray
  w_skan: np.ndarray
  kpt_label_idx: np.ndarray
  kpt_labels: list
  mu: float

  def weight(self, orb_list=None):
    """Weight of the orbitals in ``orb_list`` (all if ``None``) in each band, shape (ns, nkpts, nbnd)."""
    if orb_list is None:
      return self.w_skan.sum(axis=2)
    return self.w_skan[:, :, list(orb_list), :].sum(axis=2)


def wannier_weights_on_kpath(coqui_h5, iteration=-1, calc_type="mbpt"):
  """
  Diagonalize the interpolated Wannier Hamiltonian along the k-path and return the
  band energies together with the Wannier-orbital weight of every band.

  Parameters
  ----------
  coqui_h5 : str
      Path to the CoQuí HDF5 checkpoint file (``prefix.mbpt.h5``) on which
      ``band_interpolation`` has been run.
  iteration : int, optional
      SCF iteration to read. ``-1`` (default) selects the last one; ``0`` is the mean field.
  calc_type : str, optional
      ``"mbpt"`` (default) reads from the ``scf`` group; ``"dmft"`` from ``embed``.

  Returns
  -------
  WannierWeightsOnKpath

  Notes
  -----
  The weights partition each interpolated band among the Wannier orbitals. They equal
  the Wannier character of the Kohn–Sham bands only when the Wannier window is
  disentangled (``nImpOrbs == nOrbs_W``), where the interpolated bands reproduce the
  bands of the window. For an entangled window the interpolated bands are the
  eigenstates of the Hamiltonian projected onto the Wannier subspace, and the weights
  describe the orbital composition inside that subspace, not how much of a
  Kohn–Sham band lies in it.
  """
  if calc_type not in ["mbpt", "dmft"]:
    raise ValueError(f"Unknown calc_type = {calc_type}. \n"
                     "Acceptable options are 'mbpt' for many-body perturbation theory "
                     "and 'dmft' for dmft embedding results.")

  h5_grp = "scf" if calc_type == "mbpt" else "embed"
  with HDFArchive(coqui_h5, 'r') as ar:
    if iteration == -1:
      iteration = ar[f"{h5_grp}/final_iter"]
    if "qp_approx" in ar[f"{h5_grp}/iter{iteration}"]:
      qp_grp = ar[f"{h5_grp}/iter{iteration}/qp_approx"]
    else:
      qp_grp = ar[f"{h5_grp}/iter{iteration}"]
    if "H_skab" not in qp_grp["wannier_inter"]:
      raise KeyError(f"'{h5_grp}/iter{iteration}/.../wannier_inter/H_skab' not found in {coqui_h5}. "
                     "Run band_interpolation first.")
    mu = qp_grp["mu"] * Hartree_eV
    H_skab = qp_grp["wannier_inter/H_skab"]
    label_idx = qp_grp["wannier_inter/kpt_label_idx"]
    kpt_label_str = qp_grp["wannier_inter/kpt_labels"]

  kpt_labels = [letter if letter != 'G' else r'$\Gamma$' for letter in kpt_label_str]

  # eigh returns ascending eigenvalues; columns of U are the eigenvectors in the Wannier basis
  E_skn, U_skan = np.linalg.eigh(H_skab)
  E_skn = E_skn * Hartree_eV - mu
  w_skan = np.abs(U_skan) ** 2
  return WannierWeightsOnKpath(E_skn=E_skn, w_skan=w_skan, kpt_label_idx=label_idx,
                               kpt_labels=kpt_labels, mu=mu)


def wannier_weight_plot(ax, coqui_h5, iteration=-1, calc_type="mbpt", orb_list=None, spin=0,
                        cmap=None, vmin=0.0, vmax=1.0, linewidth=2.0, fontsize=16, verbal=True,
                        **kwargs):
  """
  Plot the Wannier-interpolated band structure with each band coloured by the weight of
  a chosen set of Wannier orbitals ("fat bands").

  Reads ``wannier_inter/H_skab`` written by ``band_interpolation``, diagonalizes it at
  every k-point of the path, and draws each band as line segments coloured by
  ``sum_{a in orb_list} |U_an(k)|^2`` on a scale from ``vmin`` to ``vmax`` (0 to 1 by default).

  Parameters
  ----------
  ax : matplotlib.axes.Axes
      Axes object to plot into.
  coqui_h5 : str
      Path to the CoQuí HDF5 checkpoint file (``prefix.mbpt.h5``).
  iteration : int, optional
      SCF iteration to read. ``-1`` (default) selects the last available iteration;
      ``0`` corresponds to the mean-field bands.
  calc_type : str, optional
      ``"mbpt"`` (default) reads from the ``scf`` group; ``"dmft"`` reads from ``embed``.
  orb_list : sequence of int, optional
      0-based indices of the Wannier orbitals whose weight is plotted, in the order of the
      projector file. ``None`` (default) sums over all orbitals, which is 1 everywhere and
      only useful as a check.
  spin : int, optional
      Spin channel to plot. Default ``0``.
  cmap : str or Colormap, optional
      Colormap for the weight. ``None`` (default) uses ``DEFAULT_WEIGHT_CMAP``, a diverging
      blue (0) -> gray (0.5) -> red (1) scale with saturated poles.
  vmin, vmax : float, optional
      Weights mapped to the two ends of the colormap. Defaults ``0.0`` and ``1.0``. Narrow the
      range (e.g. ``vmin=0.3, vmax=0.7``) to increase the contrast among strongly hybridized
      bands; weights outside the range saturate at the end colours.
  linewidth : float, optional
      Line width of the bands. Default ``2.0``.
  fontsize : int, optional
      Font size for axis labels and tick labels. Default ``16``.
  verbal : bool, optional
      If ``True`` (default), prints a summary of what was read.
  **kwargs
      Additional keyword arguments forwarded to ``matplotlib.collections.LineCollection``.

  Returns
  -------
  matplotlib.collections.LineCollection
      The plotted segments; pass it to ``fig.colorbar`` to add a weight scale.

  Notes
  -----
  See ``wannier_weights_on_kpath`` for what the weight means in an entangled window.

  Examples
  --------
  ::

      import matplotlib.pyplot as plt
      import coqui.post_proc.plot_utils as plot_utils

      fig, ax = plt.subplots()
      # 14 Wannier orbitals (5 Ni d + 9 O p); colour by the Ni d weight
      lc = plot_utils.wannier_weight_plot(ax, "lno.mbpt.h5", iteration=0, orb_list=range(5))
      fig.colorbar(lc, ax=ax, label="Ni d weight")
      ax.axhline(0, color="black", linewidth=1)
      plt.tight_layout()
      plt.savefig("lno_fat_bands.png")
  """
  res = wannier_weights_on_kpath(coqui_h5, iteration=iteration, calc_type=calc_type)
  ns, nkpts, nbnd = res.E_skn.shape
  nImpOrbs = res.w_skan.shape[2]
  if not 0 <= spin < ns:
    raise ValueError(f"spin = {spin} is out of range for ns = {ns}.")
  if orb_list is not None:
    orb_list = list(orb_list)
    if any(a < 0 or a >= nImpOrbs for a in orb_list):
      raise ValueError(f"orb_list = {orb_list} contains indices outside [0, {nImpOrbs}).")

  weight_kn = res.weight(orb_list)[spin]
  E_kn = res.E_skn[spin]

  if verbal:
    app_log(1, "  Plotting Wannier weights along the k-path")
    app_log(1, "  -----------------------------------------")
    app_log(1, "  CoQui h5           = {}".format(coqui_h5))
    if iteration == 0:
      app_log(1, "  Iteration          = {} (i.e. mean-field bands)".format(iteration))
    else:
      app_log(1, "  Iteration          = {}".format(iteration))
    app_log(1, "  Number of spins    = {} (plotting spin {})".format(ns, spin))
    app_log(1, "  Number of k-points = {}".format(nkpts))
    app_log(1, "  Number of bands    = {} (= number of Wannier orbitals)".format(nbnd))
    if orb_list is None:
      app_log(1, "  Orbitals           = all {} (weight is 1 everywhere; pass orb_list to "
                 "select a subset)".format(nImpOrbs))
    else:
      app_log(1, "  Orbitals           = {}".format(orb_list))
    app_log(1, f"  Chemical potential = {res.mu:.3f} (eV)\n")

  # one segment per (band, k -> k+1), coloured by the mean weight of its two end points
  x = np.arange(nkpts)
  segments, colors = [], []
  for n in range(nbnd):
    pts = np.column_stack([x, E_kn[:, n]])
    segments.extend(np.stack([pts[:-1], pts[1:]], axis=1))
    colors.extend(0.5 * (weight_kn[:-1, n] + weight_kn[1:, n]))
  if not vmin < vmax:
    raise ValueError(f"vmin = {vmin} must be smaller than vmax = {vmax}.")
  lc = LineCollection(segments, cmap=DEFAULT_WEIGHT_CMAP if cmap is None else cmap,
                      norm=Normalize(vmin=vmin, vmax=vmax), linewidth=linewidth, **kwargs)
  lc.set_array(np.asarray(colors))
  ax.add_collection(lc)

  ticks_pos = res.kpt_label_idx - 1
  ax.set_xticks(ticks_pos, res.kpt_labels)
  ax.tick_params(axis='both', which='major', labelsize=fontsize)
  ax.set_ylabel('$\\epsilon - \\mu$ (eV)', fontsize=fontsize)
  ax.set_xlim(0, nkpts - 1)
  ax.set_ylim(E_kn.min() - 0.5, E_kn.max() + 0.5)
  return lc
