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
"""

from mpi4py import MPI
import os
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from h5 import HDFArchive
from scipy.constants import physical_constants
Hartree_eV = physical_constants["Hartree energy in eV"][0]

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.mean_field.tests.test_qe import construct_qe_mf
from coqui.post_proc import band_interpolation
from coqui.post_proc.plot_utils import wannier_weight_plot, wannier_weights_on_kpath


@pytest.fixture(scope="module")
def svo_kpath_checkpoint(mpi):
  """band_interpolation in mean-field-only mode on the SrVO3 t2g test data."""
  mf = construct_qe_mf(mpi, "qe_svo222_sym")
  wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"
  prefix = "svo_wannier_weight"
  band_interpolation(mf, {
    "outdir": "./", "prefix": prefix, "wannier_file": wan_h5,
    "bands_num_npoints": 10,
    "kpath": "G 0.0 0.0 0.0 X 0.0 0.5 0.0 M 0.5 0.5 0.0 G 0.0 0.0 0.0",
  })
  mpi.barrier()
  yield f"./{prefix}.mbpt.h5"
  mpi.barrier()
  if mpi.root() and os.path.exists(f"./{prefix}.mbpt.h5"):
    os.remove(f"./{prefix}.mbpt.h5")


def test_wannier_weights_on_kpath_partition_unity(svo_kpath_checkpoint):
  res = wannier_weights_on_kpath(svo_kpath_checkpoint, iteration=0)

  E_skn, w_skan = res.E_skn, res.w_skan
  ns, nkpts, nbnd = E_skn.shape
  assert w_skan.shape == (ns, nkpts, nbnd, nbnd)   # 3 t2g orbitals, 3 interpolated bands
  assert np.all(w_skan >= -1e-12) and np.all(w_skan <= 1 + 1e-12)
  # eigenvectors are unitary: each band's weight sums to one over the orbitals,
  # and each orbital's weight sums to one over the bands
  np.testing.assert_allclose(w_skan.sum(axis=2), 1.0, atol=1e-10)
  np.testing.assert_allclose(w_skan.sum(axis=3), 1.0, atol=1e-10)
  # bands come out sorted, matching what band_interpolation stored
  assert np.all(np.diff(E_skn, axis=2) >= -1e-12)
  with HDFArchive(svo_kpath_checkpoint, "r") as ar:
    E_ref = ar["scf/iter0/qp_approx/wannier_inter/E_ska"] \
      if "qp_approx" in ar["scf/iter0"] else ar["scf/iter0/wannier_inter/E_ska"]
  np.testing.assert_allclose(E_skn, np.sort(E_ref, axis=2) * Hartree_eV - res.mu, atol=1e-8)
  assert len(res.kpt_labels) == len(res.kpt_label_idx) == 4


def test_wannier_weights_orb_list_selects_orbitals(svo_kpath_checkpoint):
  res = wannier_weights_on_kpath(svo_kpath_checkpoint, iteration=0)
  w_one = res.weight(orb_list=[0])
  w_all = res.weight(orb_list=None)
  assert w_one.shape == res.E_skn.shape
  np.testing.assert_allclose(w_all, 1.0, atol=1e-10)
  np.testing.assert_allclose(w_one, res.w_skan[:, :, 0, :], atol=1e-14)
  # at Gamma the three t2g bands are degenerate and the single orbital cannot own all of them
  assert w_one[0, 0, :].sum() == pytest.approx(1.0, abs=1e-10)


def test_wannier_weight_plot_draws_colored_segments(svo_kpath_checkpoint):
  fig, ax = plt.subplots()
  lc = wannier_weight_plot(ax, svo_kpath_checkpoint, iteration=0, orb_list=[0], verbal=False)
  assert isinstance(lc, LineCollection)

  res = wannier_weights_on_kpath(svo_kpath_checkpoint, iteration=0)
  ns, nkpts, nbnd = res.E_skn.shape
  assert len(lc.get_segments()) == nbnd * (nkpts - 1)
  colors = lc.get_array()
  assert colors.shape == (nbnd * (nkpts - 1),)
  assert lc.norm.vmin == 0.0 and lc.norm.vmax == 1.0
  assert len(ax.get_xticks()) == 4
  plt.close(fig)


def test_wannier_weight_plot_vmin_vmax_stretch_the_color_scale(svo_kpath_checkpoint):
  fig, ax = plt.subplots()
  lc = wannier_weight_plot(ax, svo_kpath_checkpoint, iteration=0, orb_list=[0],
                           vmin=0.2, vmax=0.8, verbal=False)
  assert lc.norm.vmin == 0.2 and lc.norm.vmax == 0.8
  # default map has saturated poles and a visible mid gray: the three anchors differ
  rgba = lc.cmap(np.array([0.0, 0.5, 1.0]))
  assert not np.allclose(rgba[0], rgba[1]) and not np.allclose(rgba[1], rgba[2])
  assert rgba[1][:3].max() < 0.75   # mid point is not near-white
  plt.close(fig)
