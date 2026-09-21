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
import shutil
import pytest
import numpy as np
import h5

import coqui
from coqui.utils.tests.test_coqui_env import mpi
from coqui.mean_field.tests.test_qe import construct_qe_mf


def _make_checkpoint(mpi, prefix):
  mf = construct_qe_mf(mpi, "qe_lih222_sym")
  eri_params = {
    "storage": "incore",
    "nIpts": mf.nbnd() * 10,
    "thresh": 1e-10,
    "ecut": mf.ecutrho(),
    "chol_block_size": 1,
    "init": True,
  }
  thc = coqui.make_thc_coulomb(mf, eri_params)
  gw_params = {
    "restart": False,
    "output": prefix,
    "niter": 2,
    "div_treatment": "gygi",
    "beta": 100,
    "iaft": {"prec": "medium", "basis": "dlr"},
    "iter_alg": {"alg": "damping", "mixing": 0.7},
  }
  coqui.run_gw(gw_params, h_int=thc)
  mpi.barrier()
  return mf, thc


def test_dielectric_function_extrapolate(mpi):
  prefix = "diel_pp"
  _, thc = _make_checkpoint(mpi, prefix)

  res = coqui.post_proc.dielectric_function(thc, {
    "prefix": prefix, "outdir": "./", "iteration": 1,
    "pi_regularization": "extrapolate", "div_treatment": "gygi",
  })
  mpi.barrier()

  assert res["iteration"] == 1
  assert res["grp_name"] == "scf"
  assert res["pi_regularization"] == "extrapolate"
  nw, nq = res["eps_inv_head_wq"].shape
  assert nq == thc.nqpts_ibz()
  assert res["eps_inv_head_w"].shape == (nw,)
  assert res["pi_head_wq"].shape == (nw, nq)
  assert res["delta_C_w"].shape == (nw,)
  assert np.all(res["delta_C_w"].imag == 0.0)       # stored real by construction
  assert res["nu"].shape == (nw,) and res["nu"][0] == 0.0 and np.all(np.diff(res["nu"]) > 0)

  # The returned arrays are exactly what the checkpoint holds.
  with h5.HDFArchive(os.path.join("./", prefix + ".mbpt.h5"), "r") as ar:
    g = ar["scf"]["iter1"]["dielectric"]["extrapolate"]
    assert np.array_equal(np.asarray(g["eps_inv_head_wq"]), res["eps_inv_head_wq"])
    assert np.array_equal(np.asarray(g["delta_C_w"]), res["delta_C_w"])
    # In-loop data untouched and still present.
    assert "eps_inv_head_wq" in ar["scf"]["iter2"]

  if mpi.root():
    os.remove(os.path.join("./", prefix + ".mbpt.h5"))
  mpi.barrier()


def test_dielectric_function_none_returns_no_dc(mpi):
  prefix = "diel_pp_none"
  _, thc = _make_checkpoint(mpi, prefix)
  res = coqui.post_proc.dielectric_function(thc, {"prefix": prefix, "iteration": 1})
  mpi.barrier()
  assert res["pi_regularization"] == "none"
  assert res["delta_C_w"] is None

  # iteration = -1 (the default) must resolve to scf/final_iter, which is 2 after a
  # 2-iteration run; the wrapper re-derives this independently of the C++ driver.
  with h5.HDFArchive(os.path.join("./", prefix + ".mbpt.h5"), "r") as ar:
    final_iter = int(ar["scf"]["final_iter"])
  assert final_iter == 2
  res_default = coqui.post_proc.dielectric_function(thc, {"prefix": prefix})
  mpi.barrier()
  assert res_default["iteration"] == final_iter

  # A checkpoint written before CoQuí stored the Matsubara mesh: everything but 'nu' must
  # still come back, with a warning. Copied at the file level, then stripped of iwn_mesh.
  old_prefix = prefix + "_no_mesh"
  if mpi.root():
    shutil.copy(os.path.join("./", prefix + ".mbpt.h5"),
                os.path.join("./", old_prefix + ".mbpt.h5"))
    with h5.HDFArchive(os.path.join("./", old_prefix + ".mbpt.h5"), "a") as ar:
      del ar["imaginary_fourier_transform"]["iwn_mesh"]
  mpi.barrier()

  with pytest.warns(UserWarning, match="iwn_mesh"):
    res_old = coqui.post_proc.dielectric_function(thc, {"prefix": old_prefix, "iteration": 1})
  mpi.barrier()
  assert res_old["nu"] is None
  assert res_old["eps_inv_head_wq"].shape == res["eps_inv_head_wq"].shape
  assert np.array_equal(res_old["eps_inv_head_wq"], res["eps_inv_head_wq"])

  if mpi.root():
    os.remove(os.path.join("./", prefix + ".mbpt.h5"))
    os.remove(os.path.join("./", old_prefix + ".mbpt.h5"))
  mpi.barrier()


def test_dielectric_function_rejects_bad_input(mpi):
  # Validation happens before any C++ call, so no checkpoint or ERI is needed; a
  # placeholder object stands in for h_int.
  from coqui.post_proc import dielectric_function
  with pytest.raises(ValueError, match="pi_regularization"):
    dielectric_function(object(), {"prefix": "x", "pi_regularization": "static"})
  with pytest.raises(ValueError, match="grp_name"):
    dielectric_function(object(), {"prefix": "x", "grp_name": "downfold_2e"})
  # screen_type itself is no longer restricted, but a recipe that needs the bosonic
  # projector must say where it comes from.
  with pytest.raises(ValueError, match="wannier_file"):
    dielectric_function(object(), {"prefix": "x", "screen_type": "gw_edmft"})
  with pytest.raises(ValueError, match="wannier_file"):
    dielectric_function(object(), {"prefix": "x", "screen_type": "crpa"})
  with pytest.raises(ValueError, match="local_polarizabilities"):
    dielectric_function(object(), {"prefix": "x"}, local_polarizabilities={"imp": None})
  with pytest.raises(ValueError, match="beta"):
    dielectric_function(object(), {"prefix": "x", "beta": 100})
  with pytest.raises(ValueError, match="iaft"):
    dielectric_function(object(), {"prefix": "x", "iaft": {"prec": "high"}})
  with pytest.raises(ValueError, match="prefix"):
    dielectric_function(object(), {})


def _const_pi_local(nw_half, nImpOrbs, value):
  """A density-density-shaped constant Pi_abcd(i nu) = value * delta_ab delta_cd."""
  pi = np.zeros((nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs), dtype=complex)
  for a in range(nImpOrbs):
    for b in range(nImpOrbs):
      pi[:, a, a, b, b] = value
  return pi


def test_dielectric_function_gw_edmft(mpi):
  # screen_type = "gw_edmft" needs the bosonic projector, which can be named in params
  # ("wannier_file") or passed as arrays ("projector_info"), and local polarizabilities,
  # which can be passed as arrays, read from the checkpoint, or be absent.
  prefix = "diel_pp_edmft"
  mf, thc = _make_checkpoint(mpi, prefix)
  filename = os.path.join("./", prefix + ".mbpt.h5")
  wannier_file = mf.outdir() + "/lih_wan.h5"
  proj_info = coqui.read_proj_info(wannier_file)
  keys = ("eps_inv_head_wq", "eps_inv_head_w", "pi_head_wq")

  base = {"prefix": prefix, "outdir": "./", "iteration": 1, "div_treatment": "gygi"}
  res_rpa = coqui.post_proc.dielectric_function(thc, base)
  mpi.barrier()

  edmft = dict(base, screen_type="gw_edmft")
  edmft_h5 = dict(edmft, wannier_file=wannier_file)

  # --- 1. no local polarizabilities anywhere: zero correction, i.e. plain RPA. This says
  # nothing about the projector (eval_Pi_qdep never dereferences it on this branch), only
  # that the fallback is harmless.
  res_h5 = coqui.post_proc.dielectric_function(thc, edmft_h5)
  mpi.barrier()
  res_arr = coqui.post_proc.dielectric_function(thc, edmft, projector_info=proj_info)
  mpi.barrier()
  for key in keys:
    assert np.array_equal(res_h5[key], res_rpa[key]), key
    assert np.array_equal(res_arr[key], res_rpa[key]), key

  nw_half = res_rpa["eps_inv_head_w"].shape[0]
  nImpOrbs = proj_info["proj_mat"].shape[3]
  zeros = _const_pi_local(nw_half, nImpOrbs, 0.0)

  # --- 2. explicit zeros must go through set_local_polarizabilities and upfold_pi_local
  # and come out unchanged.
  res_zero = coqui.post_proc.dielectric_function(
    thc, edmft, projector_info=proj_info,
    local_polarizabilities={"imp": zeros, "dc": zeros.copy()})
  mpi.barrier()
  for key in keys:
    assert np.array_equal(res_zero[key], res_rpa[key]), key

  # A non-edmft screen_type never uses them: they must be discarded (with a warning) and
  # leave the plain RPA result untouched, projector or no projector.
  res_discard = coqui.post_proc.dielectric_function(
    thc, base, local_polarizabilities={"imp": zeros, "dc": zeros.copy()})
  mpi.barrier()
  for key in keys:
    assert np.array_equal(res_discard[key], res_rpa[key]), key

  # --- 3. a NONZERO impurity polarizability. This is the case that discriminates: the
  # correction must reach W, and only here does agreement between the two projector paths
  # actually compare the two projectors.
  imp = _const_pi_local(nw_half, nImpOrbs, -0.2)
  lp = {"imp": imp, "dc": zeros.copy()}
  res_nz = coqui.post_proc.dielectric_function(
    thc, edmft, projector_info=proj_info, local_polarizabilities=lp)
  mpi.barrier()
  res_nz_h5 = coqui.post_proc.dielectric_function(
    thc, edmft_h5, local_polarizabilities=lp)
  mpi.barrier()

  d_eps = np.abs(res_nz["eps_inv_head_wq"] - res_rpa["eps_inv_head_wq"])
  d_pi = np.abs(res_nz["pi_head_wq"] - res_rpa["pi_head_wq"])
  print(f"\n[gw_edmft] max |d eps_inv_head_wq| = {d_eps.max():.4e}, "
        f"max |d pi_head_wq| = {d_pi.max():.4e}")
  assert not np.allclose(res_nz["eps_inv_head_wq"], res_rpa["eps_inv_head_wq"])
  # Far above round-off, so the test is not sitting on numerical noise.
  assert d_eps.max() > 1e-4
  assert d_pi.max() > 1e-4
  # Still a physical screening: eps_M = 1/(1+head) must stay positive.
  assert np.all(1.0 + res_nz["eps_inv_head_wq"].real > 0.0)
  # The two projector paths, on a correction that actually moves the answer.
  for key in keys:
    assert np.array_equal(res_nz_h5[key], res_nz[key]), key

  # --- 4. the same correction read from the checkpoint instead of passed in. This is the
  # production workflow: the impurity solver writes downfold_2e/iter{M}/Pi_{imp,dc}_wabcd,
  # post-processing picks them up. read_pi_local reads into a pre-sized nda array, so a
  # geometry mismatch aborts rather than failing gracefully -- worth pinning.
  if mpi.root():
    with h5.HDFArchive(filename, "a") as ar:
      ar.create_group("downfold_2e")
      df = ar["downfold_2e"]
      df["final_iter"] = 0
      df.create_group("iter0")
      it = df["iter0"]
      it["Pi_imp_wabcd"] = imp
      it["Pi_dc_wabcd"] = zeros
  mpi.barrier()

  res_ckpt = coqui.post_proc.dielectric_function(thc, edmft, projector_info=proj_info)
  mpi.barrier()
  for key in keys:
    assert np.array_equal(res_ckpt[key], res_nz[key]), key

  # The provenance string of the last run is what the checkpoint keeps.
  with h5.HDFArchive(filename, "r") as ar:
    assert ar["scf"]["iter1"]["dielectric"]["none"]["screen_type"] == "gw_edmft"

  if mpi.root():
    os.remove(filename)
  mpi.barrier()
