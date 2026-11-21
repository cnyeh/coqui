"""
CT-INT backend with occupation-centred alpha shifts (coqui.dmft.ctint.CtintSolver, alpha_mode "auto"),
checked against the exactly solvable two-orbital impurity with almost empty orbitals.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("triqs_ctint", reason="triqs_ctint is required for this backend")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from triqs.utility import mpi
from coqui.dmft.ctint import CtintSolver
from coqui.dmft.tests.kanamori_holstein_dimer import (KanamoriParams, KanamoriED, make_iaft, make_problem,
                                                      GF_STRUCT_BLOCKS, COLOR_SPIN)

# Monte Carlo budget (n_cycles is the total over ranks) and the tolerances. Measured once on 4 ranks
# at this budget over the base seeds 11 and 111, both modes: density off by up to 1.6e-3;
# |G_00(iw0)| by up to 0.16 %; G_01(iw0) = 0.20 by up to 8e-4; Sigma_dyn_00(iw0) by up to 2.9e-3;
# Sigma_dyn_01(iw0) = 0.0053 by up to 3.0e-3; Sigma_infty_00 = 0.049 by up to 5e-4; Sigma_infty_01 =
# 0.014 by up to 2.1e-4; W within 0.9 % at iv1 but 12 % at iv0 and Pi 50 % at iv0 (the plateau bias
# of the connected chiAB on an almost empty impurity; W(iv0) is bounded, Pi checked for shape).
# triqs_ctint is not reproducible at a fixed seed (two runs of the same seed on 2 ranks differ at
# the level of the seed spread), so every run is a fresh sample: the tolerances are three times the
# largest deviation over the samples. The efficiency difference is the point of the mode: at the
# occupation 0.032 the signed shifts give expansion order 17.5 and ~18 s per solve on 4 ranks, the
# occupation-centred ones order 0.17, sign 0.98 and ~5 s.
_SOLVER = {"n_cycles": 20000, "n_warmup_cycles": 500, "length_cycle": 200, "n_tau_chi2": 201,
           "random_seed": 11}
_TOL = {"density": 6e-3, "g00_rel": 6e-3, "g01": 3e-3, "sigma00": 1.5e-2, "sigma01": 1e-2,
        "sigma_infty00": 2e-3, "sigma_infty01": 8e-4, "w1_rel": 0.05, "w0_rel": 0.3}


def _i0(wn):
  """Position of the lowest positive frequency."""
  return int(np.argmin(np.where(wn > 0, wn, np.inf)))


def _compare(result, ref, i0):
  """The agreement of one production result with the ED reference on the up block."""
  up = [c for c in range(4) if COLOR_SPIN[c] == 0]
  g_ex = ref["G_iw_full"][i0][np.ix_(up, up)]
  sigma_ex = ref["Sigma_dyn_iw_full"][i0][np.ix_(up, up)]
  sigma_infty_ex = ref["Sigma_infty_full"][np.ix_(up, up)]
  assert abs(result.density - ref["occupations"].sum()) < _TOL["density"]
  g = result.G_iw_data[0][i0]
  assert abs(abs(g[0, 0]) - abs(g_ex[0, 0])) / abs(g_ex[0, 0]) < _TOL["g00_rel"]
  assert abs(g[0, 1] - g_ex[0, 1]) < _TOL["g01"]
  sigma = result.Sigma_iw_data[0][i0]
  assert abs(sigma[0, 0] - sigma_ex[0, 0]) < _TOL["sigma00"]
  assert abs(sigma[0, 1] - sigma_ex[0, 1]) < _TOL["sigma01"]
  sigma_infty = result.Sigma_infty[0]
  assert abs(sigma_infty[0, 0] - sigma_infty_ex[0, 0]) < _TOL["sigma_infty00"]
  assert abs(sigma_infty[0, 1] - sigma_infty_ex[0, 1]) < _TOL["sigma_infty01"]
  w = result.W_iw_data[0]
  assert np.abs(w[1] - ref["W_iw"][1]).max() / np.abs(ref["W_iw"][1]).max() < _TOL["w1_rel"]
  assert np.abs(w[0] - ref["W_iw"][0]).max() / np.abs(ref["W_iw"][0]).max() < _TOL["w0_rel"]
  assert result.Pi_iw_data[0].shape == w.shape and np.isfinite(result.Pi_iw_data[0]).all()


def test_ctint_alpha_auto(monkeypatch):
  """CtintSolver.solve with alpha_mode "auto" centres the shifts of an almost empty orbital on its
  lattice occupation, reproduces the exact solution at a far lower expansion order than the signed
  shifts, and warns when the lattice occupations it was centred on are wrong.

  Setup: the two-orbital impurity (beta 20, U 1, U' 0.5, J 0, g 0.3, V 1, t 0.15, V_01 0.15) with
  eps = 3.5, so every colour holds 0.032 electrons (inside the default alpha_edge 0.05), in the
  layout GF_STRUCT_BLOCKS with degenerate_blk [[0, 1]]; the budget _SOLVER; app_log replaced by a
  recorder for the warning check.

  Cases (2 and 3 compare their expansion order with 1):
    1. alpha_mode "auto": validate(problem); every quantity within _TOL of ED (_compare); the
       diagnostics hold alpha_occupation_centered all 1, alpha_centers equal to the lattice
       occupations of the problem (0.032 for every colour) and alpha_deltas equal to alpha_bracket
       (0.02), alpha_check_ok 1, average_order below 1 and average_sign above 0.95; no warning
       logged.
    2. alpha_mode "signed" on the same problem: the same agreement with ED (the alpha is
       compensated exactly, so the shifts move the cost only) at an average_order above 10 and
       no alpha_check_ok in the diagnostics.
    3. alpha_mode "auto" with the problem's occupations_sab set to 0 (a lattice occupation 0.032
       off the impurity's, beyond the bracket 0.02): the result is still within _TOL of ED, the
       bracket check fails (alpha_check_ok 0, alpha_max_offset above 0.02) and one [WARNING]
       naming the brackets is logged.
  """
  params = KanamoriParams(U=1.0, Uprime=0.5, J=0.0, g=0.3, eps=3.5, t=0.15, V_01=0.15)
  iaft = make_iaft(params, wmax=20.0, eps=1e-10)
  ref = KanamoriED(params).reference(iaft, n_bose=2)
  problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=((0, 1),),
                         gf_struct=GF_STRUCT_BLOCKS)
  i0 = _i0(ref["wn_f"])
  assert np.all(ref["occupations"] < 0.05)
  logged = []
  monkeypatch.setattr("coqui.dmft.ctint.ctint_solver.app_log", lambda level, msg: logged.append((level, msg)))

  auto = CtintSolver(dict(_SOLVER, alpha_mode="auto"), comm_size=mpi.size).solve(
    problem, occupations_only=False, chkpt=None)
  auto.validate(problem)
  _compare(auto, ref, i0)
  d = auto.diagnostics
  assert d["alpha_occupation_centered"].tolist() == [1, 1, 1, 1]
  assert np.allclose(d["alpha_centers"], ref["occupations"]) and np.allclose(d["alpha_deltas"], 0.02)
  assert d["alpha_check_ok"] == 1 and d["average_order"] < 1.0 and d["average_sign"] > 0.95
  assert not any("[WARNING]" in msg for _, msg in logged)

  signed = CtintSolver(dict(_SOLVER, alpha_mode="signed"), comm_size=mpi.size).solve(
    problem, occupations_only=False, chkpt=None)
  _compare(signed, ref, i0)
  assert signed.diagnostics["average_order"] > 10.0 and "alpha_check_ok" not in signed.diagnostics

  wrong = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=((0, 1),),
                       gf_struct=GF_STRUCT_BLOCKS)
  wrong.occupations_sab[:] = 0.0
  logged.clear()
  off = CtintSolver(dict(_SOLVER, alpha_mode="auto"), comm_size=mpi.size).solve(
    wrong, occupations_only=False, chkpt=None)
  _compare(off, ref, i0)
  assert off.diagnostics["alpha_check_ok"] == 0 and off.diagnostics["alpha_max_offset"] > 0.02
  warnings = [msg for _, msg in logged if "[WARNING]" in msg]
  assert len(warnings) == 1 and "bracket" in warnings[0]
