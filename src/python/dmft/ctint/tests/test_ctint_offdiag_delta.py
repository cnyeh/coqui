"""
CT-INT backend with matrix-valued blocks (coqui.dmft.ctint.CtintSolver, offdiag_delta): h0 and
Delta off-diagonal inside a block, checked against the exactly solvable two-orbital impurity.
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

# Monte Carlo budget (n_cycles is the total over ranks; the chiAB measurement dominates the cost,
# so length_cycle is large and the chi grid short: ~30 s per production solve on 4 ranks, expansion
# order ~30, sign 1) and the tolerances. Measured once on 4 ranks at this budget over the base seeds
# 11 and 111: density off by 2.7e-3 and 2.0e-3; |G_00(iw0)| by 0.01 % and 0.08 %; the off-diagonal
# G_01(iw0) = 0.093 by 6e-5 and 1e-4; Sigma_dyn_00(iw0) by 3.6e-4 and 1.8e-3; Sigma_dyn_01(iw0) =
# 0.0074 by 3.7e-4 and 1.2e-4; Sigma_infty_00 = 0.68 by 8.5e-4 and 6.4e-4; Sigma_infty_01 = 0.023 by
# 1.5e-4 and 1.9e-4; W within 3.1 % and 1.9 % of ED at iv0, iv1; Pi within 8.4 % and 5.1 % (the
# plateau bias of the connected chiAB, as in test_ctint_solver). triqs_ctint is not reproducible at
# a fixed seed, so every run is a fresh sample: the tolerances are three times the larger of the
# two samples, and the test passes at 1 and 2 ranks with them. The interaction is mild
# (U = 1, U' = 0.5, J = 0, so the tensor is density-density and the solver's reduction of it is
# exact) and the boson coupling g = 0.3: at g = 0.5 the Hartree-Fock and polaron terms of
# Sigma_infty_00 cancel exactly for these U, U', which would make its check vacuous.
_SOLVER = {"n_cycles": 20000, "n_warmup_cycles": 500, "length_cycle": 200, "n_tau_chi2": 201,
           "random_seed": 11}
_TOL = {"density": 0.01, "g00_rel": 0.003, "g01": 5e-4, "sigma00": 6e-3, "sigma01": 1.5e-3,
        "sigma_infty00": 3e-3, "sigma_infty01": 8e-4, "w_rel": 0.1, "pi_rel": 0.3}


def _i0(wn):
  """Position of the lowest positive frequency."""
  return int(np.argmin(np.where(wn > 0, wn, np.inf)))


def test_ctint_offdiag_delta():
  """CtintSolver.solve with one 2x2 block per spin reproduces the exact solution of an impurity whose
  h0 and Delta are both off-diagonal.

  Setup: the two-orbital impurity (beta 20, U 1, U' 0.5, J 0, eps -0.5, V 1, g 0.3, one retarded
  pole, ~2.1 electrons) with the inter-orbital hopping t = 0.15 and the cross hybridization
  V_01 = 0.15, in the layout GF_STRUCT_BLOCKS [("up_0", 2), ("down_0", 2)] with degenerate_blk
  [[0, 1]]; the budget _SOLVER; a CtintSolver for comm_size = the MPI size.

  Cases:
    1. Production solve: validate(problem); the density, the 2x2 matrices G(iw0), Sigma_dyn(iw0)
       and Sigma_infty of the up block within _TOL of the ED colour matrices (see the header:
       G_01, Sigma_dyn_01 and Sigma_infty_01 are the off-diagonal checks, a transposed or dropped
       element would miss by their full size, 0.09, 0.007 and 0.023); the down block equal to the
       up block (degenerate_blk); G_10 and G_01, measured independently, equal within 1e-3 (the
       Hamiltonian is real, so they agree to the Monte Carlo noise, 3e-5 measured);
       W and Pi at iv0 and iv1 within _TOL of ED (Pi carries the plateau bias of the connected
       correlator).
  """
  params = KanamoriParams(U=1.0, Uprime=0.5, J=0.0, eps=-0.5, g=0.3, t=0.15, V_01=0.15)
  iaft = make_iaft(params, wmax=20.0, eps=1e-10)
  ref = KanamoriED(params).reference(iaft, n_bose=2)
  problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=((0, 1),),
                         gf_struct=GF_STRUCT_BLOCKS)
  i0 = _i0(ref["wn_f"])
  up = [c for c in range(4) if COLOR_SPIN[c] == 0]
  g_ex = ref["G_iw_full"][i0][np.ix_(up, up)]
  sigma_ex = ref["Sigma_dyn_iw_full"][i0][np.ix_(up, up)]
  sigma_infty_ex = ref["Sigma_infty_full"][np.ix_(up, up)]
  assert abs(g_ex[0, 1]) > 0.05 and abs(sigma_ex[0, 1]) > 5e-3 and abs(sigma_infty_ex[0, 1]) > 0.02

  result = CtintSolver(dict(_SOLVER), comm_size=mpi.size).solve(problem, occupations_only=False,
                                                              chkpt=None)
  result.validate(problem)
  assert abs(result.density - ref["occupations"].sum()) < _TOL["density"]
  g = result.G_iw_data[0][i0]
  assert abs(abs(g[0, 0]) - abs(g_ex[0, 0])) / abs(g_ex[0, 0]) < _TOL["g00_rel"]
  assert abs(g[0, 1] - g_ex[0, 1]) < _TOL["g01"]
  assert abs(g[1, 0] - g[0, 1]) < 1e-3
  sigma = result.Sigma_iw_data[0][i0]
  assert abs(sigma[0, 0] - sigma_ex[0, 0]) < _TOL["sigma00"]
  assert abs(sigma[0, 1] - sigma_ex[0, 1]) < _TOL["sigma01"]
  sigma_infty = result.Sigma_infty[0]
  assert abs(sigma_infty[0, 0] - sigma_infty_ex[0, 0]) < _TOL["sigma_infty00"]
  assert abs(sigma_infty[0, 1] - sigma_infty_ex[0, 1]) < _TOL["sigma_infty01"]
  assert np.allclose(result.G_iw_data[1], result.G_iw_data[0])
  assert np.allclose(result.Sigma_iw_data[1], result.Sigma_iw_data[0])
  assert np.allclose(result.Sigma_infty[1], result.Sigma_infty[0])
  for key, exact, rel in (("W_iw_data", ref["W_iw"], _TOL["w_rel"]), ("Pi_iw_data", ref["Pi_iw"], _TOL["pi_rel"])):
    got = getattr(result, key)[0]
    for index in (0, 1):
      assert np.abs(got[index] - exact[index]).max() / np.abs(exact[index]).max() < rel, (key, index)
