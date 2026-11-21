"""
CT-INT's own helpers (coqui.dmft.ctint.ctint_utils): the D0 channel order, the signed alpha tensor and
the M(tau) route to G.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")

from triqs.gfs import Gf, BlockGf, MeshImTime, MeshDLRImFreq
from triqs.operators import n
from coqui.dmft.ctint.ctint_utils import (d0_channel_colors, signed_alpha, g_from_m_tau, resolve_tau_grids,
                                          auto_alpha, alpha_bracket_check, problem_occupations)
from coqui.dmft.solver_api import ImpurityProblem
from coqui.utils.imag_axes_ft import IAFT

_GF2 = [("up_0", 1), ("up_1", 1), ("down_0", 1), ("down_1", 1)]     # two orbitals, size-1 blocks


def test_d0_channel_colors():
  """d0_channel_colors lists the D0 vertices in triqs_ctint's label order, blocks then positions,
  row-major, and refuses blocks of unequal size.

  Setup: _GF2; a gf_struct mixing sizes 1 and 2.

  Cases:
    1. _GF2 gives the 16 colour pairs (b1, b2), b1 outer.
    2. Size-2 blocks [("up", 2), ("down", 2)]: (block b1, position a) -> colour 2 b1 + a, so the pairs
       are (0,0), (0,1), (1,0), (1,1), (0,2), ... in (b1, b2, a, a') order.
    3. Unequal sizes raise ValueError naming "equal size".
  """
  assert d0_channel_colors(_GF2) == [(b1, b2) for b1 in range(4) for b2 in range(4)]
  pairs = d0_channel_colors([("up", 2), ("down", 2)])
  assert pairs == [(2 * b1 + a, 2 * b2 + ap) for b1 in range(2) for b2 in range(2) for a in range(2) for ap in range(2)]
  with pytest.raises(ValueError, match="equal size"):
    d0_channel_colors([("up_0", 1), ("up_1", 2), ("down_0", 1), ("down_1", 2)])


def test_signed_alpha():
  """signed_alpha places one shift pair per vertex in triqs_ctint's layout, opposite directions for a
  repulsive coupling, the same for an attractive one.

  Setup: _GF2; h_int = 2 n_up0 n_dn0 + 3 n_up1 n_dn1 - 1 n_up0 n_dn1 (two repulsive and one attractive
  density-density term); D(tau) channels: all 16 colour pairs of _GF2, each a constant -0.5
  (attractive, as a retarded GW+EDMFT coupling is).

  Cases:
    1. center 0.5, delta 0.51: shape (3 + 16, 2, 2, 2); the cross entries [., 0, 1] and [., 1, 0] are
       zero; for a repulsive term alpha[l, 0, 0, s] = 0.5 - sgn_s 0.51 and alpha[l, 1, 1, s] =
       0.5 + sgn_s 0.51 (opposite directions, sgn_s = +1, -1 for s = 0, 1); for the attractive term
       and for an attractive D0 channel both entries move in the same direction, 0.5 + sgn_s 0.51;
       the s-average of every entry is the centre 0.5 (what the library's compensation reads off).
    2. A D(tau) channel that changes sign, [0.9, -0.5, -0.5, -0.5, -0.5] (a non-causal Weiss field):
       "dominant" means the largest |D(tau)|, here positive, so the channel is treated as repulsive
       (opposite directions) although its mean is negative; a warning, not an error. A channel that
       is numerically zero counts as attractive.
    3. A non-density-density term (c+_up0 c+_dn0 c_dn1 c_up1) raises ValueError.
    4. Per-colour shifts, centres [0.1, 0.5, 0.9, 0.5] and deltas [0.02, 0.51, 0.02, 0.51] with
       gf_struct: every vertex operator takes the centre and delta of its own colour (pair 0 is
       the first creation operator of the normal-ordered monomial, opposite direction for a
       repulsive coefficient; the D0 channel of colours (0, 2) at 0.1 and 0.9 +- 0.02), the
       s-average of every entry is its colour's centre; per-colour arrays without gf_struct raise
       ValueError.
  """
  pairs = d0_channel_colors(_GF2)
  h_int = (2.0 * n("up_0", 0) * n("down_0", 0) + 3.0 * n("up_1", 0) * n("down_1", 0)
           - 1.0 * n("up_0", 0) * n("down_1", 0))
  channels = [np.full(5, -0.5) for _ in pairs]
  alpha = signed_alpha(h_int=h_int, d0_channels=channels, center=0.5, delta=0.51)
  assert alpha.shape == (3 + 16, 2, 2, 2)
  assert np.all(alpha[:, 0, 1, :] == 0.0) and np.all(alpha[:, 1, 0, :] == 0.0)
  coefs = [float(np.real(coef)) for _, coef in h_int]
  repulsive = [l for l, coef in enumerate(coefs) if coef > 0]
  attractive = [l for l, coef in enumerate(coefs) if coef < 0]
  assert len(repulsive) == 2 and len(attractive) == 1
  for s, sgn in enumerate((1.0, -1.0)):
    assert np.allclose(alpha[repulsive, 0, 0, s], 0.5 - sgn * 0.51)       # repulsive: opposite directions
    assert np.allclose(alpha[repulsive, 1, 1, s], 0.5 + sgn * 0.51)
    assert np.allclose(alpha[attractive, 0, 0, s], 0.5 + sgn * 0.51)      # attractive: same direction
    assert np.allclose(alpha[attractive, 1, 1, s], 0.5 + sgn * 0.51)
    assert np.allclose(alpha[3:, 0, 0, s], 0.5 + sgn * 0.51)              # attractive D0: same direction
    assert np.allclose(alpha[3:, 1, 1, s], 0.5 + sgn * 0.51)
  assert np.allclose(alpha.mean(axis=-1)[:, 0, 0], 0.5) and np.allclose(alpha.mean(axis=-1)[:, 1, 1], 0.5)

  mixed = list(channels)
  mixed[3] = np.array([0.9, -0.5, -0.5, -0.5, -0.5])
  alpha_mixed = signed_alpha(h_int=h_int, d0_channels=mixed, center=0.5, delta=0.51)
  assert np.allclose(alpha_mixed[3 + 3, 0, 0, 0], 0.5 - 0.51) and np.allclose(alpha_mixed[3 + 3, 1, 1, 0], 0.5 + 0.51)
  zero = list(channels)
  zero[0] = np.zeros(5)
  alpha_zero = signed_alpha(h_int=h_int, d0_channels=zero, center=0.5, delta=0.51)
  assert np.allclose(alpha_zero[3, 0, 0, 0], 0.5 + 0.51)           # zero channel: attractive direction

  from triqs.operators import c_dag, c
  with pytest.raises(ValueError, match="density-density"):
    signed_alpha(h_int=c_dag("up_0", 0) * c_dag("down_0", 0) * c("down_1", 0) * c("up_1", 0),
                 d0_channels=channels, center=0.5, delta=0.51)

  centers = np.array([0.1, 0.5, 0.9, 0.5])
  deltas = np.array([0.02, 0.51, 0.02, 0.51])
  alpha_pc = signed_alpha(h_int=h_int, d0_channels=channels, center=centers, delta=deltas, gf_struct=_GF2)
  cols = {name: k for k, (name, _) in enumerate(_GF2)}
  channel = pairs.index((0, 2))
  for s_, sgn in enumerate((1.0, -1.0)):
    assert np.isclose(alpha_pc[3 + channel, 0, 0, s_], 0.1 + sgn * 0.02)   # attractive D0: same direction
    assert np.isclose(alpha_pc[3 + channel, 1, 1, s_], 0.9 + sgn * 0.02)
    for l, (m, coef) in enumerate(h_int):
      (_, (b0, _i0)), (_, (b1, _i1)), _, _ = m                              # pair 0 is c+_b0 ... c_b0
      direction = -np.sign(np.real(coef)) * sgn
      assert np.isclose(alpha_pc[l, 0, 0, s_], centers[cols[b0]] + direction * deltas[cols[b0]])
      assert np.isclose(alpha_pc[l, 1, 1, s_], centers[cols[b1]] + sgn * deltas[cols[b1]])
  for l, (m, _) in enumerate(h_int):
    (_, (b0, _i0)), (_, (b1, _i1)), _, _ = m
    assert np.isclose(alpha_pc[l].mean(axis=-1)[0, 0], centers[cols[b0]])
    assert np.isclose(alpha_pc[l].mean(axis=-1)[1, 1], centers[cols[b1]])
  with pytest.raises(ValueError, match="gf_struct"):
    signed_alpha(h_int=h_int, d0_channels=channels, center=centers, delta=deltas)


def test_resolve_tau_grids():
  """resolve_tau_grids sizes the G0 and D0 tau grids from the DLR mesh unless overridden.

  Setup: closed forms.

  Cases:
    1. beta 500, dlr_wmax 20: max(10001, 2 500 20 + 1) = 20001 for both grids.
    2. beta 20, dlr_wmax 20: the floor 10001 for both.
    3. Overrides: n_tau 4001 alone sets both; n_tau_bosonic 2001 with it keeps 4001 / 2001.
  """
  assert resolve_tau_grids(beta=500.0, dlr_wmax=20.0) == (20001, 20001)
  assert resolve_tau_grids(beta=20.0, dlr_wmax=20.0) == (10001, 10001)
  assert resolve_tau_grids(beta=20.0, dlr_wmax=20.0, n_tau=4001) == (4001, 4001)
  assert resolve_tau_grids(beta=20.0, dlr_wmax=20.0, n_tau=4001, n_tau_bosonic=2001) == (4001, 2001)


class Solver:
  """Stand-in for a solved triqs_ctint.Solver: the three members g_from_m_tau reads."""

  def __init__(self, M_tau, M_hartree, G0_shift_iw):
    self.M_tau = M_tau
    self.M_hartree = M_hartree
    self.G0_shift_iw = G0_shift_iw


def test_g_from_m_tau():
  """g_from_m_tau builds G = G0s + G0s (M + M_hartree) G0s per block from the DLR fit of M(tau).

  Setup: beta 10, one block "up_0" of size 1, G0s = 1/(iw - 0.3) on the fermionic DLR mesh
  (w_max 5, eps 1e-10), M(tau) on 201 uniform points; Matsubara integers n = [-3, -1, 1, 3, 21].

  Cases:
    1. M(tau) = 0, M_hartree = 0.7: G = G0s + 0.7 G0s^2 at every point to 1e-10 (the fit of a zero
       function is zero).
    2. M(tau) = -exp(-0.5 tau)/(1 + exp(-0.5 beta)) (a one-pole fermionic function, M(iw) =
       1/(iw - 0.5)), M_hartree = 0: G = G0s + G0s M(iw) G0s to 1e-8 (the DLR fit of the pole).
  """
  beta, gf_struct = 10.0, [("up_0", 1)]
  mesh_f = MeshDLRImFreq(beta=beta, statistic="Fermion", w_max=5.0, eps=1e-10, symmetrize=True)
  g0s = Gf(mesh=mesh_f, target_shape=(1, 1))
  g0s.data[:, 0, 0] = 1.0 / (1j * (2 * np.array([w.index for w in mesh_f]) + 1) * np.pi / beta - 0.3)
  G0_shift_iw = BlockGf(name_list=["up_0"], block_list=[g0s])
  mesh_t = MeshImTime(beta=beta, statistic="Fermion", n_tau=201)
  tau = np.array([float(t) for t in mesh_t])
  n_f = np.array([-3, -1, 1, 3, 21])
  iw = 1j * np.pi * n_f / beta
  g0 = 1.0 / (iw - 0.3)

  m_zero = Gf(mesh=mesh_t, target_shape=(1, 1))
  m_zero.data[:] = 0.0
  solver = Solver(M_tau=BlockGf(name_list=["up_0"], block_list=[m_zero]), M_hartree=[np.array([[0.7]])],
                  G0_shift_iw=G0_shift_iw)
  got = g_from_m_tau(S=solver, gf_struct=gf_struct, n_f=n_f, beta=beta, w_max=5.0, eps=1e-10)
  assert np.allclose(got[0][:, 0, 0], g0 + 0.7 * g0**2, atol=1e-10)

  m_pole = Gf(mesh=mesh_t, target_shape=(1, 1))
  m_pole.data[:, 0, 0] = -np.exp(-0.5 * tau) / (1.0 + np.exp(-0.5 * beta))
  solver = Solver(M_tau=BlockGf(name_list=["up_0"], block_list=[m_pole]), M_hartree=[np.zeros((1, 1))],
                  G0_shift_iw=G0_shift_iw)
  got = g_from_m_tau(S=solver, gf_struct=gf_struct, n_f=n_f, beta=beta, w_max=5.0, eps=1e-10)
  assert np.allclose(got[0][:, 0, 0], g0 + g0 * (1.0 / (iw - 0.5)) * g0, atol=1e-8)


def test_auto_alpha():
  """auto_alpha centres the shifts on the occupation of a colour near 0 or 1 and leaves the default
  shifts for the others.

  Setup: occupations [0.03, 0.5, 0.97, 0.2], delta 0.02, edge 0.05, defaults 0.5 +- 0.51.

  Cases:
    1. centers [0.03, 0.5, 0.97, 0.5], deltas [0.02, 0.51, 0.02, 0.51], is_occ [T, F, T, F]
       (0.2 is away from both edges, so it keeps the default shifts).
    2. edge 0.25: the colour at 0.2 is centred too (is_occ [T, F, T, T]); a colour exactly at the
       edge counts as near (0.05 with edge 0.05).
  """
  occ = np.array([0.03, 0.5, 0.97, 0.2])
  centers, deltas, is_occ = auto_alpha(occ=occ, delta=0.02, edge=0.05, default_center=0.5, default_delta=0.51)
  assert np.allclose(centers, [0.03, 0.5, 0.97, 0.5]) and np.allclose(deltas, [0.02, 0.51, 0.02, 0.51])
  assert is_occ.tolist() == [True, False, True, False]
  _, _, is_occ_wide = auto_alpha(occ=occ, delta=0.02, edge=0.25, default_center=0.5, default_delta=0.51)
  assert is_occ_wide.tolist() == [True, False, True, True]
  _, _, at_edge = auto_alpha(occ=np.array([0.05]), delta=0.02, edge=0.05, default_center=0.5, default_delta=0.51)
  assert at_edge.tolist() == [True]


def test_alpha_bracket_check():
  """alpha_bracket_check passes only when every occupation-centred colour's measured occupation lies
  inside its bracket and the sign reaches the minimum.

  Setup: centres [0.03, 0.5, 0.97], deltas [0.02, 0.51, 0.02], is_occ [T, F, T].

  Cases:
    1. Measured [0.04, 0.9, 0.96], sign 0.95, min 0.9: ok, max_offset 0.01, no violations (the
       default colour is off its centre by 0.4 but is not checked).
    2. Measured [0.06, 0.5, 0.96], sign 0.95: colour 0 outside (offset 0.03 >= 0.02), not ok.
    3. Measured as in 1, sign 0.8: not ok on the sign alone, no violations.
  """
  centers, deltas, is_occ = np.array([0.03, 0.5, 0.97]), np.array([0.02, 0.51, 0.02]), np.array([True, False, True])
  ok = alpha_bracket_check(occ_measured=np.array([0.04, 0.9, 0.96]), centers=centers, deltas=deltas, is_occ=is_occ,
                           sign=0.95, min_sign=0.9)
  assert ok["ok"] and np.isclose(ok["max_offset"], 0.01) and ok["violations"] == []
  out = alpha_bracket_check(occ_measured=np.array([0.06, 0.5, 0.96]), centers=centers, deltas=deltas, is_occ=is_occ,
                            sign=0.95, min_sign=0.9)
  assert not out["ok"] and out["violations"] == [0] and np.isclose(out["max_offset"], 0.03)
  low = alpha_bracket_check(occ_measured=np.array([0.04, 0.9, 0.96]), centers=centers, deltas=deltas, is_occ=is_occ,
                            sign=0.8, min_sign=0.9)
  assert not low["ok"] and low["violations"] == []


def test_problem_occupations():
  """problem_occupations reads the occupation of every colour off occupations_sab in gf_struct colour
  order and averages it over the degenerate blocks.

  Setup: gf_struct [("up_0", 2), ("down_0", 2)] (colours (0 up), (1 up), (0 dn), (1 dn)),
  occupations_sab diagonal [0.1, 0.2] for up and [0.3, 0.4] for down, an ImpurityProblem with
  zero arrays otherwise on a DLR IAFT (beta 10).

  Cases:
    1. degenerate_blk None: [0.1, 0.2, 0.3, 0.4].
    2. degenerate_blk [[0, 1]]: the two blocks averaged position by position, [0.2, 0.3, 0.2, 0.3].
  """
  gf_struct = [("up_0", 2), ("down_0", 2)]
  iaft = IAFT(beta=10.0, wmax=10.0, prec="custom", eps=1e-8, verbose=False, basis="dlr")
  nw_f, nw_b = len(iaft.wn_mesh('f')), len(iaft.wn_mesh('b', positive_only=True))
  occ = np.zeros((2, 2, 2))
  occ[0] = np.diag([0.1, 0.2])
  occ[1] = np.diag([0.3, 0.4])

  def problem(groups):
    return ImpurityProblem(gf_struct=gf_struct, beta=10.0, iaft=iaft, dlr_wmax=10.0, dlr_eps=1e-8,
                           h0_sab=np.zeros((2, 2, 2), dtype=complex),
                           delta_wsab=np.zeros((nw_f, 2, 2, 2), dtype=complex),
                           v_abcd=np.zeros((2, 2, 2, 2), dtype=complex),
                           u_weiss_wabcd=np.zeros((nw_b, 2, 2, 2, 2), dtype=complex), target_density=1.0,
                           occupations_sab=occ, degenerate_blk=groups)
  assert np.allclose(problem_occupations(problem(None)), [0.1, 0.2, 0.3, 0.4])
  assert np.allclose(problem_occupations(problem([np.array([0, 1])])), [0.2, 0.3, 0.2, 0.3])
