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
"""
CT-INT's own helpers: the D0 channel order of ``triqs_ctint``, the signed alpha tensor and the
M(tau) route to G. The numerics shared with the other QMC backends live in ``qmc_postproc``.
"""
import numpy as np
from coqui import app_log
from triqs.gfs import Gf, fit_gf_dlr
from coqui.dmft.qmc_postproc import (color_table, block_offsets, color_groups, seed_for_rank,      # noqa: F401
                                     redirect_solver_output, eval_on_points, static_color_matrix,
                                     color_matrix_at_points, d0_tau_colors, average_blocks,
                                     average_color_matrix, sigma_infty_hf, connected_by_plateau,
                                     fit_tau_correlator, pi_w_from_chi)

# an opposite-sign part of D(tau) up to this fraction of max |D| is numerical noise
_SIGN_EXCURSION_TOL = 1e-3
_D0_ZERO = 1e-12


def d0_channel_colors(gf_struct):
  """
  The colour pairs of the ``D0`` vertices in ``triqs_ctint``'s label order.

  The label of n_{b1,a}(tau) n_{b2,a'}(tau') is ``n_h_int + b1 n_bl R^2 + b2 R^2 + a R + a'`` with
  ``R`` the common block size: the first pair (b1, a) takes the alpha entry ``[., 0, 0]``, the
  second (b2, a') the entry ``[., 1, 1]``.

  Parameters
  ----------
  gf_struct : list of (str, int)
      All blocks must have the same size.

  Returns
  -------
  list of (int, int)
      ``(colour of (b1, a), colour of (b2, a'))`` for b1, b2 over the blocks and a, a' over the
      block size, row-major.

  Raises
  ------
  ValueError
      Blocks of unequal size.
  """
  sizes = {size for _, size in gf_struct}
  if len(sizes) != 1:
    raise ValueError(f"ctint_utils::d0_channel_colors: triqs_ctint's D0 needs gf_struct blocks of "
                     f"equal size, got {gf_struct}")
  size, n_blk, offs = sizes.pop(), len(gf_struct), block_offsets(gf_struct)
  return [(offs[b1] + a, offs[b2] + ap) for b1 in range(n_blk) for b2 in range(n_blk)
          for a in range(size) for ap in range(size)]


def resolve_tau_grids(*, beta, dlr_wmax, n_tau=None, n_tau_bosonic=None, n_tau_min=10001):
  """
  The tau grids of ``G0(tau)`` and ``D0(tau)`` from the impurity DLR mesh unless overridden:
  ``max(n_tau_min, 2 beta dlr_wmax + 1)`` points each (``n_tau_bosonic`` defaults to ``n_tau``).

  Returns
  -------
  n_tau, n_tau_bosonic : int
  """
  if n_tau is None:
    n_tau = max(int(n_tau_min), int(2 * beta * dlr_wmax) + 1)
  if n_tau_bosonic is None:
    n_tau_bosonic = n_tau
  return int(n_tau), int(n_tau_bosonic)


# ---------------------------------------------------------------- alpha

def signed_alpha(h_int, d0_channels, center, delta, gf_struct=None):
  """
  The alpha tensor of a density-density interaction in ``triqs_ctint``'s layout.

  Each vertex contributes ``(n_a - alpha_a)(n_b - alpha_b)`` with ``alpha = center +- delta`` over
  the two auxiliary spins: opposite directions for a repulsive coupling, the same direction for
  an attractive one. The defaults ``center = 0.5``, ``delta = 0.51`` put the shifts just outside
  [0, 1] at every filling. The library compensates any alpha exactly, so the choice affects the
  sign and the expansion order, not the result. The cross entries ``[., 0, 1]`` and ``[., 1, 0]``
  are zero. A ``D(tau)`` channel that changes sign (a non-causal bosonic Weiss field, as in the
  first GW+EDMFT iteration from a double-counting start) takes the direction of its dominant
  part, with a warning: no single direction fits, the result stays exact, the Monte Carlo sign
  may suffer.

  Parameters
  ----------
  h_int : triqs.operators.Operator
      Density-density terms ``coef n_a n_b`` only (``c+_a c+_b c_b c_a``); anything else raises.
  d0_channels : list of np.ndarray
      ``D(tau)`` of every ``D0`` channel, in the order of ``d0_channel_colors``.
  center, delta : float or array (n_color,)
      Centre and half-width of the shifts, one value for every colour or one per colour (then
      ``gf_struct`` is required to find the colour of each operator).
  gf_struct : list of (str, int), optional

  Returns
  -------
  np.ndarray
      Shape ``(n_terms + n_D0, 2, 2, 2)``: vertex, operator pair, operator pair, auxiliary spin.

  Raises
  ------
  ValueError
      A non-density-density term of ``h_int``; per-colour shifts without ``gf_struct``.
  """
  per_color = np.ndim(center) > 0 or np.ndim(delta) > 0
  if per_color and gf_struct is None:
    raise ValueError("ctint_utils::signed_alpha: per-colour centres or deltas need gf_struct")
  if per_color:
    offs = block_offsets(gf_struct)
    n_color = sum(size for _, size in gf_struct)
    center = np.broadcast_to(np.asarray(center, dtype=float), (n_color,))
    delta = np.broadcast_to(np.asarray(delta, dtype=float), (n_color,))
    colour_of = {(name, i): offs[b] + i for b, (name, size) in enumerate(gf_struct) for i in range(size)}
    pairs = d0_channel_colors(gf_struct)
  else:
    center, delta = float(center), float(delta)

  def shifts(operator_index, channel):
    """(centre, delta) of a vertex operator: by colour when per colour, the scalars otherwise."""
    if not per_color:
      return center, delta
    col = colour_of[operator_index] if channel is None else pairs[channel][operator_index]
    return float(center[col]), float(delta[col])

  terms = list(h_int)
  for monomial, _ in terms:
    (_, (b0, i0)), (_, (b1, i1)), (_, (b1p, i1p)), (_, (b0p, i0p)) = monomial
    if (b0, i0) != (b0p, i0p) or (b1, i1) != (b1p, i1p):
      raise ValueError(f"ctint_utils::signed_alpha: density-density terms only; h_int has {monomial}")
  alpha = np.zeros((len(terms) + len(d0_channels), 2, 2, 2))
  for s, sgn in enumerate((1.0, -1.0)):
    for l, (monomial, coef) in enumerate(terms):
      (_, (b0, i0)), (_, (b1, i1)), _, _ = monomial
      c0, d0 = shifts((str(b0), int(i0)), None)
      c1, d1 = shifts((str(b1), int(i1)), None)
      alpha[l, 0, 0, s] = c0 - np.sign(np.real(coef)) * sgn * d0
      alpha[l, 1, 1, s] = c1 + sgn * d1
    for d, channel in enumerate(d0_channels):
      channel = np.asarray(channel, dtype=float)
      scale = np.abs(channel).max()
      if scale < _D0_ZERO:
        sign = -1.0
      else:
        sign = float(np.sign(channel[np.abs(channel).argmax()]))     # the dominant part decides
        excursion = max(0.0, float((-sign * channel).max()))          # largest opposite-sign value
        if excursion > _SIGN_EXCURSION_TOL * scale and s == 0:
          app_log(1, f" [WARNING] D(tau) channel {d} changes sign (opposite-sign part {excursion:.2e}\n"
                     f"  vs max |D| {scale:.2e}); no single shift direction fits, the dominant part's is\n"
                     f"  used (exact; the Monte Carlo sign may suffer)\n")
      ca, da = shifts(0, d)
      cb, db = shifts(1, d)
      alpha[len(terms) + d, 0, 0, s] = ca - sign * sgn * da
      alpha[len(terms) + d, 1, 1, s] = cb + sgn * db
  return alpha


def problem_occupations(problem):
  """
  The occupation of every colour of ``problem.gf_struct`` from ``problem.occupations_sab`` (the
  lattice occupations the workflow carries), averaged over the degenerate blocks as the solution
  is symmetrized over them.

  Returns
  -------
  np.ndarray (n_color,)
  """
  gf_struct = problem.gf_struct
  _, _, spin, orb = color_table(gf_struct)
  occ = np.array([float(np.real(problem.occupations_sab[s, o, o])) for s, o in zip(spin, orb)])
  offs = block_offsets(gf_struct)
  for group in problem.degenerate_blk or []:
    blocks = [int(b) for b in group]
    for inner in range(gf_struct[blocks[0]][1]):           # position by position over the group's blocks
      idx = [offs[b] + inner for b in blocks]
      occ[idx] = occ[idx].mean()
  return occ


def auto_alpha(occ, delta, edge, default_center=0.5, default_delta=0.51):
  """
  The occupation-based shifts: a colour whose occupation lies within ``edge`` of 0 or 1 gets the
  shifts ``n +- delta`` centred on its occupation, which keeps the expansion order near 1 and the
  sign near 1 for an almost empty or full orbital; every other colour keeps the default shifts
  ``default_center +- default_delta``, because bracketing an occupation away from the edges puts
  the shifts inside (0, 1), where the sign collapses.

  Parameters
  ----------
  occ : array (n_color,)
      The occupation of every colour (``problem_occupations``).
  delta : float
      Half-width of the occupation-centred shifts.
  edge : float
      Distance from 0 or 1 within which a colour is occupation-centred.

  Returns
  -------
  centers, deltas : np.ndarray (n_color,)
  is_occ : np.ndarray (n_color,) of bool
      ``True`` for the occupation-centred colours.
  """
  occ = np.asarray(occ, dtype=float)
  is_occ = (occ <= float(edge)) | (occ >= 1.0 - float(edge))
  centers = np.where(is_occ, occ, float(default_center))
  deltas = np.where(is_occ, float(delta), float(default_delta))
  return centers, deltas, is_occ


def alpha_bracket_check(occ_measured, centers, deltas, is_occ, sign, min_sign):
  """
  Whether the occupation-centred shifts were adequate: every occupation-centred colour's measured
  occupation lies inside its bracket ``|n - c| < delta`` (outside it the sign collapses and the
  result is unreliable) and the average sign is at least ``min_sign``.

  Returns
  -------
  dict
      ``ok``, ``max_offset`` (largest ``|n - c|`` over the occupation-centred colours, 0 without
      any) and ``violations`` (their colour indices).
  """
  occ_measured, centers, deltas = (np.asarray(x, dtype=float) for x in (occ_measured, centers, deltas))
  is_occ = np.asarray(is_occ, dtype=bool)
  offsets = np.abs(occ_measured - centers)
  violations = [int(i) for i in np.flatnonzero(is_occ & (offsets >= deltas))]
  max_offset = float(offsets[is_occ].max()) if is_occ.any() else 0.0
  return {"ok": bool(not violations and float(sign) >= float(min_sign)), "max_offset": max_offset,
          "violations": violations}


# ---------------------------------------------------------------- G from M(tau)

def g_from_m_tau(S, gf_struct, n_f, beta, w_max, eps):
  """
  G at the Matsubara integers ``n_f`` from the backend's own DLR fit of the measured ``M(tau)``.

  ``G = G0s + G0s M G0s`` per block with ``G0s = S.G0_shift_iw`` and ``M(iw)`` the DLR fit
  (``w_max``, ``eps``) of ``S.M_tau`` plus the equal-time ``S.M_hartree``, as ``triqs_ctint``'s
  post-processing builds it, but with a fit accuracy that matches the Monte Carlo data instead
  of the solver's ``dlr_eps``, and evaluated directly at ``n_f`` (no second fit of a noisy G on
  DLR nodes). Needs the solve to have run with ``post_process=True``.

  Returns
  -------
  list of np.ndarray (len(n_f), d, d), complex, in block order
  """
  out = []
  for b, (name, _) in enumerate(gf_struct):
    mt = S.M_tau[name]
    m = Gf(mesh=mt.mesh, target_shape=tuple(mt.target_shape))     # the library's M(tau) is real; the fit needs complex
    m.data[:] = np.asarray(mt.data)
    M = eval_on_points(fit_gf_dlr(m, w_max=w_max, eps=eps, symmetrize=True), n_f, "Fermion", beta)
    M = M + np.asarray(S.M_hartree[b])
    g0s = eval_on_points(S.G0_shift_iw[name], n_f, "Fermion", beta)
    out.append(g0s + np.einsum("wab,wbc,wcd->wad", g0s, M, g0s))
  return out
