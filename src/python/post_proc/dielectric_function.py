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

import json
import warnings
import numpy as np
from mpi4py import MPI

from coqui._lib import post_proc_module as pproc_mod

# src/methods/scr_coulomb/dielectric_pproc.cpp holds the authoritative sets; these copies
# exist so an invalid value raises a catchable ValueError instead of reaching the C++
# utils::check, whose MPI_Abort no test can catch. Extend both when a value is added.
_VALID_REGULARIZATION = ("none", "dynamic", "insulator", "extrapolate")
_VALID_GRP = ("scf", "embed")


def _read_head_group(h5, filename, grp_name, iteration, reg):
  """Read one {grp_name}/iter{N}/dielectric/{reg} group. Root-rank only; see the caller.

  Returns (dict, resolved_iteration). A negative `iteration` is resolved here against
  {grp_name}/final_iter, and the resolved value is returned so every rank reports the same
  provenance."""
  out = {}
  with h5.HDFArchive(filename, "r") as ar:
    top = ar[grp_name]
    if iteration < 0:
      iteration = int(top["final_iter"])
    g = top[f"iter{iteration}"]["dielectric"][reg]
    for key in ("eps_inv_head_wq", "eps_inv_head_w", "pi_head_wq"):
      out[key] = np.asarray(g[key])
    out["delta_C_w"] = np.asarray(g["delta_C_w"]) if "delta_C_w" in g else None
    # Checkpoints written by older CoQuí versions carry no 'iwn_mesh' (iaft_utils.cpp
    # warns and skips the grid check on those files), so the frequency labels are
    # optional: the head datasets above are complete and correct without them.
    ia = ar["imaginary_fourier_transform"] if "imaginary_fourier_transform" in ar else None
    has_mesh = ia is not None and "iwn_mesh" in ia and "beta" in ia
    if not has_mesh:
      warnings.warn("dielectric_function: this checkpoint stores no "
                    "'imaginary_fourier_transform/iwn_mesh' (or no 'beta'); it predates the "
                    "stored frequency mesh, so the Matsubara frequencies cannot be "
                    "reconstructed and 'nu' is None. The head datasets were computed and "
                    "written correctly.")
      out["nu"] = None
    else:
      # 'iwn_mesh/boson' stores the CoQuí index n of iν_n = n·π/β over the full (negative
      # and positive) mesh; the stored head lives on the positive half, which starts at
      # nw_b//2 (IAFT.icc, tau_to_w_PHsym: iw = nw_b/2 + n).
      wn = np.asarray(ia["iwn_mesh"]["boson"])
      off = wn.shape[0] // 2
      nw = out["eps_inv_head_w"].shape[0]
      # A mesh that is present but misaligned is a real inconsistency, not an old file.
      if off + nw > wn.shape[0] or wn[off] != 0:
        raise ValueError("dielectric_function: the bosonic mesh of the checkpoint has no "
                         f"zero frequency at index {off} (nw_b = {wn.shape[0]}, nw_half = {nw}); "
                         "the stored head cannot be mapped onto Matsubara frequencies")
      out["nu"] = np.pi * wn[off:off + nw] / float(ia["beta"])

  return out, iteration


def dielectric_function(h_int, params, *, projector_info=None, local_polarizabilities=None):
  """
  Head of the inverse dielectric function from a checkpoint Green's function, with an
  optional polarization-head regularization applied in post-processing.

  The polarization ``Π_PQ(q, iν_n)`` is rebuilt from ``{grp_name}/iter{N}/G_tskij`` of the
  checkpoint, regularized exactly as ``pi_regularization`` does inside ``run_gw``, and pushed
  through the Dyson equation for ``W``. Only the head ``ε⁻¹₀₀(q, iν_n) − 1`` and its
  ``q → 0`` limit are stored, under ``{grp_name}/iter{N}/dielectric/{pi_regularization}/``.
  The in-loop datasets in ``{grp_name}/iter{N}`` are never modified; rerunning an option
  replaces only that option's subgroup. Two runs differing only in ``div_treatment`` write to
  the same subgroup (the stored ``div_treatment`` string is the only trace).

  Two facts worth knowing before comparing numbers:

  * The in-loop head stored at ``scf/iter{N}`` was built from the Green's function of
    iteration ``N−1``. Post-processing iteration ``N−1`` with ``"none"`` therefore
    reproduces ``scf/iter{N}/eps_inv_head_wq`` (to round-off), and iteration ``0`` gives
    the G₀W₀ dielectric function.
  * With a regularization, the projection acts on a ``G`` that was converged with its own
    (possibly unregularized) ``W``. The result is a one-shot regularized dielectric
    function from that ``G``, not the in-loop result.

  Parameters
  ----------
  h_int : ThcCoulomb
      THC Coulomb integrals from ``make_thc_coulomb``, built through the ISDF path so they
      carry the ``G = 0`` head vectors (LS-THC from Cholesky ERIs is refused). Cholesky
      integrals are not accepted.
  params : dict
      - ``prefix`` *(str, required)* — checkpoint prefix; the file is
        ``outdir/prefix.mbpt.h5``.
      - ``outdir`` *(str, default ``"./"``)*.
      - ``grp_name`` *(str, default ``"scf"``)* — ``"scf"`` or ``"embed"``.
      - ``iteration`` *(int, default ``-1``)* — negative selects ``{grp_name}/final_iter``.
        The iteration must hold ``G_tskij`` (Dyson-type checkpoints; qpGW/evGW iterations
        are refused).
      - ``pi_regularization`` *(str, default ``"none"``)* — ``"none"``, ``"dynamic"``,
        ``"insulator"`` or ``"extrapolate"``; see ``run_gw`` for their meaning.
      - ``screen_type`` *(str, default ``"rpa"``)* — any screening recipe ``scr_coulomb_t``
        understands: ``"rpa"``, ``"crpa"``, ``"crpa_ks"``, ``"crpa_vasp"``, ``"gw_edmft"``,
        ``"gw_edmft_rpa"``, ``"gw_edmft_density"``. A ``crpa`` or ``edmft`` variant needs
        the bosonic projector, supplied either through ``params["wannier_file"]`` or
        through ``projector_info``.
      - ``wannier_file`` *(str)* — HDF5 file holding the Wannier transformation matrices.
        Required when ``screen_type`` needs a projector and ``projector_info`` is not given.
      - ``translate_home_cell`` *(bool, default ``False``)* — projector option, as in
        ``downfold_coulomb``.
      - ``div_treatment`` *(str, default ``"gygi"``)* — ``q → 0`` extrapolation of the head.

      ``beta`` and ``iaft`` are rejected: the frequency grid is the checkpoint's.
  projector_info : dict, optional
      Projector supplied directly instead of read from ``wannier_file``. Expected keys:

      - ``"proj_mat"`` — the projector matrix.
      - ``"band_window"`` — the band window information.
      - ``"kpts_w90"`` — the k-points in Wannier90 format.

      ``coqui.read_proj_info(<wannier file>)`` returns such a dict.
  local_polarizabilities : dict, optional
      Local polarizability corrections in the downfolded basis, used only by the ``edmft``
      recipes. Expected keys:

      - ``"imp"`` — impurity polarizability.
      - ``"dc"`` — double-counting polarizability.

      When omitted, they are read from ``downfold_2e/iter{M}/{Pi_imp_wabcd, Pi_dc_wabcd}``
      of the checkpoint; when those are absent too, CoQuí warns and proceeds with a zero
      local correction, i.e. with the plain RPA polarization.

  Returns
  -------
  dict
      - ``eps_inv_head_wq`` : complex ``(nw_half, nqpts_ibz)`` — ``ε⁻¹₀₀(q, iν_n) − 1`` on
        the irreducible ``q`` mesh, positive bosonic frequencies. ``ε_M = 1 / (1 + head)``.
      - ``eps_inv_head_w`` : complex ``(nw_half,)`` — the ``q → 0`` limit used for the
        Madelung term.
      - ``pi_head_wq`` : complex ``(nw_half, nqpts_ibz)`` — ``Π₀₀(q, iν_n)`` before the
        projection, the Ward-violation diagnostic.
      - ``delta_C_w`` : complex ``(nw_half,)`` or ``None`` — the applied shift; ``None`` for
        ``"none"``.
      - ``nu`` : float ``(nw_half,)`` or ``None`` — the non-negative bosonic Matsubara
        frequencies of the checkpoint's sampling mesh, in Hartree (``nu[0] == 0``); CoQuí
        stores the even index ``m = 2n`` of ``iν = m·π/β``, so the physical ``ν_n = 2πn/β``.
        ``None`` (with a warning) if the checkpoint predates the stored frequency mesh —
        the head datasets themselves are unaffected.
      - ``iteration``, ``grp_name``, ``pi_regularization`` — the resolved provenance.

  Examples
  --------
  ::

      from coqui.post_proc import dielectric_function

      res = dielectric_function(thc, {
          "prefix": "si", "outdir": "./", "iteration": -1,
          "pi_regularization": "extrapolate",
      })
      eps_M = 1.0 / (1.0 + res["eps_inv_head_wq"].real)   # (nu, q)
  """
  p = dict(params)
  if "prefix" not in p:
    raise ValueError("dielectric_function: 'prefix' is required")
  for key in ("beta", "iaft"):
    if key in p:
      raise ValueError(f"dielectric_function: '{key}' is not accepted; the frequency grid "
                       "is read from the checkpoint")
  reg = str(p.get("pi_regularization", "none")).lower()
  if reg not in _VALID_REGULARIZATION:
    raise ValueError(f"dielectric_function: pi_regularization = '{reg}'; valid values are "
                     f"{_VALID_REGULARIZATION}")
  grp_name = str(p.get("grp_name", "scf")).lower()
  if grp_name not in _VALID_GRP:
    raise ValueError(f"dielectric_function: grp_name = '{grp_name}'; valid values are {_VALID_GRP}")
  screen_type = str(p.get("screen_type", "rpa")).lower()
  # scr_coulomb_t searches screen_type for keywords rather than matching a fixed list, so
  # no whitelist is enforced here either; only the projector it implies is checked.
  if ("edmft" in screen_type or "crpa" in screen_type) \
      and projector_info is None and "wannier_file" not in p:
    raise ValueError(f"dielectric_function: screen_type = '{screen_type}' needs the bosonic "
                     "projector; provide it either through params['wannier_file'] or through "
                     "the projector_info keyword argument")
  if local_polarizabilities is not None:
    missing = {"imp", "dc"} - local_polarizabilities.keys()
    if missing:
      raise ValueError(f"dielectric_function: missing keys in local_polarizabilities: {missing}")
  p["pi_regularization"] = reg
  p["grp_name"] = grp_name
  p["screen_type"] = screen_type

  # Imported before the C++ call so a missing TRIQS h5 costs nothing: the bubble and the
  # Dyson solve are minutes of work whose result this function could not then read back.
  # Read the result with TRIQS h5 (the project-wide HDF5 library; never h5py).
  try:
    import h5
  except ImportError as e:
    raise ImportError("dielectric_function: reading the result back requires the TRIQS h5 "
                      "package (module triqs-collections). Nothing has been computed yet; "
                      "load the module and call again") from e

  if projector_info is None:
    pproc_mod.dielectric_function_with_projector_from_h5(
      h_int, json.dumps(p), local_polarizabilities=local_polarizabilities
    )
  else:
    pproc_mod.dielectric_function(
      h_int, json.dumps(p), projector_info.get("proj_mat"),
      projector_info.get("band_window"), projector_info.get("kpts_w90"),
      local_polarizabilities=local_polarizabilities
    )

  # Deliberately the same naive concatenation as dielectric_pproc.cpp, so the file read
  # back here is always the file the driver wrote. os.path.join would differ for an
  # absolute "prefix" (it drops "outdir", the C++ side does not).
  filename = p.get("outdir", "./") + "/" + p["prefix"] + ".mbpt.h5"
  iteration = int(p.get("iteration", -1))

  # ONLY the root rank reads the result back, and broadcasts it.
  #
  # Every rank opening the checkpoint would be both wasteful (one concurrent open per rank
  # of a file that can be many GB) and racy. The driver writes on root and closes before
  # its barrier, so the data are on disk -- but on a parallel filesystem the freshly
  # created subgroup is not necessarily visible to the other nodes yet, and a rank that
  # sees the older `dielectric` group without the new option subgroup fails with
  # "KeyError: Key <option> does not exist" on a calculation that in fact succeeded.
  # Reading on one rank removes both the race and the read-only/read-write lock conflict
  # between back-to-back calls.
  # TODO Revisit the MPI design. Do we want to stick with CoQui's MPI handler or use mpi4py? 
  comm = MPI.COMM_WORLD
  payload, failure = None, None
  if comm.Get_rank() == 0:
    try:
      payload, iteration = _read_head_group(h5, filename, grp_name, iteration, reg)
    except Exception as e:                     # re-raised on every rank below
      failure = f"{type(e).__name__}: {e}"
  # A failure on root alone would deadlock the others in the broadcast, so the outcome
  # travels with the data and every rank raises the same error.
  payload, iteration, failure = comm.bcast((payload, iteration, failure), root=0)
  if failure is not None:
    raise RuntimeError(f"dielectric_function: reading the result back from {filename} "
                       f"failed on the root rank with {failure}. The head datasets "
                       f"themselves were written; they are under "
                       f"{grp_name}/iter{iteration}/dielectric/{reg}/")
  out = payload

  out["iteration"] = iteration
  out["grp_name"] = grp_name
  out["pi_regularization"] = reg
  return out
