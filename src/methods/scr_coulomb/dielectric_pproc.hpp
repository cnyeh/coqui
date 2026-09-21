/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */

#ifndef COQUI_DIELECTRIC_PPROC_HPP
#define COQUI_DIELECTRIC_PPROC_HPP

#include <map>
#include <optional>
#include <string>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {

  /**
   * Post-processing dielectric function with polarization-head regularization.
   *
   * Rebuilds the polarization Pi_PQ(q, i nu_n) from the Green's function stored at
   * {grp_name}/iter{N}/G_tskij of a CoQui checkpoint, applies `pi_regularization` exactly as
   * scr_coulomb_t does inside the self-consistent loop, solves the Dyson equation for W,
   * and stores the head of the inverse dielectric function under
   *
   *   {grp_name}/iter{N}/dielectric/{pi_regularization}/
   *     eps_inv_head_wq     (nw_half, nqpts_ibz)   eps^-1_00(q, i nu_n) - 1
   *     eps_inv_head_w      (nw_half,)             div_treatment q -> 0 limit
   *     pi_head_wq          (nw_half, nqpts_ibz)   Pi_00(q, i nu_n) before projection
   *     delta_C_w           (nw_half,)             applied shift; absent for "none"
   *     screen_type, div_treatment, pi_regularization   strings
   *
   * eps_inv_head_wq, eps_inv_head_w and pi_head_wq are always written. Requires a THC
   * object with G=0 head vectors (ISDF path); refuses otherwise.
   *
   * The subgroup for the requested option is replaced on rerun; sibling options and every
   * dataset outside dielectric/ are left untouched. The subgroup is keyed on
   * pi_regularization ALONE, so two runs that differ only in div_treatment (or in
   * screen_type) overwrite each other, with the stored provenance strings as the only
   * trace of which one survived. Note that the in-loop head stored at
   * scf/iter{N} was built from the G of iteration N-1, so post-processing iteration N-1
   * with "none" reproduces scf/iter{N}/eps_inv_head_* (to round-off), and iteration 0
   * gives the G0W0 dielectric function.
   *
   * Keys in `pt`:
   *   prefix            (required)   checkpoint prefix; file is outdir/prefix.mbpt.h5
   *   outdir            "./"
   *   grp_name          "scf"        "scf" or "embed". With "embed" the input G is the
   *                                  EDMFT lattice Green's function; note that "embed" has
   *                                  no in-loop head to compare against, as
   *                                  dump_eps_inv_head only ever writes under "scf"
   *   iteration         -1           negative -> {grp_name}/final_iter
   *   pi_regularization "none"       "none" | "dynamic" | "insulator" | "extrapolate"
   *   screen_type       "rpa"        any recipe scr_coulomb_t understands, e.g. "rpa",
   *                                  "crpa", "crpa_ks", "crpa_vasp", "gw_edmft",
   *                                  "gw_edmft_rpa", "gw_edmft_density". A "crpa" or
   *                                  "edmft" variant needs the bosonic projector
   *   wannier_file      -            required when screen_type needs a projector and the
   *                                  projector is not passed as arrays (second overload)
   *   translate_home_cell  false     projector option, as elsewhere
   *   div_treatment     "gygi"       q -> 0 extrapolation of the head
   * "beta" and "iaft" are rejected: the frequency grid is the checkpoint's.
   *
   * For an "edmft" screen_type the local corrections Pi_imp/Pi_dc are taken from the
   * `local_polarizabilities` argument when it is given; otherwise they are read from
   * downfold_2e/iter{M}/{Pi_imp_wabcd, Pi_dc_wabcd} of the checkpoint. When neither is
   * available the routine warns and proceeds with a zero local correction, i.e. with the
   * plain RPA polarization. Note that "gw_edmft_rpa" is an exception on both counts: it
   * carries the "edmft" keyword, so a projector is demanded and the local corrections are
   * resolved, but eval_Pi_qdep returns the bare RPA bubble before either is used. The
   * "Effective polarization" line of the log always names the recipe that actually ran.
   *
   * Applying the projection here acts on a G that was converged with its own (possibly
   * unregularized) W: the result is a one-shot regularized dielectric function from that
   * G, not the in-loop one.
   *
   * @param thc - [INPUT] THC Coulomb integrals; must carry the G=0 head vectors
   * @param pt  - [INPUT] parameters, see above
   * @param local_polarizabilities - [INPUT, optional] {"imp", "dc"} local polarizabilities
   *                                 in the downfolded basis; used only by "edmft" recipes
   */
  void dielectric_function(
      THC_ERI auto &thc, ptree const &pt,
      std::optional<std::map<std::string, nda::array<ComplexType, 5> > >
          local_polarizabilities = std::nullopt);

  /**
   * Same as above, with the bosonic projector supplied directly instead of being read
   * from an HDF5 Wannier file. The "wannier_file" key is then unused.
   *
   * @param thc             - [INPUT] THC Coulomb integrals; must carry the G=0 head vectors
   * @param pt              - [INPUT] parameters, see the other overload
   * @param projector_ksIai - [INPUT] projector from Bloch states to the local basis
   * @param band_window     - [INPUT] band window of the projector
   * @param kpts_crys       - [INPUT] k-points of the projector in crystal coordinates
   * @param local_polarizabilities - [INPUT, optional] see the other overload
   */
  void dielectric_function(
      THC_ERI auto &thc, ptree const &pt,
      nda::array<ComplexType, 5> const &projector_ksIai,
      nda::array<long, 3> const &band_window,
      nda::array<RealType, 2> const &kpts_crys,
      std::optional<std::map<std::string, nda::array<ComplexType, 5> > >
          local_polarizabilities = std::nullopt);

} // methods

#endif // COQUI_DIELECTRIC_PPROC_HPP
