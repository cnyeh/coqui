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

#include <array>
#include <filesystem>
#include <set>
#include <string>
#include <utility>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/scr_coulomb/dielectric_pproc.hpp"

namespace methods {

  namespace {

    // Everything the two overloads and the shared implementation agree on: the parsed
    // input, the resolved iteration and the provenance strings that go into the log.
    struct diel_params_t {
      std::string prefix;
      std::string outdir;
      std::string grp_name;
      std::string pi_reg;
      std::string screen_type;
      std::string div_treatment;
      std::string output;        // outdir/prefix
      std::string filename;      // output + ".mbpt.h5"
      std::string proj_source;   // filled in by the overload, for the log only
      long iter = -1;            // resolved iteration index
      bool next_head = false;    // {grp}/iter{iter+1}/eps_inv_head_wq exists
      bool needs_projector = false;
    };

    // Resolve a negative iteration to {grp_name}/final_iter and check that the iteration
    // holds a G_tskij dataset. Root reads, everyone receives.
    template<typename comm_t>
    std::array<long, 2> resolve_iteration(comm_t &comm, std::string const &filename,
                                          std::string const &grp_name, long iteration) {
      std::string err = "dielectric_pproc.cpp::dielectric_function: ";
      std::array<long, 3> buf = {iteration, 0, 0};   // {iteration, has_G, next_head}
      if (comm.root()) {
        app_log(2, "Checking {} for {}/iter{}/G_tskij ...", filename, grp_name,
                (iteration < 0) ? std::string("final_iter") : std::to_string(iteration));
        h5::file file(filename, 'r');
        auto root = h5::group(file);
        utils::check(root.has_subgroup(grp_name),
                     err + "group \"{}\" not found in {}.", grp_name, filename);
        auto grp = root.open_group(grp_name);
        if (buf[0] < 0) {
          utils::check(grp.has_dataset("final_iter"),
                       err + "\"{}/final_iter\" not found in {}; pass \"iteration\" explicitly.",
                       grp_name, filename);
          h5::h5_read(grp, "final_iter", buf[0]);
        }
        std::string iter_name = "iter" + std::to_string(buf[0]);
        buf[1] = (grp.has_subgroup(iter_name) and
                  grp.open_group(iter_name).has_dataset("G_tskij")) ? 1 : 0;
        if (buf[1] == 1) app_log(2, "  found {}/{}/G_tskij.", grp_name, iter_name);
        // The in-loop head built from THIS iteration's G is the one stored at iteration
        // N+1, which exists only if the run went at least one iteration further. It is
        // also always absent under grp_name = "embed": dump_eps_inv_head hardcodes the
        // "scf" group and the EDMFT loops never call update_w.
        buf[2] = grp.has_dataset("iter" + std::to_string(buf[0] + 1) +
                                 "/eps_inv_head_wq") ? 1 : 0;
      }
      comm.broadcast_n(buf.data(), buf.size(), 0);
      utils::check(buf[1] == 1,
                   err + "\"{}/iter{}/G_tskij\" not found in {}. Only Dyson-type checkpoints "
                   "(scGW, GW+EDMFT) are supported; quasiparticle (evGW/qpGW) iterations "
                   "carry no G_tskij.", grp_name, buf[0], filename);
      return {buf[0], buf[2]};
    }

    // Parse and validate the input, and resolve the iteration. Done before any MBState is
    // built so that a typo does not first pay for reading a Wannier file.
    diel_params_t read_params(ptree const &pt, THC_ERI auto &thc) {
      std::string err = "dielectric_pproc.cpp::dielectric_function - Incorrect input - ";
      diel_params_t p;
      p.prefix        = io::get_value<std::string>(pt, "prefix", err + "prefix");
      p.outdir        = io::get_value_with_default<std::string>(pt, "outdir", "./");
      p.grp_name      = io::tolower_copy(
                          io::get_value_with_default<std::string>(pt, "grp_name", "scf"));
      p.pi_reg        = io::tolower_copy(
                          io::get_value_with_default<std::string>(pt, "pi_regularization", "none"));
      p.screen_type   = io::tolower_copy(
                          io::get_value_with_default<std::string>(pt, "screen_type", "rpa"));
      p.div_treatment = io::tolower_copy(
                          io::get_value_with_default<std::string>(pt, "div_treatment", "gygi"));
      auto iteration  = io::get_value_with_default<long>(pt, "iteration", -1);

      utils::check(not pt.get_child_optional("beta") and not pt.get_child_optional("iaft"),
                   err + "\"beta\" and \"iaft\" are not accepted: the frequency grid is read "
                   "from the checkpoint.");
      utils::check(p.grp_name == "scf" or p.grp_name == "embed",
                   err + "grp_name = \"{}\"; valid values are \"scf\" and \"embed\".", p.grp_name);
      // Keep in sync with scr_coulomb_t's valid set (see regularize_Pi_head); this routine
      // only forwards the string, so a value it accepts must be one scr_coulomb_t accepts.
      static const std::set<std::string> valid_reg =
          {"none", "dynamic", "insulator", "extrapolate"};
      utils::check(valid_reg.count(p.pi_reg) == 1,
                   err + "pi_regularization = \"{}\"; valid values are \"none\", \"dynamic\", "
                   "\"insulator\", \"extrapolate\".", p.pi_reg);
      // screen_type is deliberately NOT checked against a whitelist: scr_coulomb_t treats
      // it as free-form and searches it for keywords, so a list here would go stale.
      // Mirrors the branches of div_utils::extrapolate_eps_inv_q0, which otherwise only
      // rejects a bad value after the bubble and the Dyson solve have already been paid for.
      utils::check(p.div_treatment.find("gygi") != std::string::npos or
                   p.div_treatment == "ignore_g0",
                   err + "div_treatment = \"{}\"; expected \"ignore_g0\" or a \"gygi\" variant "
                   "(gygi, gygi_extrplt, gygi_average, gygi_smallest_q, ... ).", p.div_treatment);
      // The head IS the output here; a zero-filled head from a THC object without G=0
      // vectors would be a fabricated measurement (root CLAUDE.md: "skip, don't fake").
      utils::check(thc.has_basis_head(),
                   err + "this THC object carries no G=0 interpolating-vector heads (LS-THC "
                   "fitted from Cholesky ERIs), so the dielectric head cannot be computed. "
                   "Build the THC ERIs through the ISDF path.");

      p.needs_projector = (p.screen_type.find("edmft") != std::string::npos or
                           p.screen_type.find("crpa") != std::string::npos);

      p.output   = p.outdir + "/" + p.prefix;
      p.filename = p.output + ".mbpt.h5";
      utils::check(std::filesystem::exists(p.filename),
                   err + "checkpoint {} does not exist.", p.filename);

      auto info = resolve_iteration(thc.mpi()->comm, p.filename, p.grp_name, iteration);
      p.iter = info[0];
      p.next_head = (info[1] == 1);
      return p;
    }

    inline double eps_M(ComplexType head) { return 1.0 / (1.0 + head.real()); }

    // Shared implementation. `mb_state` already carries the projector (or not), and its
    // `ft` member points at the IAFT owned by the calling overload, which must therefore
    // outlive this call.
    void dielectric_function_impl(
        THC_ERI auto &thc, diel_params_t const &p, MBState &&mb_state,
        std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {

      auto mpi = thc.mpi();
      auto mf  = thc.MF();
      auto &ft = *mb_state.ft;
      auto npos = std::string::npos;

      // What eval_Pi_qdep will actually do with this string. screen_type is free-form and
      // keyword-matched, so an unsupported or misspelled value ("bare", "gw_edmt") quietly
      // degrades to plain RPA while the checkpoint records the user's spelling. Naming the
      // recipe here is the only place that discrepancy becomes visible.
      bool is_edmft = (p.screen_type.find("edmft") != npos);
      bool is_crpa  = (p.screen_type.find("crpa") != npos);
      // eval_Pi_qdep returns the bare bubble for these before reaching either correction.
      bool plain_rpa = (p.screen_type.find("gw_edmft_rpa") != npos or
                        p.screen_type == "rpa" or p.screen_type == "rpa_k");
      std::string recipe =
          plain_rpa   ? (is_edmft ? "RPA (gw_edmft_rpa ignores the projector and Pi_imp/Pi_dc)"
                                  : "RPA")
        : (is_crpa and is_edmft) ? "RPA - Pi_active + (Pi_imp - Pi_dc)"
        : is_edmft  ? "RPA + (Pi_imp - Pi_dc)"
        : is_crpa   ? "RPA - Pi_active (cRPA)"
        : "RPA (no cRPA/EDMFT keyword in screen_type)";

      // Local corrections for the EDMFT recipes. eval_Pi_qdep only warns and silently
      // drops the correction when they are missing, so resolve the source here, exactly
      // as scr_coulomb_t::update_w does inside the self-consistent loop. Resolved before
      // the banner so that the banner can name the source, and before the Green's
      // function is read because none of it depends on G.
      std::string lp_source = "not used by this screen_type";
      bool lp_missing = false, lp_discarded = false;
      if (is_edmft) {
        if (local_polarizabilities) {
          mb_state.set_local_polarizabilities(std::move(local_polarizabilities.value()));
          local_polarizabilities.reset();
          lp_source = "provided directly as arrays";
        } else if (mb_state.read_local_polarizabilities()) {
          lp_source = "read from the checkpoint (downfold_2e)";
        } else {
          lp_source = "none found";
          lp_missing = true;
        }
      } else if (local_polarizabilities) {
        lp_source = "provided, but discarded";
        lp_discarded = true;
      }

      app_log(1, "\n[==== Post-processing: dielectric function ====]\n"
                 "  Checkpoint                    = {}\n"
                 "  Green's function              = {}/iter{}\n"
                 "  Screening type                = {}\n"
                 "  Effective polarization        = {}\n"
                 "  Bosonic projector             = {}\n"
                 "  Local polarizabilities        = {}\n"
                 "  Divergent treatment at q->0   = {}\n"
                 "  Polarization regularization   = {}\n"
                 "  Output group                  = {}/iter{}/dielectric/{}\n",
              p.filename, p.grp_name, p.iter, p.screen_type, recipe, p.proj_source,
              lp_source, p.div_treatment, p.pi_reg, p.grp_name, p.iter, p.pi_reg);
      if (lp_missing) {
        app_log(1, " [WARNING] No local polarizabilities were provided, and none were found\n"
                   "           in the checkpoint. Proceeding with a zero local correction,\n"
                   "           i.e. with the plain RPA polarization.\n");
      }
      if (lp_discarded) {
        app_log(1, " [WARNING] Local polarizabilities were provided, but screen_type = \"{}\"\n"
                   "           carries no \"edmft\" keyword, so they are discarded without\n"
                   "           being used or shape-checked.\n", p.screen_type);
      }
      if (p.pi_reg != "none") {
        // Point at the in-loop head only when the run actually stored one for this G; see
        // resolve_iteration. Kept inside a single app_log so the note cannot be split
        // apart by another rank's output.
        std::string compare_hint;
        if (p.next_head) {
          compare_hint = "\n         Compare with " + p.grp_name + "/iter" +
                         std::to_string(p.iter + 1) + "/eps_inv_head_wq to see what\n"
                         "         self-consistency adds.";
        }
        app_log(1, "\n[ NOTE ] The projection is applied to Pi built from a Green's function that\n"
                   "         was converged with its own (possibly unregularized) W. This is a\n"
                   "         one-shot regularized dielectric function from that G, not the\n"
                   "         in-loop result.{}\n", compare_hint);
      }

      mb_state.sG_tskij.emplace(
          read_greens_function(*mpi, mf.get(), p.filename, p.iter, p.grp_name));

      // Pi(tau) -> Pi(i nu) -> W(i nu), with the projection applied inside dyson_W_in_place.
      solvers::scr_coulomb_t scr(&ft, p.screen_type, p.div_treatment, p.pi_reg);
      auto dPi_tqPQ = scr.eval_Pi_qdep(mb_state, thc);
      auto [w_pgrid, w_bsize] = solvers::scr_coulomb_t::W_omega_proc_grid(
          mpi->comm.size(), mf->nqpts_ibz(), ft.nw_b(), thc.Np());
      auto dW_wqPQ = scr.tau_to_w(dPi_tqPQ, w_pgrid, w_bsize, true);
      scr.dyson_W_in_place(dW_wqPQ, thc);
      // Recorded by dyson_W_in_place, so read only after the call.
      auto const& pi_head_wq = scr.pi_head_wq();
      auto [eps_inv_head_wq, eps_inv_head_w] =
          solvers::div_utils::eps_inv_head_w(dW_wqPQ, thc, *mf, p.div_treatment);
      dW_wqPQ.reset();

      if (mpi->comm.root()) {
        h5::file file(p.filename, 'a');
        auto iter_grp = h5::group(file).open_group(p.grp_name)
                            .open_group("iter" + std::to_string(p.iter));
        auto diel_grp = iter_grp.has_subgroup("dielectric") ?
                        iter_grp.open_group("dielectric") : iter_grp.create_group("dielectric");
        // Replace this option's subgroup only; siblings stay.
        auto out = diel_grp.create_group(p.pi_reg, /*delete_if_exists=*/true);
        nda::h5_write(out, "eps_inv_head_wq", eps_inv_head_wq, false);
        nda::h5_write(out, "eps_inv_head_w", eps_inv_head_w, false);
        nda::h5_write(out, "pi_head_wq", pi_head_wq, false);
        // Present only when a projection ran; its absence records pi_regularization = "none".
        if (scr.has_delta_C()) nda::h5_write(out, "delta_C_w", scr.delta_C_w(), false);
        h5::h5_write(out, "screen_type", p.screen_type);
        h5::h5_write(out, "div_treatment", p.div_treatment);
        h5::h5_write(out, "pi_regularization", p.pi_reg);
      }
      mpi->comm.barrier();

      // Summary in the same normalization as the stored head: eps_M = 1 / (1 + head).
      if (mf->nqpts_ibz() > 1 and eps_inv_head_wq.shape(0) > 1) {
        // find_smallest_qabs returns -1 when every q-point is Gamma; nqpts_ibz > 1 does
        // not by itself exclude that.
        long iq_min = solvers::div_utils::find_smallest_qabs(mf->Qpts_ibz(), false);
        long iq_gamma = solvers::div_utils::find_gamma_index(mf->Qpts_ibz());
        if (iq_min >= 0) {
          app_log(1, "  eps_M(q_min, i nu_0) = {:.4f}\n"
                     "  eps_M(q_min, i nu_1) = {:.4f}\n"
                     "  eps_M(q->0,  i nu_1) = {:.4f}   ({} extrapolation)",
                  eps_M(eps_inv_head_wq(0, iq_min)), eps_M(eps_inv_head_wq(1, iq_min)),
                  eps_M(eps_inv_head_w(1)), p.div_treatment);
        }
        if (iq_gamma >= 0) {
          app_log(1, "  Pi_00(Gamma, i nu_1) before projection = {:.4e}\n",
                  pi_head_wq(1, iq_gamma).real());
        }
      }
      app_log(1, "Dielectric function written to {}/iter{}/dielectric/{} in {}.\n",
              p.grp_name, p.iter, p.pi_reg, p.filename);
    }

  } // anonymous namespace

  void dielectric_function(
      THC_ERI auto &thc, ptree const &pt,
      std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    std::string err = "dielectric_pproc.cpp::dielectric_function - Incorrect input - ";
    auto p  = read_params(pt, thc);
    auto mf = thc.MF();

    // The IAFT must outlive dielectric_function_impl: MBState only stores a pointer to it.
    imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(p.filename, false));
    if (p.needs_projector) {
      auto wannier_file = io::get_value<std::string>(
          pt, "wannier_file",
          err + "wannier_file. screen_type = \"" + p.screen_type + "\" needs the bosonic "
          "projector; give \"wannier_file\", or pass the projector arrays to the "
          "dielectric_function overload that takes them.");
      auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
      p.proj_source = wannier_file;
      dielectric_function_impl(thc, p, MBState(ft, p.output, mf, wannier_file, trans_home_cell),
                               std::move(local_polarizabilities));
    } else {
      p.proj_source = "none (not needed by this screen_type)";
      dielectric_function_impl(thc, p, MBState(thc.mpi(), ft, p.output),
                               std::move(local_polarizabilities));
    }
  }

  void dielectric_function(
      THC_ERI auto &thc, ptree const &pt,
      nda::array<ComplexType, 5> const &projector_ksIai,
      nda::array<long, 3> const &band_window,
      nda::array<RealType, 2> const &kpts_crys,
      std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto p  = read_params(pt, thc);
    auto mf = thc.MF();
    auto trans_home_cell = io::get_value_with_default<bool>(pt, "translate_home_cell", false);
    p.proj_source = "supplied arrays";

    // The IAFT must outlive dielectric_function_impl: MBState only stores a pointer to it.
    imag_axes_ft::IAFT ft(imag_axes_ft::read_iaft(p.filename, false));
    dielectric_function_impl(
        thc, p,
        MBState(ft, p.output, mf, projector_ksIai, band_window, kpts_crys, trans_home_cell),
        std::move(local_polarizabilities));
  }

  // template instantiation
  template void dielectric_function(
      thc_reader_t&, ptree const&,
      std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

  template void dielectric_function(
      thc_reader_t&, ptree const&,
      nda::array<ComplexType, 5> const&, nda::array<long, 3> const&,
      nda::array<RealType, 2> const&,
      std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

} // methods
