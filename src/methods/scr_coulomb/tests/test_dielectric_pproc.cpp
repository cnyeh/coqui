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

#undef NDEBUG

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "catch2/catch.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/mb_state/mb_state.hpp"
#include "methods/HF/hf_t.h"
#include "methods/GW/gw_t.h"
#include "methods/GW/g0_div_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "numerics/iter_scf/iter_scf_utils.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/scr_coulomb/dielectric_pproc.hpp"

namespace bdft_tests {

  using utils::VALUE_EQUAL;
  namespace mpi3 = boost::mpi3;
  using namespace methods;

  namespace {

    /**
     * Two-iteration Dyson GW on LiH 2x2x2 with div_treatment = gygi, written to
     * <output>.mbpt.h5. Iteration N's in-loop dielectric head is built from the G of
     * iteration N-1, so scf/iter2/eps_inv_head_* is the head of scf/iter1/G_tskij.
     */
    void make_gw_checkpoint(std::shared_ptr<utils::mpi_context_t<>> &mpi,
                            std::shared_ptr<mf::MF> &mf, thc_reader_t &thc,
                            imag_axes_ft::IAFT &ft, std::string const &output) {
      std::string filename = output + ".mbpt.h5";
      if (mpi->comm.root()) std::remove(filename.c_str());
      mpi->comm.barrier();

      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "gygi", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "gygi", "none");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi, ft, output);
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");

      scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
               /*niter=*/2, /*restart=*/false, /*conv_tol=*/1e-9, /*const_mu=*/false,
               /*input_grp=*/"scf", /*input_iter=*/-1, /*eval_thermodynamics=*/false);
      mpi->comm.barrier();
    }

    ptree pproc_params(std::string const &output, long iteration, std::string const &reg) {
      ptree pt;
      pt.put("prefix", output);
      pt.put("outdir", "./");
      pt.put("grp_name", "scf");
      pt.put("iteration", iteration);
      pt.put("pi_regularization", reg);
      pt.put("screen_type", "rpa");
      pt.put("div_treatment", "gygi");
      return pt;
    }

    struct head_data {
      nda::array<ComplexType, 2> eps_inv_head_wq;
      nda::array<ComplexType, 1> eps_inv_head_w;
      nda::array<ComplexType, 2> pi_head_wq;
      nda::array<ComplexType, 1> delta_C_w;   // size 0 when absent
      bool has_delta_C = false;
    };

    // Root-only reader; call inside `if (mpi->comm.root())`.
    head_data read_heads(std::string const &filename, std::string const &grp_path) {
      head_data d;
      h5::file file(filename, 'r');
      auto grp = h5::group(file).open_group(grp_path);
      nda::h5_read(grp, "eps_inv_head_wq", d.eps_inv_head_wq);
      nda::h5_read(grp, "eps_inv_head_w", d.eps_inv_head_w);
      nda::h5_read(grp, "pi_head_wq", d.pi_head_wq);
      d.has_delta_C = grp.has_dataset("delta_C_w");
      if (d.has_delta_C) nda::h5_read(grp, "delta_C_w", d.delta_C_w);
      return d;
    }

    /**
     * Null case: post-processing the G of iteration 1 with pi_regularization = "none"
     * must reproduce the in-loop head that iteration 2 built from that same G.
     */
    void check_null(std::shared_ptr<utils::mpi_context_t<>> &mpi, thc_reader_t &thc,
                    std::string const &output) {
      INFO("check_null");
      std::string filename = output + ".mbpt.h5";

      dielectric_function(thc, pproc_params(output, 1, "none"));

      if (mpi->comm.root()) {
        {
          // An early-stopped SCF loop would leave no reference to compare against.
          h5::file file(filename, 'r');
          REQUIRE(h5::group(file).has_subgroup("scf/iter2"));
        }
        auto ref = read_heads(filename, "scf/iter2");
        auto pp  = read_heads(filename, "scf/iter1/dielectric/none");

        REQUIRE(pp.eps_inv_head_wq.shape() == ref.eps_inv_head_wq.shape());
        REQUIRE(pp.eps_inv_head_w.shape()  == ref.eps_inv_head_w.shape());
        REQUIRE(pp.pi_head_wq.shape()      == ref.pi_head_wq.shape());
        REQUIRE_FALSE(pp.has_delta_C);

        long nw = ref.eps_inv_head_wq.shape(0), nq = ref.eps_inv_head_wq.shape(1);
        double max_d_eps = 0.0, max_d_pi = 0.0;
        for (long w = 0; w < nw; ++w) {
          // The in-loop head is extracted on the tau axis and transformed to nu; this one
          // is extracted on nu directly. The tau<->nu round trip is exact on the IR
          // sampling mesh, so the two agree to ~1e-14, not merely to IAFT precision.
          VALUE_EQUAL(pp.eps_inv_head_w(w), ref.eps_inv_head_w(w), 1e-10);
          for (long q = 0; q < nq; ++q) {
            VALUE_EQUAL(pp.eps_inv_head_wq(w, q), ref.eps_inv_head_wq(w, q), 1e-10);
            // Same G, same bubble, same head functional, no transform in between.
            VALUE_EQUAL(pp.pi_head_wq(w, q), ref.pi_head_wq(w, q), 1e-10);
            max_d_eps = std::max(max_d_eps,
                                 std::abs(pp.eps_inv_head_wq(w, q) - ref.eps_inv_head_wq(w, q)));
            max_d_pi  = std::max(max_d_pi,
                                 std::abs(pp.pi_head_wq(w, q) - ref.pi_head_wq(w, q)));
          }
        }
        app_log(2, "  check_null: max |d eps_inv_head_wq| = {:.3e}, max |d pi_head_wq| = {:.3e}",
                max_d_eps, max_d_pi);

        // Provenance strings.
        h5::file file(filename, 'r');
        auto grp = h5::group(file).open_group("scf/iter1/dielectric/none");
        std::string s;
        h5::h5_read(grp, "screen_type", s);   REQUIRE(s == "rpa");
        h5::h5_read(grp, "div_treatment", s); REQUIRE(s == "gygi");
        h5::h5_read(grp, "pi_regularization", s); REQUIRE(s == "none");
      }
      mpi->comm.barrier();
    }

    // "dynamic" on the same iteration: dC pinned to the recorded head, projection reaches W
    // at finite q and nu != 0, static sector untouched.
    void check_dynamic(std::shared_ptr<utils::mpi_context_t<>> &mpi, thc_reader_t &thc,
                       std::shared_ptr<mf::MF> &mf, std::string const &output) {
      INFO("check_dynamic");
      std::string filename = output + ".mbpt.h5";
      long iq_gamma = solvers::div_utils::find_gamma_index(mf->Qpts_ibz());
      REQUIRE(iq_gamma >= 0);
      long iq_min = solvers::div_utils::find_smallest_qabs(mf->Qpts_ibz(), false);
      REQUIRE(iq_min >= 0);
      REQUIRE(iq_min != iq_gamma);

      // check_null already produced scf/iter1/dielectric/none; add the projected run.
      dielectric_function(thc, pproc_params(output, 1, "dynamic"));

      if (mpi->comm.root()) {
        {
          // Written by check_null: reordering the helpers must fail with a message here
          // rather than with a raw h5 exception on root while the others wait.
          h5::file file(filename, 'r');
          REQUIRE(h5::group(file).has_subgroup("scf/iter1/dielectric/none"));
        }
        auto none = read_heads(filename, "scf/iter1/dielectric/none");
        auto dyn  = read_heads(filename, "scf/iter1/dielectric/dynamic");
        long nw = dyn.pi_head_wq.shape(0), nq = dyn.pi_head_wq.shape(1);
        REQUIRE(nw > 1);
        REQUIRE(dyn.has_delta_C);
        REQUIRE(dyn.delta_C_w.shape(0) == nw);

        // The recorded head is the one BEFORE projection, so it is option-independent.
        double max_d_pi = 0.0;
        for (long w = 0; w < nw; ++w)
          for (long q = 0; q < nq; ++q) {
            VALUE_EQUAL(dyn.pi_head_wq(w, q), none.pi_head_wq(w, q), 1e-12);
            max_d_pi = std::max(max_d_pi, std::abs(dyn.pi_head_wq(w, q) - none.pi_head_wq(w, q)));
          }

        // "dynamic": static sector untouched (literal zero); dC(n>=1) = Re Pi_00(Gamma, i nu_n).
        REQUIRE(dyn.delta_C_w(0) == ComplexType(0.0));
        double dc_scale = 0.0;
        for (long w = 1; w < nw; ++w) {
          VALUE_EQUAL(dyn.delta_C_w(w),
                      ComplexType(dyn.pi_head_wq(w, iq_gamma).real(), 0.0), 1e-14);
          dc_scale = std::max(dc_scale, std::abs(dyn.delta_C_w(w)));
        }
        REQUIRE(dc_scale > 1e-8);   // a dressed G violates the Ward identity: something to remove

        // The projection must have reached W at finite q and finite frequency ...
        double d1 = std::abs(dyn.eps_inv_head_wq(1, iq_min) - none.eps_inv_head_wq(1, iq_min));
        app_log(2, "  check_dynamic: |d eps_inv_head_wq(nu_1, q_min)| = {:.3e}, "
                   "max_n |dC| = {:.3e}, max |d pi_head_wq| = {:.3e}", d1, dc_scale, max_d_pi);
        REQUIRE(d1 > 1e-8);
        // ... and must have left the static sector alone.
        for (long q = 0; q < nq; ++q)
          VALUE_EQUAL(dyn.eps_inv_head_wq(0, q), none.eps_inv_head_wq(0, q), 1e-12);
      }
      mpi->comm.barrier();
    }

    /**
     * "insulator" and "extrapolate" against "dynamic". All three apply the same rank-one
     * projection; they differ ONLY in the nu = 0 entry of dC, and that entry is the entire
     * content of the choice, so it is the only thing worth asserting.
     */
    void check_options(std::shared_ptr<utils::mpi_context_t<>> &mpi, thc_reader_t &thc,
                       std::shared_ptr<mf::MF> &mf, std::string const &output) {
      INFO("check_options");
      std::string filename = output + ".mbpt.h5";
      long iq_gamma = solvers::div_utils::find_gamma_index(mf->Qpts_ibz());
      REQUIRE(iq_gamma >= 0);

      // These land beside the "none"/"dynamic" subgroups that check_groups later reruns;
      // an option writes only its own subgroup, so the two helpers do not interact.
      dielectric_function(thc, pproc_params(output, 1, "insulator"));
      dielectric_function(thc, pproc_params(output, 1, "extrapolate"));

      if (mpi->comm.root()) {
        auto none = read_heads(filename, "scf/iter1/dielectric/none");
        auto dyn  = read_heads(filename, "scf/iter1/dielectric/dynamic");
        auto ins  = read_heads(filename, "scf/iter1/dielectric/insulator");
        auto ext  = read_heads(filename, "scf/iter1/dielectric/extrapolate");
        REQUIRE_FALSE(none.has_delta_C);   // the absence of dC is what records "none"
        REQUIRE(dyn.has_delta_C);
        REQUIRE(ins.has_delta_C);
        REQUIRE(ext.has_delta_C);

        long nw = none.pi_head_wq.shape(0), nq = none.pi_head_wq.shape(1);
        REQUIRE(nw > 1);
        REQUIRE(ins.delta_C_w.shape(0) == nw);
        REQUIRE(ext.delta_C_w.shape(0) == nw);

        // The head is recorded BEFORE the projection, so all four options share it.
        for (long w = 0; w < nw; ++w)
          for (long q = 0; q < nq; ++q) {
            VALUE_EQUAL(ins.pi_head_wq(w, q), none.pi_head_wq(w, q), 1e-12);
            VALUE_EQUAL(ext.pi_head_wq(w, q), none.pi_head_wq(w, q), 1e-12);
          }

        double pi0 = none.pi_head_wq(0, iq_gamma).real();   // Re Pi_00(Gamma, nu = 0)
        double d_zero = std::abs(ext.delta_C_w(0));                            // vs "dynamic"
        double d_stat = std::abs(ext.delta_C_w(0) - ComplexType(pi0, 0.0));    // vs "insulator"
        app_log(2, "  check_options: dC(0) none/dynamic=0, insulator={:.6e}, "
                   "extrapolate={:.6e}, Re Pi_00(G,0)={:.6e}\n"
                   "                 separations: |dC_ext(0)| = {:.3e}, "
                   "|dC_ext(0) - Re Pi_00(G,0)| = {:.3e}",
                ins.delta_C_w(0).real(), ext.delta_C_w(0).real(), pi0, d_zero, d_stat);

        // "insulator" projects the static sector too: dC(0) is the raw static head.
        VALUE_EQUAL(ins.delta_C_w(0), ComplexType(pi0, 0.0), 1e-14);

        // "extrapolate" replaces dC(0) by the nu -> 0 limit of the n >= 1 shifts, so it is
        // neither what "dynamic" uses (0) nor what "insulator" uses (the raw static head).
        // The second separation is the static-only excess that "extrapolate" exists to
        // retain, and on a gapped insulator at beta = 1000 it is legitimately tiny: 7.5e-08
        // here, against a violation of 1.9e-01. It is still ~9 orders of magnitude above
        // double round-off on a quantity of that scale, so it is a resolved number and not
        // noise -- but the margin over this gate is only ~7.5x, so a change of fixture,
        // beta or extrap_npts may need it revisited rather than merely relaxed.
        const double gate = 1e-8;
        REQUIRE(d_zero > gate);
        REQUIRE(d_stat > gate);

        // Away from nu = 0 the three are the same projection, to round-off.
        for (long w = 1; w < nw; ++w) {
          VALUE_EQUAL(ins.delta_C_w(w), dyn.delta_C_w(w), 1e-14);
          VALUE_EQUAL(ext.delta_C_w(w), dyn.delta_C_w(w), 1e-14);
        }
      }
      mpi->comm.barrier();
    }

    // Group hygiene: rerun replaces only its own option subgroup; in-loop data untouched.
    void check_groups(std::shared_ptr<utils::mpi_context_t<>> &mpi, thc_reader_t &thc,
                      std::string const &output) {
      INFO("check_groups");
      std::string filename = output + ".mbpt.h5";

      // Snapshot of the in-loop data that must survive untouched, and of the two option
      // subgroups that the reruns must reproduce.
      head_data before1, before2, before_none, before_dyn;
      if (mpi->comm.root()) {
        before1 = read_heads(filename, "scf/iter1");
        before2 = read_heads(filename, "scf/iter2");
        before_none = read_heads(filename, "scf/iter1/dielectric/none");
        before_dyn  = read_heads(filename, "scf/iter1/dielectric/dynamic");
        REQUIRE(before_dyn.has_delta_C);
      }
      mpi->comm.barrier();

      // Rerun of an existing option must replace, not duplicate or fail; the sibling stays.
      dielectric_function(thc, pproc_params(output, 1, "dynamic"));
      dielectric_function(thc, pproc_params(output, 1, "none"));

      if (mpi->comm.root()) {
        {
          h5::file file(filename, 'r');
          auto iter_grp = h5::group(file).open_group("scf/iter1");
          REQUIRE(iter_grp.has_subgroup("dielectric"));
          auto diel = iter_grp.open_group("dielectric");
          REQUIRE(diel.has_subgroup("none"));
          REQUIRE(diel.has_subgroup("dynamic"));
          for (auto const& name : {"none", "dynamic"}) {
            auto g = diel.open_group(name);
            REQUIRE(g.has_dataset("eps_inv_head_wq"));
            REQUIRE(g.has_dataset("eps_inv_head_w"));
            REQUIRE(g.has_dataset("pi_head_wq"));
            REQUIRE(g.has_dataset("screen_type"));
            REQUIRE(g.has_dataset("div_treatment"));
            REQUIRE(g.has_dataset("pi_regularization"));
          }
          REQUIRE_FALSE(diel.open_group("none").has_dataset("delta_C_w"));
          REQUIRE(diel.open_group("dynamic").has_dataset("delta_C_w"));
          // The in-loop iteration group must not have grown a stray dataset.
          REQUIRE_FALSE(iter_grp.has_dataset("delta_C_w"));
        }
        // nda::operator== is element-wise exact equality reduced to one bool: "bit-identical".
        // The in-loop groups are never written by the driver, and the driver itself is
        // deterministic, so exact equality is the right predicate everywhere below.
        {
          INFO("in-loop scf/iter1");
          auto after1 = read_heads(filename, "scf/iter1");
          REQUIRE(after1.eps_inv_head_wq == before1.eps_inv_head_wq);
          REQUIRE(after1.eps_inv_head_w  == before1.eps_inv_head_w);
          REQUIRE(after1.pi_head_wq      == before1.pi_head_wq);
        }
        {
          INFO("in-loop scf/iter2");
          auto after2 = read_heads(filename, "scf/iter2");
          REQUIRE(after2.eps_inv_head_wq == before2.eps_inv_head_wq);
          REQUIRE(after2.eps_inv_head_w  == before2.eps_inv_head_w);
          REQUIRE(after2.pi_head_wq      == before2.pi_head_wq);
        }
        {
          INFO("rerun of scf/iter1/dielectric/none");
          auto after_none = read_heads(filename, "scf/iter1/dielectric/none");
          REQUIRE_FALSE(after_none.has_delta_C);
          REQUIRE(after_none.eps_inv_head_wq == before_none.eps_inv_head_wq);
          REQUIRE(after_none.eps_inv_head_w  == before_none.eps_inv_head_w);
          REQUIRE(after_none.pi_head_wq      == before_none.pi_head_wq);
        }
        {
          INFO("rerun of scf/iter1/dielectric/dynamic");
          auto after_dyn = read_heads(filename, "scf/iter1/dielectric/dynamic");
          REQUIRE(after_dyn.has_delta_C);
          REQUIRE(after_dyn.eps_inv_head_wq == before_dyn.eps_inv_head_wq);
          REQUIRE(after_dyn.eps_inv_head_w  == before_dyn.eps_inv_head_w);
          REQUIRE(after_dyn.pi_head_wq      == before_dyn.pi_head_wq);
          REQUIRE(after_dyn.delta_C_w       == before_dyn.delta_C_w);
        }
      }
      mpi->comm.barrier();
    }

  } // anonymous namespace

  /**
   * One two-iteration GW checkpoint shared by every check below: the SCF loop dominates
   * the cost of this test, so each new case is a helper call, not a TEST_CASE of its own
   * (a Catch2 SECTION would re-run the whole body, including the loop).
   */
  TEST_CASE("dielectric_pproc", "[methods][scr_coulomb][dielectric]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*12, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    REQUIRE(thc.has_basis_head());
    REQUIRE(mf->nqpts_ibz() > 1);

    const std::string output = "dielectric_pproc";
    const std::string filename = output + ".mbpt.h5";
    make_gw_checkpoint(mpi, mf, thc, ft, output);

    check_null(mpi, thc, output);
    check_dynamic(mpi, thc, mf, output);
    check_options(mpi, thc, mf, output);
    check_groups(mpi, thc, output);

    if (mpi->comm.root()) std::remove(filename.c_str());
    mpi->comm.barrier();
  }

} // bdft_tests
