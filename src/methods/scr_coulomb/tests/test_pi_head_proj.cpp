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

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "itertools/itertools.hpp"

#include "utilities/test_common.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "hamiltonian/one_body_hamiltonian.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/eri_utils.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/scr_coulomb/scr_coulomb_t.h"

namespace bdft_tests {

  using utils::VALUE_EQUAL;
  namespace mpi3 = boost::mpi3;
  using namespace methods;

  TEST_CASE("pi_head_norm", "[methods][scr_coulomb][pi_head]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*12, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));

    // sigma and N are properties of the THC auxiliary basis, evaluated once when the head
    // vectors are built; scr_coulomb_t only reads them.
    REQUIRE(thc.has_basis_head());
    auto sigma = thc.basis_head_overlap();
    auto Nq    = thc.basis_head_norm();

    long nq = mf->nqpts_ibz();
    REQUIRE(sigma.shape(0) == nq);
    REQUIRE(Nq.shape(0) == nq);

    // sigma(q) = sum_P Bbar_P conj(B_P) is the G = 0 element of the projector onto the
    // auxiliary basis, so it must sit at unity up to the ISDF leakage at every q, and
    // N(q) = |sigma(q)|^2. The transposed sum is NOT near unity (8.75e-01 off on this
    // fixture); it is computed here only to document that the two are distinguishable, so
    // a future "convention" cannot be mistaken for a free choice.
    auto B    = thc.basis_head();
    auto Bbar = thc.basis_bar_head();
    for (long q = 0; q < nq; ++q) {
      ComplexType sig_check(0.0), sig_transposed(0.0);
      for (long P = 0; P < thc.Np(); ++P) {
        sig_check      += Bbar(q, P) * std::conj(B(q, P));
        sig_transposed += Bbar(q, P) * B(q, P);
      }
      app_log(2, "  q = {}: sigma = ({:.8f}, {:.8f}), N = {:.8f}, |sum Bbar_P B_P| = {:.3e}",
              q, sigma(q).real(), sigma(q).imag(), Nq(q), std::abs(sig_transposed));
      VALUE_EQUAL(sigma(q), sig_check, 1e-12);
      REQUIRE(std::abs(sigma(q) - ComplexType(1.0)) < 0.05);
      VALUE_EQUAL(Nq(q), std::norm(sigma(q)), 1e-10);
    }
  }

  // Build a deterministic Hermitian Pi_PQ(q,w) with a nonzero head at every q and w.
  // The array is built on MPI_COMM_SELF with a {1,1,1,1} process grid, so every rank
  // holds the whole thing and the test's index arithmetic stays simple.
  static auto make_synthetic_Pi(std::shared_ptr<mf::MF> &mf,
                                mpi3::communicator &comm,
                                long nw, long Np) {
    using local_Array_4D_t = nda::array<ComplexType, 4>;
    using math::nda::make_distributed_array;

    long nq = mf->nqpts_ibz();
    auto dPi = make_distributed_array<local_Array_4D_t>(
        comm, {1, 1, 1, 1}, {nw, nq, Np, Np}, {1, 1, 1, 1});

    auto Pi = dPi.local();
    for (long w = 0; w < nw; ++w) {
      for (long q = 0; q < nq; ++q) {
        for (long P = 0; P < Np; ++P) {
          for (long Q = 0; Q < Np; ++Q) {
            double re = std::cos(0.7 * P + 1.3 * Q + 0.11 * q) / (1.0 + w);
            double im = std::sin(0.7 * P - 1.3 * Q + 0.11 * q) / (1.0 + w);
            Pi(w, q, P, Q) = ComplexType(re, im);
          }
        }
        // Hermitize: Pi <- (Pi + Pi^dagger)/2
        for (long P = 0; P < Np; ++P) {
          for (long Q = P; Q < Np; ++Q) {
            auto h = 0.5 * (Pi(w, q, P, Q) + std::conj(Pi(w, q, Q, P)));
            Pi(w, q, P, Q) = h;
            Pi(w, q, Q, P) = std::conj(h);
          }
        }
      }
    }
    return dPi;
  }

  // Spread `np` ranks over all four axes of (w, q, P, Q) so that a test exercises
  // w-pools, q-pools and a genuine P/Q block split at once. Factors that will not fit on
  // an axis fall through to the next one; P and Q are always large enough to absorb them.
  static std::array<long, 4> spread_pgrid(long np, std::array<long, 4> ext) {
    std::array<long, 4> pg = {1, 1, 1, 1};
    int i = 0;
    for (long f = 2; np > 1; ) {
      if (np % f != 0) { ++f; continue; }
      int placed = -1;
      for (int t = 0; t < 4; ++t) {
        int ax = (i + t) % 4;
        if (ext[ax] / pg[ax] >= f) { placed = ax; i = (i + t + 1) % 4; break; }
      }
      utils::check(placed >= 0,
                   "test_pi_head_proj::spread_pgrid: cannot place factor {} on any axis.", f);
      pg[placed] *= f;
      np /= f;
    }
    return pg;
  }

  TEST_CASE("pi_head_projection", "[methods][scr_coulomb][pi_head]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*12, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    long Np = thc.Np();
    long nw = 4;
    long iq_gamma = solvers::div_utils::find_gamma_index(mf->Qpts_ibz());
    REQUIRE(iq_gamma >= 0);

    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0", "dynamic");

    auto& self_comm = mpi3::environment::get_self_instance();
    auto dPi = make_synthetic_Pi(mf, self_comm, nw, Np);
    // Snapshot the input for the wing-invariance test.
    nda::array<ComplexType, 4> Pi_in(dPi.local());

    auto head_in = solvers::div_utils::head_from_prod_basis(dPi, thc, false);
    auto dC = scr_eri.regularize_Pi_head(dPi, thc, head_in);
    auto head_out = solvers::div_utils::head_from_prod_basis(dPi, thc, false);

    SECTION("head is zeroed for nu_n != 0 and untouched at nu_n = 0") {
      // The synthetic Pi must actually have a head to remove, or the test is vacuous.
      REQUIRE(std::abs(head_in(1, iq_gamma)) > 1e-6);

      for (long w = 1; w < nw; ++w) {
        VALUE_EQUAL(head_out(w, iq_gamma), ComplexType(0.0), 1e-10);
        VALUE_EQUAL(dC(w), head_in(w, iq_gamma), 1e-10);
      }
      // "dynamic" leaves the static sector alone.
      VALUE_EQUAL(head_out(0, iq_gamma), head_in(0, iq_gamma), 1e-10);
      VALUE_EQUAL(dC(0), ComplexType(0.0), 1e-14);
    }

    SECTION("idempotent") {
      auto dC2 = scr_eri.regularize_Pi_head(dPi, thc, head_out);
      for (long w = 0; w < nw; ++w) {
        VALUE_EQUAL(dC2(w), ComplexType(0.0), 1e-10);
      }
    }

    SECTION("head at finite q shifts by exactly dC") {
      for (long q = 0; q < mf->nqpts_ibz(); ++q) {
        for (long w = 1; w < nw; ++w) {
          VALUE_EQUAL(head_out(w, q), head_in(w, q) - dC(w), 1e-10);
        }
      }
    }

    SECTION("wings are invariant") {
      REQUIRE(mf->nqpts_ibz() > 1);
      long q = (iq_gamma == 0) ? 1 : 0;

      auto B    = thc.basis_head();
      auto Bbar = thc.basis_bar_head();
      auto dotc_ = [&](auto const &a, auto const &b) {
        ComplexType s(0.0);
        for (long P = 0; P < Np; ++P) s += std::conj(a(P)) * b(P);
        return s;
      };

      // Orthonormalize the PAIR {Bbar, conj(Bbar)} first -- one extra pass against v1
      // does NOT converge to the intersection of two non-orthogonal complements.
      // Then take u = the part of B(q) orthogonal to that span, normalized: choosing
      // B(q) itself (not an arbitrary vector) maximizes |u^H B| / |u|, which is what
      // sets the discriminating signal, and normalizing keeps VALUE_EQUAL on its
      // relative branch rather than its 1e-10 absolute floor.
      nda::array<ComplexType, 1> e1(Np), e2(Np), u(Np);
      for (long P = 0; P < Np; ++P) {
        e1(P) = Bbar(q, P);
        e2(P) = std::conj(Bbar(q, P));
        u(P)  = B(q, P);
      }
      e1 /= std::sqrt(dotc_(e1, e1).real());
      e2 -= dotc_(e1, e2) * e1;
      e2 /= std::sqrt(dotc_(e2, e2).real());
      u  -= dotc_(e1, u) * e1 + dotc_(e2, u) * e2;
      u  /= std::sqrt(dotc_(u, u).real());

      // u is made orthogonal to both Bbar and conj(Bbar): the kernel's direction is
      // conj(Bbar), and the second orthogonality keeps the section insensitive to a
      // transposed direction as well, so it isolates the wing statement.
      VALUE_EQUAL(dotc_(u, e1), ComplexType(0.0), 1e-10);
      VALUE_EQUAL(dotc_(u, e2), ComplexType(0.0), 1e-10);

      ComplexType uB(0.0);
      for (long P = 0; P < Np; ++P) uB += std::conj(u(P)) * B(q, P);
      auto Nq = thc.basis_head_norm();

      auto quad = [&](const nda::array<ComplexType, 4> &M, long w) {
        ComplexType s(0.0);
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q)
            s += std::conj(u(P)) * M(w, q, P, Q) * u(Q);
        return s;
      };

      nda::array<ComplexType, 4> Pi_out(dPi.local());
      for (long w = 1; w < nw; ++w) {
        auto ref = quad(Pi_in, w);
        // Assert the DISCRIMINATOR, not a proxy for it: this is exactly how much the
        // direct-vector variant would shift u^dag Pi u. Below ~100x the tolerance this
        // section cannot tell the two choices apart, and silently stops testing the
        // central design decision of the feature.
        double contamination = std::abs(dC(w)) * std::norm(uB) / Nq(q);
        double tol           = std::max(1e-10, 1e-8 * std::abs(ref));
        app_log(2, "  wing discriminator w={}: direct-vector shift {:.3e}, tol {:.3e}, "
                   "margin {:.1f}x", w, contamination, tol, contamination / tol);
        REQUIRE(contamination > 100.0 * tol);

        VALUE_EQUAL(quad(Pi_out, w), ref, 1e-10);
      }
    }

    SECTION("distributed process grid reproduces the replicated result") {
      // The COMM_SELF case above keeps the index arithmetic simple but leaves the code
      // path that actually ships untested: local_range indexing, Bbar(q, P_rng) row
      // slicing, and global-vs-local w/q indexing under real w-pools and q-pools.
      long nq = mf->nqpts_ibz();
      auto pgrid = spread_pgrid(mpi->comm.size(), {nw, nq, Np, Np});
      app_log(2, "  distributed pgrid = ({}, {}, {}, {})",
              pgrid[0], pgrid[1], pgrid[2], pgrid[3]);

      auto dPi_d = math::nda::make_distributed_array<nda::array<ComplexType, 4> >(
          mpi->comm, pgrid, {nw, nq, Np, Np}, {1, 1, 1, 1});

      auto w_rng = dPi_d.local_range(0);
      auto q_rng = dPi_d.local_range(1);
      auto P_rng = dPi_d.local_range(2);
      auto Q_rng = dPi_d.local_range(3);
      auto loc   = dPi_d.local();
      for (auto [iw, w] : itertools::enumerate(w_rng))
        for (auto [iq, q] : itertools::enumerate(q_rng))
          for (auto [iP, P] : itertools::enumerate(P_rng))
            for (auto [iQ, Q] : itertools::enumerate(Q_rng))
              loc(iw, iq, iP, iQ) = Pi_in(w, q, P, Q);

      // Same unprojected head as the replicated build, so the two runs see the same dC.
      auto head_in_d = solvers::div_utils::head_from_prod_basis(dPi_d, thc, false);
      for (long w = 0; w < nw; ++w)
        for (long q = 0; q < nq; ++q)
          VALUE_EQUAL(head_in_d(w, q), head_in(w, q), 1e-10);

      auto dC_d = scr_eri.regularize_Pi_head(dPi_d, thc, head_in_d);
      for (long w = 0; w < nw; ++w) VALUE_EQUAL(dC_d(w), dC(w), 1e-10);

      // Element-wise against the already-projected replicated array: this is what pins
      // down the local_range / row-slicing arithmetic.
      auto Pi_ref = dPi.local();
      for (auto [iw, w] : itertools::enumerate(w_rng))
        for (auto [iq, q] : itertools::enumerate(q_rng))
          for (auto [iP, P] : itertools::enumerate(P_rng))
            for (auto [iQ, Q] : itertools::enumerate(Q_rng))
              VALUE_EQUAL(loc(iw, iq, iP, iQ), Pi_ref(w, q, P, Q), 1e-10);

      auto head_out_d = solvers::div_utils::head_from_prod_basis(dPi_d, thc, false);
      for (long w = 0; w < nw; ++w)
        for (long q = 0; q < nq; ++q)
          VALUE_EQUAL(head_out_d(w, q), head_out(w, q), 1e-10);
    }

    SECTION("hermiticity is preserved") {
      auto Pi = dPi.local();
      for (long w = 0; w < nw; ++w)
        for (long P = 0; P < Np; ++P)
          for (long Q = 0; Q < Np; ++Q)
            VALUE_EQUAL(Pi(w, iq_gamma, P, Q), std::conj(Pi(w, iq_gamma, Q, P)), 1e-10);
    }
  }

  /**
   * The test that pins the rank-one direction against the Coulomb matrix. dyson_W_in_place
   * leaves W_c = sum_{n>=1} (Z Pi)^n Z in place, and eval_eps_inv_q reports
   * eps^{-1} - 1 = (q^2 Omega / 4 pi) Bbar^T W_c conj(Bbar). For a pure head-channel
   * polarization Pi = c u u^dag / N the series sums in closed form:
   *
   *     eps^{-1} - 1 = x / (1 - x),   x = c v_q * [ (Bbar^T Z u)(u^dag Z conj(Bbar)) / (v_q^2 N) ],
   *
   * with v_q = 4 pi / (q^2 Omega). For u = conj(Bbar) and N = |sigma|^2 the bracket is
   * 1 - O(lambda), so the head of W follows c exactly as a physical head change must. For
   * the transposed direction u = Bbar the bracket is |B^T Bbar|^2 / N ~ 1e-12: W does not
   * respond at all, although the transposed head functional would report that Pi's head is
   * c. sigma cannot tell these two apart (it is the same equation conjugated); only the
   * coupling through Z can, which is why this section exists. Gamma is skipped: the
   * prefactor q^2 makes eps^{-1} - 1 vanish there identically.
   */
  TEST_CASE("pi_head_z_coupling", "[methods][scr_coulomb][pi_head]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*12, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    long Np = thc.Np();
    long nq = mf->nqpts_ibz();
    long nw = 2;
    long iq_gamma = solvers::div_utils::find_gamma_index(mf->Qpts_ibz());
    REQUIRE(iq_gamma >= 0);
    REQUIRE(nq > 1);

    auto B    = thc.basis_head();
    auto Bbar = thc.basis_bar_head();
    nda::array<ComplexType, 1> sigma(nq);
    sigma() = ComplexType(0.0);
    for (long q = 0; q < nq; ++q)
      for (long P = 0; P < Np; ++P) sigma(q) += Bbar(q, P) * std::conj(B(q, P));

    // Strength of the synthetic head per frequency, in the units of Pi_PQ. Small enough
    // that x = c v_q stays well inside the radius of convergence of the Dyson series.
    auto c_w = [](long w) { return -0.1 * (1.0 + 0.5 * w); };

    auto [pgrid, bsize] = solvers::scr_coulomb_t::W_omega_proc_grid(
        mpi->comm.size(), nq, 2*nw, Np);

    auto run = [&](bool transposed) {
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0", "none");
      auto dPi = math::nda::make_distributed_array<nda::array<ComplexType, 4> >(
          mpi->comm, pgrid, {nw, nq, Np, Np}, bsize);
      auto w_rng = dPi.local_range(0);
      auto q_rng = dPi.local_range(1);
      auto P_rng = dPi.local_range(2);
      auto Q_rng = dPi.local_range(3);
      auto loc   = dPi.local();
      for (auto [iw, w] : itertools::enumerate(w_rng))
        for (auto [iq, q] : itertools::enumerate(q_rng)) {
          double N = std::norm(sigma(q));
          for (auto [iP, P] : itertools::enumerate(P_rng))
            for (auto [iQ, Q] : itertools::enumerate(Q_rng)) {
              ComplexType uP = transposed ? Bbar(q, P) : std::conj(Bbar(q, P));
              ComplexType uQ = transposed ? Bbar(q, Q) : std::conj(Bbar(q, Q));
              loc(iw, iq, iP, iQ) = c_w(w) * uP * std::conj(uQ) / N;
            }
        }
      scr_eri.dyson_W_in_place(dPi, thc);
      return solvers::div_utils::eval_eps_inv_q(dPi, thc, *mf);   // (nw, nq): eps^{-1} - 1
    };

    auto e_dual = run(false);
    auto e_transposed = run(true);

    double vol = mf->volume();
    for (long q = 0; q < nq; ++q) {
      if (q == iq_gamma) continue;
      auto qpt = mf->Qpts_ibz(q);
      double q2 = qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2);
      double vq = 4.0 * 3.14159265358979323846 / (q2 * vol);
      for (long w = 0; w < nw; ++w) {
        double x = c_w(w) * vq;
        double expected = x / (1.0 - x);
        app_log(2, "  q = {} w = {}: eps^-1 - 1 = {:+.6e} (dual), {:+.3e} (transposed), "
                   "expected {:+.6e}", q, w, e_dual(w, q).real(), e_transposed(w, q).real(),
                expected);
        // ISDF leakage on this fixture is ~6e-6; 1e-3 leaves room for any reasonable fit.
        REQUIRE(std::abs(e_dual(w, q) - ComplexType(expected)) < 1e-3 * std::abs(expected));
        // The transposed direction must be invisible to W: measured ~1e-12 of the signal.
        REQUIRE(std::abs(e_transposed(w, q)) < 1e-6 * std::abs(expected));
      }
    }
  }

  TEST_CASE("pi_head_none_is_null", "[methods][scr_coulomb][pi_head]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*12, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    long Np = thc.Np();
    long nq = mf->nqpts_ibz();
    long nw = 4;
    long iq_gamma = solvers::div_utils::find_gamma_index(mf->Qpts_ibz());
    REQUIRE(iq_gamma >= 0);

    // dyson_W_in_place splits thc.mpi()->comm by the array's (w, q) origins and asserts
    // that the resulting sub-communicators match the P/Q grid, so it needs a genuine
    // production grid over the WORLD communicator -- not the COMM_SELF {1,1,1,1} grid the
    // sections above use. W_omega_proc_grid is the very partitioner the SCF path feeds it;
    // nw_b = 2*nw makes its internal half-mesh equal to our nw.
    auto [pgrid, bsize] = solvers::scr_coulomb_t::W_omega_proc_grid(
        mpi->comm.size(), nq, 2*nw, Np);
    app_log(2, "  dyson pgrid = ({}, {}, {}, {}), bsize = ({}, {}, {}, {})",
            pgrid[0], pgrid[1], pgrid[2], pgrid[3],
            bsize[0], bsize[1], bsize[2], bsize[3]);

    // Replicated reference input, so both runs start from bit-identical data.
    auto& self_comm = mpi3::environment::get_self_instance();
    auto dPi_ref = make_synthetic_Pi(mf, self_comm, nw, Np);
    nda::array<ComplexType, 4> Pi_in(dPi_ref.local());
    auto head_ref = solvers::div_utils::head_from_prod_basis(dPi_ref, thc, false);
    // Non-vacuous only if the synthetic Pi actually has a Gamma head to remove.
    REQUIRE(std::abs(head_ref(1, iq_gamma)) > 1e-6);

    // Run the full Dyson solve on the distributed grid and gather W on every rank. The
    // grid tiles (w, q, P, Q) exactly once, so a zero-padded all_reduce reconstructs it.
    auto run = [&](std::string reg,
                   nda::array<ComplexType, 4> &W,
                   nda::array<ComplexType, 2> &head,
                   bool &has_dC,
                   nda::array<ComplexType, 1> &dC) {
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0", reg);
      auto dPi = math::nda::make_distributed_array<nda::array<ComplexType, 4> >(
          mpi->comm, pgrid, {nw, nq, Np, Np}, bsize);
      auto w_rng = dPi.local_range(0);
      auto q_rng = dPi.local_range(1);
      auto P_rng = dPi.local_range(2);
      auto Q_rng = dPi.local_range(3);
      auto loc   = dPi.local();
      for (auto [iw, w] : itertools::enumerate(w_rng))
        for (auto [iq, q] : itertools::enumerate(q_rng))
          for (auto [iP, P] : itertools::enumerate(P_rng))
            for (auto [iQ, Q] : itertools::enumerate(Q_rng))
              loc(iw, iq, iP, iQ) = Pi_in(w, q, P, Q);

      scr_eri.dyson_W_in_place(dPi, thc);

      W() = ComplexType(0.0);
      for (auto [iw, w] : itertools::enumerate(w_rng))
        for (auto [iq, q] : itertools::enumerate(q_rng))
          for (auto [iP, P] : itertools::enumerate(P_rng))
            for (auto [iQ, Q] : itertools::enumerate(Q_rng))
              W(w, q, P, Q) = loc(iw, iq, iP, iQ);
      mpi->comm.all_reduce_in_place_n(W.data(), W.size(), std::plus<>{});

      head   = scr_eri.pi_head_wq();
      has_dC = scr_eri.has_delta_C();
      if (has_dC) dC = scr_eri.delta_C_w();
    };

    nda::array<ComplexType, 4> W_none(nw, nq, Np, Np), W_dyn(nw, nq, Np, Np),
                               W_ins(nw, nq, Np, Np);
    nda::array<ComplexType, 2> head_none(nw, nq), head_dyn(nw, nq), head_ins(nw, nq);
    nda::array<ComplexType, 1> dC_none(nw), dC_dyn(nw), dC_ins(nw);
    bool has_dC_none = true, has_dC_dyn = false, has_dC_ins = false;

    run("none",      W_none, head_none, has_dC_none, dC_none);
    run("dynamic",   W_dyn,  head_dyn,  has_dC_dyn,  dC_dyn);
    run("insulator", W_ins,  head_ins,  has_dC_ins,  dC_ins);

    SECTION("the head is recorded even when the projection is off") {
      // The prologue evaluates the head unconditionally -- it is the diagnostic that tells
      // a user whether the projection is needed at all.
      for (long w = 0; w < nw; ++w)
        for (long q = 0; q < nq; ++q)
          VALUE_EQUAL(head_none(w, q), head_ref(w, q), 1e-10);
    }

    SECTION("\"none\" applies no projection") {
      REQUIRE_FALSE(has_dC_none);
    }

    SECTION("the recorded head is the UNPROJECTED one under \"dynamic\" too") {
      // head_dyn is captured before regularize_Pi_head() runs, so it must match the
      // "none" run exactly; if the prologue were reordered this would move.
      for (long w = 0; w < nw; ++w)
        for (long q = 0; q < nq; ++q)
          VALUE_EQUAL(head_dyn(w, q), head_none(w, q), 1e-10);
    }

    SECTION("\"dynamic\" projects with dC read off the Gamma column") {
      REQUIRE(has_dC_dyn);
      VALUE_EQUAL(dC_dyn(0), ComplexType(0.0), 1e-14);
      for (long w = 1; w < nw; ++w)
        VALUE_EQUAL(dC_dyn(w), ComplexType(head_ref(w, iq_gamma).real(), 0.0), 1e-10);
    }

    SECTION("W moves under \"dynamic\" but its static sector does not") {
      // Scale of W, and the shift split three ways: nu_n != 0 at finite q, nu_n != 0 at
      // Gamma, and the whole nu_n = 0 sector.
      double w_scale = 0.0, d_finite_q = 0.0, d_gamma = 0.0, d_static = 0.0;
      for (long w = 0; w < nw; ++w) {
        for (long q = 0; q < nq; ++q) {
          for (long P = 0; P < Np; ++P) {
            for (long Q = 0; Q < Np; ++Q) {
              double d = std::abs(W_dyn(w, q, P, Q) - W_none(w, q, P, Q));
              w_scale = std::max(w_scale, std::abs(W_none(w, q, P, Q)));
              if (w == 0)                 d_static   = std::max(d_static, d);
              else if (q == iq_gamma)     d_gamma    = std::max(d_gamma, d);
              else                        d_finite_q = std::max(d_finite_q, d);
            }
          }
        }
      }
      app_log(2, "  max|W| = {:.6e}; max |W_dyn - W_none|: nu_n != 0 finite q -> {:.6e} "
                 "({:.2e} relative), nu_n != 0 at Gamma -> {:.6e}, nu_n = 0 -> {:.6e}",
              w_scale, d_finite_q, d_finite_q / w_scale, d_gamma, d_static);

      // The projection must actually reach W. The claim is relative, not absolute: the
      // synthetic Pi drives a W of only ~4e-7 here (W ~ Z Pi Z and the THC Coulomb is
      // small), so an absolute 1e-8 floor would be a statement about the fixture's scale
      // rather than about the projection. The measured ratio is ~3e-3.
      REQUIRE(w_scale > 0.0);
      // Measured 2.68e-03 relative; 1e-4 keeps a ~27x margin while still noticing a large
      // regression in the projection's magnitude. A 1e-6 floor proves only non-nullity.
      REQUIRE(d_finite_q > 1e-4 * w_scale);

      // "dynamic" skips nu_n = 0, and nothing downstream of the projection reintroduces a
      // dependence on it, so that whole sector must come back bit-for-bit identical.
      REQUIRE(d_static == 0.0);

      // At Gamma the shift is suppressed to round-off (~1e-19 against a W of ~4e-7). This
      // is NOT a defect and NOT a property of the projection: the rank-one update lives
      // entirely in the D D^dag channel, and the THC Coulomb at the zone center has its
      // divergent head removed, so Z(Gamma) annihilates exactly that channel. It is
      // logged rather than asserted because it is a statement about the divergence
      // treatment. Note the consequence for test design: a Gamma-only check of "the
      // projection changed W" would read as a null result.
      app_log(2, "  (Gamma shift {:.2e} relative -- Z(Gamma) has no head channel)",
              d_gamma / w_scale);
    }

    SECTION("\"insulator\" additionally projects the static sector") {
      // The Dyson hook has exactly two behaviours -- project nu_n = 0 or leave it alone --
      // and this is the only place the second one is exercised anywhere in the codebase.
      // The two modes differ ONLY at n = 0: under "insulator" dC(0) is the Gamma head
      // rather than zero, so the static sector must move, and every nu_n != 0 sector must
      // come back bit-for-bit identical to "dynamic".
      REQUIRE(has_dC_ins);
      VALUE_EQUAL(dC_ins(0), head_ref(0, iq_gamma), 1e-10);
      VALUE_EQUAL(dC_dyn(0), ComplexType(0.0), 1e-14);
      for (long w = 1; w < nw; ++w) VALUE_EQUAL(dC_ins(w), dC_dyn(w), 1e-14);

      double d_static = 0.0, d_dynamic = 0.0, w_scale = 0.0;
      for (long w = 0; w < nw; ++w) {
        for (long q = 0; q < nq; ++q) {
          for (long P = 0; P < Np; ++P) {
            for (long Q = 0; Q < Np; ++Q) {
              w_scale = std::max(w_scale, std::abs(W_none(w, q, P, Q)));
              if (w == 0)
                d_static = std::max(d_static,
                                    std::abs(W_ins(w, q, P, Q) - W_none(w, q, P, Q)));
              else
                d_dynamic = std::max(d_dynamic,
                                     std::abs(W_ins(w, q, P, Q) - W_dyn(w, q, P, Q)));
            }
          }
        }
      }
      app_log(2, "  insulator: max|W_ins - W_none| at nu=0 -> {:.6e} ({:.2e} relative); "
                 "max|W_ins - W_dyn| at nu!=0 -> {:.6e}",
              d_static, d_static / w_scale, d_dynamic);

      REQUIRE(w_scale > 0.0);
      // Measured 5.36e-03 relative -- roughly twice the dynamic finite-q shift, as the
      // static head should be. 1e-4 keeps a ~54x margin.
      REQUIRE(d_static > 1e-4 * w_scale);
      REQUIRE(d_dynamic == 0.0);
    }
  }

  /**
   * "extrapolate": dC(0) is the nu -> 0 limit of dC(i nu_n), n >= 1, so that a static-only
   * term in Pi_00(Gamma, 0) -- the compressibility -dn/dmu, which the Ward identity confines
   * to n = 0 -- survives while the smooth Ward-violating constant is removed. The fit is
   * linear in nu^2, so a synthetic head that IS linear in nu^2 for n >= 1 makes the
   * expected dC(0) exact, and a deliberate excess at n = 0 must be what remains after the
   * projection. The frequency axis has to be the positive half of the IAFT bosonic mesh,
   * since the kernel reads nu_n from the IAFT.
   */
  TEST_CASE("pi_head_extrapolate", "[methods][scr_coulomb][pi_head]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*12, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));
    long Np = thc.Np();
    long nq = mf->nqpts_ibz();
    long iq_gamma = solvers::div_utils::find_gamma_index(mf->Qpts_ibz());
    REQUIRE(iq_gamma >= 0);

    auto wn_b = ft.wn_mesh_b();
    long nw_b = wn_b.shape(0);
    long off  = nw_b / 2;
    long nw   = nw_b - off;                       // positive half mesh, index 0 <-> nu = 0
    REQUIRE(wn_b(off) == 0);
    REQUIRE(nw > 5);
    auto nu = [&](long n) { return 6.283185307179586 * wn_b(off + n) / ft.beta(); };

    // Pi(w) = f(w) * base, f linear in nu^2 for n >= 1 plus a static-only excess at n = 0.
    const double c0 = 1.0, excess = 0.25;
    const double c2 = 0.3 / (nu(1) * nu(1));
    auto f = [&](long w) { return (w == 0) ? c0 + excess : c0 + c2 * nu(w) * nu(w); };

    auto& self_comm = mpi3::environment::get_self_instance();
    auto dBase = make_synthetic_Pi(mf, self_comm, 1, Np);    // f = 1 at w = 0 by construction
    auto dPi = math::nda::make_distributed_array<nda::array<ComplexType, 4> >(
        self_comm, {1, 1, 1, 1}, {nw, nq, Np, Np}, {1, 1, 1, 1});
    {
      auto base = dBase.local();
      auto Pi   = dPi.local();
      for (long w = 0; w < nw; ++w)
        Pi(w, nda::ellipsis{}) = f(w) * base(0, nda::ellipsis{});
    }

    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0", "extrapolate");
    auto head_in  = solvers::div_utils::head_from_prod_basis(dPi, thc, false);
    auto dC       = scr_eri.regularize_Pi_head(dPi, thc, head_in);
    auto head_out = solvers::div_utils::head_from_prod_basis(dPi, thc, false);

    // The base head at Gamma sets the scale of every assertion; it must not be vacuous.
    ComplexType hb = head_in(1, iq_gamma) / f(1);
    REQUIRE(std::abs(hb) > 1e-6);
    // VALUE_EQUAL(A, B, m) passes on |A - B| < m OR a relative 1e-8; the absolute floor
    // covers the zero checks (the contraction's round-off is ~1e-14), the relative part
    // the nonzero ones.
    const double tol = 1e-10;
    app_log(2, "  extrapolate: base head {:.6e}, dC(0) = {:.6e} (expected {:.6e}), "
               "head_out(0,Gamma) = {:.6e} (expected {:.6e})",
            std::abs(hb), dC(0).real(), (c0 * hb).real(),
            head_out(0, iq_gamma).real(), (excess * hb).real());

    SECTION("dC(0) is the nu -> 0 limit of the n >= 1 column, not the n = 0 head") {
      VALUE_EQUAL(dC(0), c0 * hb, tol);
      // and NOT the raw Gamma head, which includes the excess
      REQUIRE(std::abs(dC(0) - head_in(0, iq_gamma)) > 0.5 * std::abs(excess * hb));
    }

    SECTION("nu_n != 0 sectors are projected exactly as under dynamic/insulator") {
      // The synthetic head grows as c2 nu^2 and reaches ~1e4 at the top of the mesh, so the
      // round-off floor of the projected head scales with |head_in(w)|; the zero check must too.
      for (long w = 1; w < nw; ++w) {
        double scale = std::max(1.0, std::abs(head_in(w, iq_gamma)));
        // dC is Re of the head by construction (the kernel truncates the round-off
        // imaginary part), so compare against the real part.
        VALUE_EQUAL(dC(w), ComplexType(head_in(w, iq_gamma).real(), 0.0), tol * scale);
        VALUE_EQUAL(head_out(w, iq_gamma), ComplexType(0.0), tol * scale);
      }
    }

    SECTION("the static-only excess survives the projection") {
      VALUE_EQUAL(head_out(0, iq_gamma), excess * hb, tol);
    }

    SECTION("finite q shifts by the same dC(0) at nu = 0") {
      for (long q = 0; q < nq; ++q)
        VALUE_EQUAL(head_out(0, q), head_in(0, q) - dC(0), tol);
    }
  }

  /**
   * The head diagnostics are persisted by scr_coulomb_t::dump_eps_inv_head, which only
   * runs on the update_w() path and only for h5_iter >= 0. The sections above exercise
   * dyson_W_in_place() directly and therefore cannot see the write at all, so this case
   * drives the production entry point: a mean-field Green's function, then one update_w().
   * That is the cheapest driver that reaches the write -- it skips the self-energy
   * evaluation a full scf_loop() would add without changing anything on the path under
   * test, since dump_eps_inv_head is called from update_w() itself.
   */
  TEST_CASE("pi_head_h5_datasets", "[methods][scr_coulomb][pi_head]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*12, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));

    long nw_half   = (ft.nw_b() % 2 == 0) ? ft.nw_b()/2 : ft.nw_b()/2 + 1;
    long iq_gamma  = solvers::div_utils::find_gamma_index(mf->Qpts_ibz());
    REQUIRE(iq_gamma >= 0);
    const long h5_iter = 1;

    using Array_view_4D_t = nda::array_view<ComplexType, 4>;
    using Array_view_5D_t = nda::array_view<ComplexType, 5>;

    auto run = [&](std::string reg, std::string output) {
      std::string filename = output + ".mbpt.h5";
      // dump_eps_inv_head opens the checkpoint in append mode, so a leftover file from an
      // earlier run would let a stale dataset masquerade as a fresh write.
      if (mpi->comm.root()) remove(filename.c_str());
      mpi->comm.barrier();

      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0", reg);
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi, ft, output);

      mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
          *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
      mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
          *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
      mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
          *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
      mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
          *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));

      // Same initialization as the non-restart branch of scf_loop: Fock from the
      // mean-field, zero dynamic self-energy, then one Dyson solve for G.
      hamilt::set_fock(*mf, dyson.PSP(), mb_state.sF_skij.value(), true);
      double mu = 0.0;
      update_G(dyson, *mf, ft, mb_state.sDm_skij.value(), mb_state.sG_tskij.value(),
               mb_state.sF_skij.value(), mb_state.sSigma_tskij.value(), mu, false);

      scr_eri.update_w(mb_state, thc, h5_iter);
      mb_state.dW_qtPQ.reset();
      mpi->comm.barrier();
      return filename;
    };

    SECTION("\"dynamic\" writes both the head and the applied dC") {
      auto filename = run("dynamic", "pi_head_h5_dyn");

      if (mpi->comm.root()) {
        nda::array<ComplexType, 2> pi_head_wq;
        nda::array<ComplexType, 1> delta_C_w;
        {
          h5::file file(filename, 'r');
          auto iter_grp = h5::group(file).open_group("scf")
                                         .open_group("iter" + std::to_string(h5_iter));
          REQUIRE(iter_grp.has_dataset("pi_head_wq"));
          REQUIRE(iter_grp.has_dataset("delta_C_w"));
          nda::h5_read(iter_grp, "pi_head_wq", pi_head_wq);
          nda::h5_read(iter_grp, "delta_C_w", delta_C_w);
        }

        REQUIRE(pi_head_wq.shape(0) == nw_half);
        REQUIRE(pi_head_wq.shape(1) == mf->nqpts_ibz());
        REQUIRE(delta_C_w.shape(0) == nw_half);

        // "dynamic" leaves the static sector alone, and regularize_Pi_head writes that
        // zero literally rather than computing it, so the round-trip must be exact.
        REQUIRE(delta_C_w(0) == ComplexType(0.0));

        // The two datasets must be mutually consistent: dC is the real part of the Gamma
        // column of the recorded (unprojected) head. This is what pins down that
        // pi_head_wq is stored with q as its SECOND index -- a transposed write would
        // still have the right shape whenever nw_half == nqpts_ibz, but would break here.
        double dc_scale = 0.0;
        for (long w = 1; w < nw_half; ++w) {
          VALUE_EQUAL(delta_C_w(w),
                      ComplexType(pi_head_wq(w, iq_gamma).real(), 0.0), 1e-14);
          dc_scale = std::max(dc_scale, std::abs(delta_C_w(w)));
        }
        // Non-vacuous only if the projection actually had something to remove.
        app_log(2, "  max_n |dC| = {:.6e}", dc_scale);
        REQUIRE(dc_scale > 1e-8);

        remove(filename.c_str());
      }
      mpi->comm.barrier();
    }

    SECTION("\"none\" writes the head but no dC") {
      auto filename = run("none", "pi_head_h5_none");

      if (mpi->comm.root()) {
        nda::array<ComplexType, 2> pi_head_wq;
        {
          h5::file file(filename, 'r');
          auto iter_grp = h5::group(file).open_group("scf")
                                         .open_group("iter" + std::to_string(h5_iter));
          // The head is the diagnostic that tells a user whether the projection is needed
          // at all, so it is recorded with the projection off; the absence of delta_C_w is
          // itself the record that nothing was subtracted. The asymmetry is deliberate.
          REQUIRE(iter_grp.has_dataset("pi_head_wq"));
          REQUIRE_FALSE(iter_grp.has_dataset("delta_C_w"));
          nda::h5_read(iter_grp, "pi_head_wq", pi_head_wq);
        }
        REQUIRE(pi_head_wq.shape(0) == nw_half);
        REQUIRE(pi_head_wq.shape(1) == mf->nqpts_ibz());
        REQUIRE(std::abs(pi_head_wq(1, iq_gamma)) > 1e-8);

        remove(filename.c_str());
      }
      mpi->comm.barrier();
    }
  }

  /**
   * The physical calibration of the feature. The single-shot (G0W0) polarization is a
   * bubble of MEAN-FIELD propagators, whose self-energy carries no frequency dependence,
   * so DeltaSigma = Sigma(k+q) - Sigma(k) vanishes identically and the frequency sector
   * of the longitudinal Ward identity is satisfied exactly:
   *
   *     Pi_00(q = 0, i nu_n != 0) = 0    exactly, for a mean-field G
   *
   * (ward_identity_note.tex, eq:null-test). Whatever dC the code reports here is therefore
   * pure numerical residue -- the floor of the extraction on this mesh and this auxiliary
   * basis -- and it is what gives the self-consistent number a scale to be measured
   * against. Unlike the post-processing tool, the in-loop path reads dC straight off the
   * stored Gamma column, so there is no extrapolation window and no fit truncation error
   * folded into the residue.
   *
   * The ratio |dC| / |Pi_00(q_min)| is the null test proper. But dC is by construction a
   * GAMMA quantity, and Gamma is precisely where this feature's failure modes hide: the
   * THC head vectors are real there, so the two candidate conjugation conventions are
   * degenerate, and Z(Gamma) carries no head channel. A Gamma-only assertion is passed
   * *perfectly* by a bug that zeroes the head at every q. The finite-q assertions below --
   * that the head is small at Gamma RELATIVE to its finite-q values, and that those
   * finite-q values are genuinely nonzero -- are what close that hole.
   */
  TEST_CASE("pi_head_g0w0_null", "[methods][scr_coulomb][pi_head]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*24, "", "incore", "", "bdft",
                                               1e-10, mf->ecutrho(), 1, 1024));

    long nq       = mf->nqpts_ibz();
    long iq_gamma = solvers::div_utils::find_gamma_index(mf->Qpts_ibz());
    REQUIRE(iq_gamma >= 0);
    // Without at least one finite q there is nothing to compare the Gamma head against,
    // and the projection degenerates (regularize_Pi_head warns about exactly this).
    REQUIRE(nq > 1);

    using Array_view_4D_t = nda::array_view<ComplexType, 4>;
    using Array_view_5D_t = nda::array_view<ComplexType, 5>;

    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0", "dynamic");
    simple_dyson dyson(mf.get(), &ft);
    MBState mb_state(mpi, ft, "pi_head_g0w0_null");

    mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
    mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
        *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
    mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
    mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {ft.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));

    // The non-restart branch of scf_loop: Fock from the mean-field, a ZERO dynamic
    // self-energy, then one Dyson solve. The zero Sigma is the whole point -- it is what
    // makes G a mean-field propagator and the exact-zero claim above applicable.
    hamilt::set_fock(*mf, dyson.PSP(), mb_state.sF_skij.value(), true);
    double mu = 0.0;
    update_G(dyson, *mf, ft, mb_state.sDm_skij.value(), mb_state.sG_tskij.value(),
             mb_state.sF_skij.value(), mb_state.sSigma_tskij.value(), mu, false);

    // h5_iter = -1: no checkpoint write. The dump is covered by pi_head_h5_datasets.
    scr_eri.update_w(mb_state, thc, -1);
    mb_state.dW_qtPQ.reset();

    REQUIRE(scr_eri.has_delta_C());
    auto& dC      = scr_eri.delta_C_w();
    auto& pi_head = scr_eri.pi_head_wq();
    long nw = pi_head.shape(0);
    REQUIRE(pi_head.shape(1) == nq);
    REQUIRE(dC.shape(0) == nw);
    REQUIRE(nw > 1);

    long iq_small = solvers::div_utils::find_smallest_qabs(mf->Qpts_ibz(), false);
    REQUIRE(iq_small >= 0);
    REQUIRE(iq_small != iq_gamma);

    // Cross-reference for the residue: a poorly resolved uniform density mode shows up
    // here first, and would explain a large ratio without any bug in the contraction.
    auto sigma = thc.basis_head_overlap();
    double sig_dev = 0.0;
    for (long q = 0; q < nq; ++q)
      sig_dev = std::max(sig_dev, std::abs(sigma(q) - ComplexType(1.0)));

    auto qabs2 = [&](long q) {
      auto qpt = mf->Qpts_ibz(q);
      return qpt(0)*qpt(0) + qpt(1)*qpt(1) + qpt(2)*qpt(2);
    };

    // On the record: the head at every irreducible q for the first nonzero bosonic
    // frequency, with |q|^2 alongside so the q -> 0 behaviour is visible.
    app_log(2, "\n  G0W0 null test (max_q |sigma-1| = {:.4e}):", sig_dev);
    app_log(2, "  Pi_00(q, i nu_1) across the irreducible q:");
    double head_finite_q = 0.0;
    for (long q = 0; q < nq; ++q) {
      app_log(2, "    q = {:2d}: |q|^2 = {:.6e}  Pi_00 = ({:.6e}, {:.6e})  |Pi_00| = {:.6e}{}",
              q, qabs2(q), pi_head(1, q).real(), pi_head(1, q).imag(),
              std::abs(pi_head(1, q)), (q == iq_gamma) ? "   <- Gamma" : "");
      if (q != iq_gamma) head_finite_q = std::max(head_finite_q, std::abs(pi_head(1, q)));
    }

    // The frequency dependence is the decisive signature (see the note's benchmark table),
    // so record the whole half mesh -- and assert on it. The residue tracks the head
    // itself (both decay together at large nu_n), so the ratio stays bounded and the
    // whole-mesh statement costs nothing over the nu_1 one; measured max is 1.8e-08.
    app_log(2, "  frequency dependence:");
    double ratio_max = 0.0, gamma_ratio_max = 0.0;
    for (long w = 1; w < nw; ++w) {
      double s = std::abs(pi_head(w, iq_small));
      double fq = 0.0;
      for (long q = 0; q < nq; ++q)
        if (q != iq_gamma) fq = std::max(fq, std::abs(pi_head(w, q)));
      // Log before asserting, so a failure still leaves the offending row on the record.
      double r  = (s  > 0.0) ? std::abs(dC(w)) / s : -1.0;
      double gr = (fq > 0.0) ? std::abs(pi_head(w, iq_gamma)) / fq : -1.0;
      app_log(2, "    n = {:2d}: dC = {:.6e}, |Pi_00(q_min)| = {:.6e}, ratio = {:.4e}, "
                 "|Pi_00(Gamma)|/max_q|Pi_00| = {:.4e}", w, std::abs(dC(w)), s, r, gr);
      REQUIRE(s > 0.0);
      REQUIRE(fq > 0.0);
      ratio_max       = std::max(ratio_max, r);
      gamma_ratio_max = std::max(gamma_ratio_max, gr);
    }
    app_log(2, "  max over the half mesh: ratio = {:.4e}, "
               "|Pi_00(Gamma)|/max_q|Pi_00| = {:.4e}", ratio_max, gamma_ratio_max);
    REQUIRE(ratio_max < 0.05);
    REQUIRE(gamma_ratio_max < 0.05);

    // --- the null test ---------------------------------------------------------------
    // Against the scale of the polarization head itself at the smallest sampled |q|, not
    // against an absolute threshold: what matters is that the residue is a small FRACTION
    // of the signal.
    double scale = std::abs(pi_head(1, iq_small));
    REQUIRE(scale > 0.0);
    app_log(2, "\n  dC(i nu_1) = {:.6e}, |Pi_00(q_min, i nu_1)| = {:.6e}, ratio = {:.4e}",
            std::abs(dC(1)), scale, std::abs(dC(1)) / scale);
    REQUIRE(std::abs(dC(1)) / scale < 0.05);

    // --- the finite-q guard ----------------------------------------------------------
    // This is the assertion an all-zero head cannot satisfy: without it the ratio above
    // is passed perfectly by a broken contraction that returns zero everywhere.
    REQUIRE(head_finite_q > 0.0);
    app_log(2, "  max_{{q != Gamma}} |Pi_00(q, i nu_1)| = {:.6e}, "
               "|Pi_00(Gamma)| / max_{{q != Gamma}} |Pi_00| = {:.4e}",
            head_finite_q, std::abs(pi_head(1, iq_gamma)) / head_finite_q);
    // The physically meaningful statement: the head vanishes at the zone center while
    // staying finite everywhere else.
    REQUIRE(std::abs(pi_head(1, iq_gamma)) < 0.05 * head_finite_q);
  }

} // bdft_tests
