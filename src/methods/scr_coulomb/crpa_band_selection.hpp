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

#ifndef COQUI_CRPA_BAND_SELECTION_HPP
#define COQUI_CRPA_BAND_SELECTION_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <format>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

namespace methods {
namespace solvers {
namespace crpa {

  /**
   * Band selection for the active-space polarization of the "crpa_ks" (spectral cRPA,
   * Kaltak et al., PRB 112, 245102 (2025)) and "crpa_vasp" (projector cRPA) schemes.
   *
   * At each spin and k-point the nImpOrbs window bands with the largest Wannier weight
   *   w_i = P_ii = sum_a |C_ai|^2
   * are selected. The struct records everything needed to report and reproduce the choice.
   *
   * The weights, captured weight and margin are diagnostics: they are reported by
   * log_band_selection() and deliberately go no further than the log — they are not
   * written to the checkpoint and not exposed to Python.
   */
  struct band_selection_t {
    nda::array<double, 3> weight_ski;   // (ns, nk, nOrbs_W)  Wannier weight of every window band
    nda::array<long, 3>   bands_ski;    // (ns, nk, nImpOrbs) absolute indices of the selected bands, ascending
    nda::array<double, 2> captured_sk;  // (ns, nk)           sum of the selected weights (<= nImpOrbs)
    nda::array<double, 2> margin_sk;    // (ns, nk)           smallest selected - largest rejected weight;
                                        //                    +inf when nothing is rejected (nImpOrbs == nOrbs_W)
    long band_offset = 0;               // absolute index of the first window band
  };

  /**
   * Indices of the n_select largest entries of w, returned in ascending index order.
   * Ranking is by weight in descending order; ties are resolved toward the lower index,
   * so the result is deterministic.
   */
  inline std::vector<long> select_bands_by_weight(nda::ArrayOfRank<1> auto const& w, long n_select) {
    long n = w.shape(0);
    utils::check(n_select >= 0 and n_select <= n,
                 "crpa_band_selection::select_bands_by_weight: n_select = {} outside [0, {}].",
                 n_select, n);
    std::vector<long> idx(n);
    std::iota(idx.begin(), idx.end(), 0L);
    // stable sort keeps the lower index first among equal weights
    std::stable_sort(idx.begin(), idx.end(), [&w](long a, long b) { return w(a) > w(b); });
    idx.resize(n_select);
    std::sort(idx.begin(), idx.end());
    return idx;
  }

  /**
   * Select bands for every spin and k-point of one impurity.
   * @param C_skai       - [INPUT] projector (ns, nk, nImpOrbs, nOrbs_W): C_ai(k) = <w_a k | psi_i k>
   * @param band_offset  - [INPUT] absolute index of the first window band (W_rng.first())
   */
  inline band_selection_t select_bands(nda::ArrayOfRank<4> auto const& C_skai, long band_offset) {
    auto [ns, nk, nImpOrbs, nOrbs_W] = C_skai.shape();
    utils::check(nImpOrbs <= nOrbs_W,
                 "crpa_band_selection::select_bands: nImpOrbs ({}) exceeds the window size ({}).",
                 nImpOrbs, nOrbs_W);
    constexpr double inf = std::numeric_limits<double>::infinity();

    band_selection_t sel{nda::array<double, 3>(ns, nk, nOrbs_W),
                         nda::array<long, 3>(ns, nk, nImpOrbs),
                         nda::array<double, 2>(ns, nk),
                         nda::array<double, 2>(ns, nk),
                         band_offset};

    for (long is = 0; is < ns; ++is) {
      for (long ik = 0; ik < nk; ++ik) {
        auto w = sel.weight_ski(is, ik, nda::range::all);
        for (long i = 0; i < nOrbs_W; ++i) {
          double wi = 0.0;
          for (long a = 0; a < nImpOrbs; ++a) wi += std::norm(C_skai(is, ik, a, i));
          w(i) = wi;
        }

        auto chosen = select_bands_by_weight(w, nImpOrbs);
        std::vector<bool> is_chosen(nOrbs_W, false);
        double captured = 0.0, min_sel = inf, max_rej = -inf;
        for (long a = 0; a < nImpOrbs; ++a) {
          long i = chosen[a];
          is_chosen[i] = true;
          captured += w(i);
          min_sel = std::min(min_sel, w(i));
          sel.bands_ski(is, ik, a) = band_offset + i;
        }
        for (long i = 0; i < nOrbs_W; ++i)
          if (not is_chosen[i]) max_rej = std::max(max_rej, w(i));

        sel.captured_sk(is, ik) = captured;
        sel.margin_sk(is, ik) = (nImpOrbs == nOrbs_W) ? inf : min_sel - max_rej;
      }
    }
    return sel;
  }

  /**
   * Number of k-points, per spin, whose selected set differs from that of the previous k-point.
   * A proxy for how discontinuous the active-space polarization is in k.
   */
  inline nda::array<long, 1> count_selection_changes(band_selection_t const& sel) {
    auto [ns, nk, nImpOrbs] = sel.bands_ski.shape();
    nda::array<long, 1> changes(ns);
    changes() = 0;
    for (long is = 0; is < ns; ++is) {
      for (long ik = 1; ik < nk; ++ik) {
        bool differs = false;
        for (long a = 0; a < nImpOrbs and not differs; ++a)
          differs = sel.bands_ski(is, ik, a) != sel.bands_ski(is, ik - 1, a);
        if (differs) changes(is) += 1;
      }
    }
    return changes;
  }

  /**
   * Report the selection: a summary at output level 1, the per-k table at level 2, and the
   * rows whose margin is below margin_tol at level 1 together with a warning.
   * @param sel         - [INPUT] result of select_bands
   * @param kpts_crys   - [INPUT] k-points in crystal coordinates, (nk, 3), same order as sel
   * @param scheme      - [INPUT] screen_type label for the header
   * @param margin_tol  - [INPUT] margin below which the ranking is considered fragile
   */
  inline void log_band_selection(band_selection_t const& sel,
                                 nda::ArrayOfRank<2> auto const& kpts_crys,
                                 std::string scheme, double margin_tol = 0.1) {
    auto [ns, nk, nOrbs_W] = sel.weight_ski.shape();
    long nImpOrbs = sel.bands_ski.shape(2);
    utils::check(kpts_crys.shape(0) == nk,
                 "crpa_band_selection::log_band_selection: kpts_crys has {} rows, expected {}.",
                 kpts_crys.shape(0), nk);
    if (nk == 0 or ns == 0) return;

    long offset = sel.band_offset;

    // aggregate over spin and k
    double captured_avg = 0.0, captured_min = std::numeric_limits<double>::infinity();
    double margin_min = std::numeric_limits<double>::infinity();
    long s_cmin = 0, k_cmin = 0, s_mmin = 0, k_mmin = 0, n_fragile = 0;
    for (long is = 0; is < ns; ++is) {
      for (long ik = 0; ik < nk; ++ik) {
        double c = sel.captured_sk(is, ik), m = sel.margin_sk(is, ik);
        captured_avg += c;
        if (c < captured_min) { captured_min = c; s_cmin = is; k_cmin = ik; }
        if (m < margin_min) { margin_min = m; s_mmin = is; k_mmin = ik; }
        if (m < margin_tol) ++n_fragile;
      }
    }
    captured_avg /= double(ns * nk);
    auto changes = count_selection_changes(sel);
    long n_changes = 0;
    for (long is = 0; is < ns; ++is) n_changes += changes(is);

    bool disentangled = std::isinf(margin_min);

    app_log(1, "\n  Band selection for the active-space polarization (screen_type = {})\n"
               "    - window bands (0-based, half-open)    = [{}, {}), {} selected per spin and k-point\n"
               "    - weight of a band = sum_a |C_ai(k)|^2 over the target orbitals; weights sum to {} at each k\n"
               "    - Wannier weight captured by selection = {:.3f} of {} (average), min {:.3f} at (s, k) = ({}, {})",
            scheme, offset, offset + nOrbs_W, nImpOrbs, nImpOrbs, captured_avg, nImpOrbs, captured_min, s_cmin, k_cmin);
    if (disentangled) {
      app_log(1, "    - the window is disentangled (nImpOrbs = nOrbs_W): every window band\n"
                 "      is selected at every k-point\n");
      return;
    }
    app_log(1, "    - selection margin                     = {:.3f} (min over spin and k, at (s, k) = ({}, {}))\n"
               "        (smallest selected weight - largest rejected weight)\n"
               "    - k-points with margin < {:.2f}           = {} of {}\n"
               "    - k-points whose selection differs from the previous k-point = {} of {}\n",
            margin_min, s_mmin, k_mmin, margin_tol, n_fragile, ns * nk, n_changes, ns * (nk - 1));

    if (n_fragile > 0) {
      app_log(1, " [WARNING] The Wannier-weight ranking that selects the active bands is nearly tied at\n"
                 "           {} of {} (spin, k) pairs (margin < {:.2f}; worst {:.3f} at (s, k) = ({}, {})).\n"
                 "           Ties are resolved toward the lower band index. The active-space polarization\n"
                 "           is discontinuous in k there; the affected rows are listed below.\n",
              n_fragile, ns * nk, margin_tol, margin_min, s_mmin, k_mmin);
    }

    // per-k table: all rows at level 2, fragile rows at level 1
    std::string header = std::format("    {:>2} {:>4}  {:^23}  {:<{}}  {:<{}}  {:>8}  {}",
                                     "s", "k", "k (crystal)", "selected bands", std::max<long>(14, 4 * nImpOrbs),
                                     "weights", std::max<long>(7, 6 * nImpOrbs), "captured", "max rejected");
    app_log(n_fragile > 0 ? 1 : 2, "{}", header);
    for (long is = 0; is < ns; ++is) {
      for (long ik = 0; ik < nk; ++ik) {
        std::string bands, weights;
        for (long a = 0; a < nImpOrbs; ++a) {
          long i = sel.bands_ski(is, ik, a);
          bands += std::format("{:>3} ", i);
          weights += std::format("{:5.2f} ", sel.weight_ski(is, ik, i - offset));
        }
        double max_rej = -1.0;
        long i_rej = -1;
        for (long i = 0; i < nOrbs_W; ++i) {
          bool chosen = false;
          for (long a = 0; a < nImpOrbs; ++a) chosen |= (sel.bands_ski(is, ik, a) == offset + i);
          if (not chosen and sel.weight_ski(is, ik, i) > max_rej) {
            max_rej = sel.weight_ski(is, ik, i);
            i_rej   = offset + i;
          }
        }
        bool fragile = sel.margin_sk(is, ik) < margin_tol;
        std::string row = std::format("    {:>2} {:>4}  {:7.3f} {:7.3f} {:7.3f}  {:<{}}  {:<{}}  {:8.2f}  {:5.2f} (band {}){}",
                                      is, ik, kpts_crys(ik, 0), kpts_crys(ik, 1), kpts_crys(ik, 2),
                                      bands, std::max<long>(14, 4 * nImpOrbs),
                                      weights, std::max<long>(7, 6 * nImpOrbs),
                                      sel.captured_sk(is, ik), max_rej, i_rej, fragile ? "  <- margin below tolerance" : "");
        app_log(fragile ? 1 : 2, "{}", row);
      }
    }
    app_log(n_fragile > 0 ? 1 : 2, "");
  }

} // crpa
} // solvers
} // methods

#endif // COQUI_CRPA_BAND_SELECTION_HPP
