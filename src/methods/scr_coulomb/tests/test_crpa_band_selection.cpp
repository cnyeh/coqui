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

#include <cmath>
#include <limits>
#include <vector>

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "nda/nda.hpp"

#include "methods/scr_coulomb/crpa_band_selection.hpp"

namespace bdft_tests {

  using namespace methods::solvers::crpa;

  TEST_CASE("select_bands_by_weight ranks by weight, ties broken by lower band index",
            "[crpa_band_selection]") {
    nda::array<double, 1> w = {0.2, 0.9, 0.5, 0.9};

    auto two = select_bands_by_weight(w, 2);
    REQUIRE(two == std::vector<long>{1, 3});

    auto one = select_bands_by_weight(w, 1);
    REQUIRE(one == std::vector<long>{1});      // tie 0.9/0.9 -> lower index

    nda::array<double, 1> flat = {0.5, 0.5, 0.5};
    REQUIRE(select_bands_by_weight(flat, 2) == std::vector<long>{0, 1});

    REQUIRE(select_bands_by_weight(w, 4) == std::vector<long>{0, 1, 2, 3});
  }

  TEST_CASE("select_bands reports weights, captured weight and margin for an entangled window",
            "[crpa_band_selection]") {
    // one spin, one k, one target orbital, a window of three bands
    long ns = 1, nk = 1, nImpOrbs = 1, nOrbs_W = 3, offset = 20;
    nda::array<ComplexType, 4> C(ns, nk, nImpOrbs, nOrbs_W);
    C(0, 0, 0, 0) = std::sqrt(0.7) * std::exp(ComplexType(0.0, 0.3));
    C(0, 0, 0, 1) = std::sqrt(0.2) * std::exp(ComplexType(0.0, -1.1));
    C(0, 0, 0, 2) = std::sqrt(0.1);

    auto sel = select_bands(C, offset);

    REQUIRE(sel.weight_ski.shape() == std::array<long, 3>{ns, nk, nOrbs_W});
    REQUIRE(sel.weight_ski(0, 0, 0) == Approx(0.7));
    REQUIRE(sel.weight_ski(0, 0, 1) == Approx(0.2));
    REQUIRE(sel.weight_ski(0, 0, 2) == Approx(0.1));

    REQUIRE(sel.bands_ski.shape() == std::array<long, 3>{ns, nk, nImpOrbs});
    REQUIRE(sel.bands_ski(0, 0, 0) == offset + 0);   // absolute band index
    REQUIRE(sel.band_offset == offset);

    REQUIRE(sel.captured_sk(0, 0) == Approx(0.7));
    REQUIRE(sel.margin_sk(0, 0) == Approx(0.7 - 0.2));
  }

  TEST_CASE("select_bands on a disentangled window selects everything with infinite margin",
            "[crpa_band_selection]") {
    long ns = 1, nk = 1, nImpOrbs = 2, nOrbs_W = 2, offset = 5;
    nda::array<ComplexType, 4> C(ns, nk, nImpOrbs, nOrbs_W);
    double c = std::cos(0.4), s = std::sin(0.4);
    C(0, 0, 0, 0) = c;  C(0, 0, 0, 1) = s;
    C(0, 0, 1, 0) = -s; C(0, 0, 1, 1) = c;   // unitary: C C^dagger = 1

    auto sel = select_bands(C, offset);

    REQUIRE(sel.weight_ski(0, 0, 0) == Approx(1.0));
    REQUIRE(sel.weight_ski(0, 0, 1) == Approx(1.0));
    REQUIRE(sel.bands_ski(0, 0, 0) == offset + 0);
    REQUIRE(sel.bands_ski(0, 0, 1) == offset + 1);
    REQUIRE(sel.captured_sk(0, 0) == Approx(2.0));
    REQUIRE(sel.margin_sk(0, 0) == std::numeric_limits<double>::infinity());
  }

  TEST_CASE("count_selection_changes counts k-points whose selected set differs from k-1",
            "[crpa_band_selection]") {
    long ns = 2, nk = 3, nImpOrbs = 1, nOrbs_W = 2, offset = 0;
    nda::array<ComplexType, 4> C(ns, nk, nImpOrbs, nOrbs_W);
    C() = 0.0;
    // spin 0: band 0 at k=0,1; band 1 at k=2  -> one change
    C(0, 0, 0, 0) = 1.0;
    C(0, 1, 0, 0) = 1.0;
    C(0, 2, 0, 1) = 1.0;
    // spin 1: band 1 everywhere -> no change
    C(1, 0, 0, 1) = 1.0;
    C(1, 1, 0, 1) = 1.0;
    C(1, 2, 0, 1) = 1.0;

    auto sel = select_bands(C, offset);
    auto changes = count_selection_changes(sel);

    REQUIRE(changes.shape() == std::array<long, 1>{ns});
    REQUIRE(changes(0) == 1);
    REQUIRE(changes(1) == 0);
  }

  TEST_CASE("log_band_selection runs on a valid selection", "[crpa_band_selection]") {
    long ns = 1, nk = 2, nImpOrbs = 1, nOrbs_W = 2, offset = 3;
    nda::array<ComplexType, 4> C(ns, nk, nImpOrbs, nOrbs_W);
    C() = 0.0;
    C(0, 0, 0, 0) = std::sqrt(0.55); C(0, 0, 0, 1) = std::sqrt(0.45);   // margin 0.10
    C(0, 1, 0, 0) = 1.0;
    nda::array<double, 2> kpts(nk, 3);
    kpts() = 0.0;
    kpts(1, 0) = 0.5;

    auto sel = select_bands(C, offset);
    REQUIRE_NOTHROW(log_band_selection(sel, kpts, "crpa_ks", 0.2));
  }

} // bdft_tests
