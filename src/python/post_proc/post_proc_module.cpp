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


#include <c2py/c2py.hpp>
#include "IO/app_loggers.h"
#include "methods/pproc/pproc_drivers.hpp"

#include "python/mean_field/mf_module.hpp"
#include "python/mean_field/mf_module.wrap.hxx"
#include "python/interaction/eri_module.hpp"
#include "python/interaction/eri_module.wrap.hxx"
#include "methods/scr_coulomb/dielectric_pproc.hpp"

namespace coqui_py::post_proc {

  void band_interpolation(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("band_interpolation", mf.get_mf(), parser.get_root());
  }

  void spectral_interpolation(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("spectral_interpolation", mf.get_mf(), parser.get_root());
  }

  void local_dos(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("local_dos", mf.get_mf(), parser.get_root());
  }

  void unfold_bz(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("unfold_bz", mf.get_mf(), parser.get_root());
  }

  void dump_vxc(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("dump_vxc", mf.get_mf(), parser.get_root());
  }

  void dump_hartree(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("dump_hartree", mf.get_mf(), parser.get_root());
  }

  /**
   * Dielectric function from a checkpoint Green's function with polarization-head
   * regularization. See methods::dielectric_function (dielectric_pproc.hpp) for the keys
   * and the output group layout. A "crpa" or "edmft" screen_type needs the bosonic
   * projector, which this overload reads from the HDF5 file named by params["wannier_file"].
   */
  template<typename eri_handler_t>
  void dielectric_function_with_projector_from_h5(
      eri_handler_t &eri, const std::string &params,
      std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(params);
    methods::dielectric_function(eri.get_eri(), parser.get_root(),
                                 std::move(local_polarizabilities));
  }

  /**
   * Same, with the bosonic projector supplied directly instead of read from a Wannier file.
   */
  template<typename eri_handler_t>
  void dielectric_function(
      eri_handler_t &eri, const std::string &params,
      const nda::array<ComplexType, 5> &projector_ksIai,
      const nda::array<long, 3> &band_window,
      const nda::array<RealType, 2> &kpts_crys,
      std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(params);
    methods::dielectric_function(eri.get_eri(), parser.get_root(),
                                 projector_ksIai, band_window, kpts_crys,
                                 std::move(local_polarizabilities));
  }

  auto pade(nda::array<ComplexType, 2> A_iw, nda::array<ComplexType, 1> iw_mesh,
            double w_min, double w_max, long Nw, int Nfit, double eta, bool is_iw_pos_only)
  -> std::tuple<nda::array<ComplexType, 2>, nda::array<ComplexType, 1>> {
    auto w_grid = analyt_cont::AC_t::w_grid(w_min, w_max, Nw, eta);
    return std::make_tuple(
      methods::pade(std::move(A_iw), std::move(iw_mesh), w_min, w_max, Nw, eta, is_iw_pos_only, Nfit), 
      w_grid
    );
  }

  // public template instantiation: THC only (Cholesky ERIs carry no auxiliary-basis heads)
  template void dielectric_function_with_projector_from_h5(
    ThcCoulomb&, const std::string&,
    std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

  template void dielectric_function(
    ThcCoulomb&, const std::string&,
    const nda::array<ComplexType, 5> &,
    const nda::array<long, 3> &,
    const nda::array<RealType, 2> &,
    std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

} // coqui_py

#include "post_proc_module.wrap.cxx"
