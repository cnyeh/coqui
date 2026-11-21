"""
Re-solve the impurity problem stored in a GW+EDMFT checkpoint with an impurity solver of choice,
without any GW, downfolding or embedding step.

Every GW+EDMFT iteration stores the impurity problem it handed to the solver (h0, Delta, the bare
and retarded interactions, the occupations) under dmft/iter{N}/impurity_{I}/inputs of the DMFT
checkpoint. This script reads that problem back and runs only the solver stage, which is how to
compare solvers on one and the same problem, to re-solve an iteration with more statistics, or to
try different solver settings. 

With save_results=True the new solution is written into the checkpoint, into the same iteration
(dmft/iter{N}/impurity_{I}/{results,solver_results}), replacing the stored one; work on a copy of the
checkpoint to keep the original run intact, as done here.

    mpirun -n N python 05_solve_impurity_from_chkpt.py
"""
import shutil

import numpy as np

import coqui
import coqui.dmft as coqui_dmft

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=2)

# the checkpoint of 02_gw_edmft_ctseg.py, copied so that the stored solution stays untouched
source_h5 = "svo_ctseg.dmft.h5"
chkpt_h5 = "svo_resolve.dmft.h5"
if coqui_mpi.root():
  shutil.copyfile(source_h5, chkpt_h5)
coqui_mpi.barrier()

params = {
    "edmft": {
        "chkpt_h5": chkpt_h5,
        # "iaft": {"wmax": 5, "eps": 1e-10},   # DLR mesh of the impurity problem (default: the stored one)

        # one block per impurity, as in the GW+EDMFT scripts; here the solver is CT-HYB with the
        # Kanamori interaction, on the problem that CT-SEG solved with the density-density one
        "impurity": [{
            "degenerate_blk": [[0, 1, 2, 3, 4, 5]],
            "chemical_potential": {"tolerance": 0.1},
            "solver": {
                "type": "cthyb",
                "h_int": "kanamori",
                "n_cycles": 2000000,           # total over all MPI ranks
                "n_warmup_cycles": 6000,
                "length_cycle": 60,
                "perform_tail_fit": True,
                "fit_max_moment": 7,
                "fit_min_w": 1.8,              # Ha
                "fit_max_w": 4.5,              # Ha
                "mu_search": {"n_cycles": 100000},
            },
        }],
    },
}

# dmft_iteration=-1: the latest stored iteration; imp_indices=None: every impurity of the checkpoint
results = coqui_dmft.solve_impurities_from_chkpt(coqui_mpi, dmft_iteration=-1, imp_indices=None,
                                                 params=params, save_results=True)

if coqui_mpi.root():
  for imp_index, result in enumerate(results):
    print(f"impurity {imp_index}: density {result['density']:.4f}, "
          f"average sign {result['average_sign']:.3f}, average order {result['average_order']:.1f}")
    # the impurity self-energy at the first Matsubara frequency, per gf_struct block
    for block, sigma in zip(result["gf_struct"], result["Sigma_iw_data"]):
      print(f"  Sigma_imp({block[0]}) at iw_0: {np.diag(sigma[len(sigma) // 2]).real}")
