# Evolving BSSNOK formulation using Dendro

> **RIT refinement modes & parameters** (Sphere-in-Sphere / Box-in-Box and
> constraint-based refinement, plus the `BSSN_REFINEMENT_MODE` renumbering and new
> parameters) are documented in [doc/REFINEMENT_AND_PARAMS_rit.md](doc/REFINEMENT_AND_PARAMS_rit.md).

## CMake Flags
* BSSN_ENABLE_VTU_OUTPUT (ON) :  Enables .pvtu output that is used for solution visualization
* BSSN_COMPUTE_CONSTRAINTS (ON) : Compute BSSN constraint violations
* BSSN_ENABLE_VTU_CONSTRAINT_OUTPUT (ON) : Output constraint variables to .pvtu 
* BSSN_ETA_FUNCTION (OFF) : Enable eta damping as a function
* BSSN_ENABLE_CUDA (OFF) : Enable Cuda execution for the BSSNSolver (tested, but not highly optmized)
* BSSN_GAUGE_ROCHESTER (OFF) : Turn on the rochester gauge conditions (not tested)
* BSSN_EXTRACT_BH_LOCATIONS (ON) : Extract punture location 
* BSSN_REFINE_BASE_EH (OFF) : Force refinement based on the puncture location
* BSSN_EXTRACT_GRAVITATIONAL_WAVES (ON) : Enable extraction of GWs (psi4 extraction)
* BSSN_USE_4TH_ORDER_DERIVS (OFF) : Use 4th order spatial derivatives
* BSSN_USE_6TH_ORDER_DERIVS (ON) : Use 6th order spatial derivatives 
* BSSN_USE_8TH_ORDER_DERIVS (OFF) : Use 8th order spatial derivatives (Not Tested)
* BSSN_ENABLE_AVX (OFF): Use AVX vectorization for RHS and derivative computations (compiler based auto vectorization)

