/**
 * @brief RIT Sphere-in-Sphere / Box-in-Box and constraint-based refinement.
 *
 * Ported from the legacy chi-tests-SiS-limit branch (originally in dataUtils.cpp)
 * and isolated here to keep the diff against master small. See
 * BSSN_GR/doc/REFINEMENT_AND_PARAMS_rit.md for the user-facing description.
 *
 * Entry points (dispatched from BSSNCtx::is_remesh):
 *   - isRemeshSinS           : RefinementMode::SPHERE_IN_SPHERE
 *   - isRemeshConstraint     : RefinementMode::CONSTRAINT       (value-based)
 *   - isReMeshWAMRConstraint : RefinementMode::CONSTRAINT_ERROR (wavelet-based)
 *
 * NOTE: the legacy SIS_OUT_WAMR_IN combination mode and its helpers
 * (isRemeshSiSCombination / isRemeshSinSHelper / isReMeshWAMRHelper) were
 * intentionally NOT ported.
 */

#ifndef DENDRO_5_0_REFINEMENT_SIS_H
#define DENDRO_5_0_REFINEMENT_SIS_H

#include <functional>
#include <vector>

#include "mesh.h"
#include "parameters.h"
#include "point.h"

namespace bssn {

/**@brief Sphere-in-Sphere / Box-in-Box static refinement about the punctures.*/
bool isRemeshSinS(ot::Mesh* pMesh, const Point* bhLoc);

/**@brief Shared SiS helper: returns per-element refine flags (no mesh update).*/
std::vector<unsigned int> isRemeshSinSInitHelper(ot::Mesh* pMesh,
                                                 const Point* bhLoc);

/**@brief Value-based constraint refinement on log10|grad2_chi/chi^2|.*/
bool isRemeshConstraint(ot::Mesh* pMesh, const Point* bhLoc,
                        const double** unzippedcVec,
                        const unsigned int varId_grad2_chi,
                        const double** unzippedVec,
                        const unsigned int varId_chi);
std::vector<unsigned int> isRemeshConstraintHelper(
    ot::Mesh* pMesh, const Point* bhLoc, const double** unzippedcVec,
    const unsigned int varId_grad2_chi, const double** unzippedVec,
    unsigned int varId_chi);

/**@brief Constraint-based WAMR (wavelet error on a chi constraint), with a
 * time-gated SiS->WAMR transition and SiS level-limiting.*/
bool isReMeshWAMRConstraint(
    ot::Mesh* pMesh, const Point* bhLoc, const double** unzippedcVec,
    const unsigned int varId_grad_grad2_chi_expression,
    std::function<double(double, double, double, double*)> wavelet_tol,
    double amr_coarse_fac);
std::vector<unsigned int> isReMeshWAMRConstraintHelper(
    ot::Mesh* pMesh, const Point* bhLoc, const double** unzippedcVec,
    const unsigned int varId_grad_grad2_chi_expression,
    std::function<double(double, double, double, double*)> wavelet_tol,
    double amr_coarse_fac);

}  // namespace bssn

#endif  // DENDRO_5_0_REFINEMENT_SIS_H
