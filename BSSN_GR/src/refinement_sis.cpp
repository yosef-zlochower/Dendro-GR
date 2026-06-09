/**
 * @brief RIT Sphere-in-Sphere / Box-in-Box and constraint-based refinement.
 *        Ported from the legacy chi-tests-SiS-limit branch (dataUtils.cpp) and
 *        isolated here to keep the diff against master small.
 *        See BSSN_GR/doc/REFINEMENT_AND_PARAMS_rit.md.
 */

#include "refinement_sis.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "dataUtils.h"  // brings mesh.h / TreeNode / point / grDef / parameters / normL2
#include "oct2vtk.h"

// OCT_IGNORE is an internal marker used by CONSTRAINT_ERROR ("not in WAMR
// region"). Newer dendrolib defines it in mesh.h; define it locally otherwise.
#ifndef OCT_IGNORE
#define OCT_IGNORE 10u
#endif

namespace bssn {

// ---------------------------------------------------------------------------
// Geometry helpers (file-local).
// ---------------------------------------------------------------------------
static inline double point_linf(const Point& p) {
    double lx = std::abs(p.x());
    double ly = std::abs(p.y());
    double lz = std::abs(p.z());
    return std::max(std::max(lx, ly), lz);
}

static inline double min_distance_cell_to_point_1D(const double xmin,
                                                   const double xmax,
                                                   const double x) {
    double lx = 0;
    if (xmin <= x && x <= xmax) {
        lx = 0;
    } else if (x <= xmin) {
        lx = xmin - x;
    } else {
        lx = x - xmax;
    }
    return lx;
}

// Distance from a point to a cell: Euclidean for BSSN_BOX_TYPE==0 (sphere),
// L-infinity otherwise (box). This is the Sphere-in-Sphere vs Box-in-Box switch.
static inline double min_distance_cell_to_point(const Point& p_min,
                                                const Point& p_max,
                                                const Point& p) {
    const double lx = min_distance_cell_to_point_1D(p_min.x(), p_max.x(), p.x());
    const double ly = min_distance_cell_to_point_1D(p_min.y(), p_max.y(), p.y());
    const double lz = min_distance_cell_to_point_1D(p_min.z(), p_max.z(), p.z());
    if (bssn::BSSN_BOX_TYPE == 0) {
        return sqrt(lx * lx + ly * ly + lz * lz);
    } else {
        return std::max(std::max(lx, ly), lz);
    }
}

// ---------------------------------------------------------------------------
// CONSTRAINT_ERROR : constraint-based WAMR.
// ---------------------------------------------------------------------------
bool isReMeshWAMRConstraint(
    ot::Mesh* pMesh, const Point* bhLoc, const double** unzippedcVec,
    const unsigned int varId_grad_grad2_chi_expression,
    std::function<double(double, double, double, double*)> wavelet_tol,
    double amr_coarse_fac) {
    bool isOctChange                 = false;
    bool isOctChange_g               = false;
    const unsigned int eleLocalBegin = pMesh->getElementLocalBegin();
    const unsigned int eleLocalEnd   = pMesh->getElementLocalEnd();

    std::vector<unsigned int> refine_flags;
    std::vector<unsigned int> refine_flags_WAMR;

    if (pMesh->isActive()) {
        refine_flags = bssn::isRemeshSinSInitHelper(pMesh, bhLoc);
        if (bssn::BSSN_CURRENT_RK_COORD_TIME >
            bssn::BSSN_SIS_TO_CONSTRAINT_WAMR_TRANSITION_TIME) {
            refine_flags_WAMR = bssn::isReMeshWAMRConstraintHelper(
                pMesh, bhLoc, unzippedcVec, varId_grad_grad2_chi_expression,
                wavelet_tol, amr_coarse_fac);
            for (unsigned int ele = eleLocalBegin; ele < eleLocalEnd; ele++) {
                if (refine_flags_WAMR[ele - eleLocalBegin] != OCT_IGNORE) {
                    refine_flags[ele - eleLocalBegin] =
                        refine_flags_WAMR[ele - eleLocalBegin];
                }
            }
        }
        isOctChange = pMesh->setMeshRefinementFlags(refine_flags);
    }
    MPI_Allreduce(&isOctChange, &isOctChange_g, 1, MPI_CXX_BOOL, MPI_LOR,
                  pMesh->getMPIGlobalCommunicator());
    return isOctChange_g;
}

std::vector<unsigned int> isReMeshWAMRConstraintHelper(
    ot::Mesh* pMesh, const Point* bhLoc, const double** unzippedcVec,
    const unsigned int varId_grad_grad2_chi_expression,
    std::function<double(double, double, double, double*)> wavelet_tol,
    double amr_coarse_fac) {
    std::vector<unsigned int> refine_flags;
    const double r_near[2] = {bssn::BSSN_BH1_AMR_R, bssn::BSSN_BH2_AMR_R};

    const unsigned int eleLocalBegin = pMesh->getElementLocalBegin();
    const unsigned int eleLocalEnd   = pMesh->getElementLocalEnd();
    bool isOctChange                 = false;
    bool isOctChange_g               = false;
    Point d1, d2, temp;

    const double* bssn_box_radii_at[] = {bssn::BSSN_BOX_RADII_1,
                                         bssn::BSSN_BOX_RADII_2};

    const unsigned int eOrder = pMesh->getElementOrder();
    const double dBH          = (BSSN_BH_LOC[0] - BSSN_BH_LOC[1]).abs();
    const unsigned int refLevMin =
        std::min(bssn::BSSN_BH1_MAX_LEV, bssn::BSSN_BH2_MAX_LEV);

    // per-element wavelet error (used for the optional VTU diagnostic dump)
    double* constraint_error_ptr = pMesh->createElementVector(0.0, 1);

    if (pMesh->isActive()) {
        if (!pMesh->getMPIRank()) printf("BH coord sep: %.8E \n", dBH);

        const RefElement* refEl    = pMesh->getReferenceElement();
        wavelet::WaveletEl* wrefEl = new wavelet::WaveletEl((RefElement*)refEl);

        refine_flags.resize(pMesh->getNumLocalMeshElements(), OCT_NO_CHANGE);
        const ot::TreeNode* pNodes = pMesh->getAllElements().data();

        double wtol_val = 0;

        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        const unsigned int nx                 = (2 * eOrder + 1);
        const unsigned int ny                 = (2 * eOrder + 1);
        const unsigned int nz                 = (2 * eOrder + 1);

        const unsigned int sz_per_dof = nx * ny * nz;
        const unsigned int isz[]      = {nx, ny, nz};
        std::vector<double> eVecTmp;
        eVecTmp.resize(sz_per_dof);

        std::vector<double> wCout;
        wCout.resize(sz_per_dof);

        for (unsigned int blk = 0; blk < blkList.size(); blk++) {
            const unsigned int pw    = blkList[blk].get1DPadWidth();
            const unsigned int bflag = blkList[blk].getBlkNodeFlag();
            assert(pw == (eOrder >> 1u));

            for (unsigned int ele = blkList[blk].getLocalElementBegin();
                 ele < blkList[blk].getLocalElementEnd(); ele++) {
                const bool isBdyOct = pMesh->isBoundaryOctant(ele);
                const double oct_dx =
                    (1u << (m_uiMaxDepth - pNodes[ele].getLevel())) /
                    (double(eOrder));

                Point oct_pt1 = Point(pNodes[ele].minX(), pNodes[ele].minY(),
                                      pNodes[ele].minZ());
                Point oct_pt2 = Point(pNodes[ele].minX() + oct_dx,
                                      pNodes[ele].minY() + oct_dx,
                                      pNodes[ele].minZ() + oct_dx);
                Point domain_pt1, domain_pt2, dx_domain;
                pMesh->octCoordToDomainCoord(oct_pt1, domain_pt1);
                pMesh->octCoordToDomainCoord(oct_pt2, domain_pt2);
                dx_domain               = domain_pt2 - domain_pt1;
                const unsigned int ln   = 1u << (m_uiMaxDepth - pNodes[ele].getLevel());
                double hx[3]            = {dx_domain.x(), dx_domain.y(),
                                           dx_domain.z()};
                const double tol_ele = wavelet_tol(domain_pt1.x(), domain_pt1.y(),
                                                   domain_pt1.z(), hx);
                unsigned int punct_id = 0;

                const double x_min = pNodes[ele].minX();
                const double y_min = pNodes[ele].minY();
                const double z_min = pNodes[ele].minZ();

                const double x_max  = pNodes[ele].minX() + ln;
                const double y_max  = pNodes[ele].minY() + ln;
                const double z_max  = pNodes[ele].minZ() + ln;
                const Point oct_min = Point(x_min, y_min, z_min);
                const Point oct_max = Point(x_max, y_max, z_max);
                Point coord_min;
                Point coord_max;
                pMesh->octCoordToDomainCoord(oct_min, coord_min);
                pMesh->octCoordToDomainCoord(oct_max, coord_max);

                const double rp1 =
                    min_distance_cell_to_point(coord_min, coord_max, bhLoc[0]);
                const double rp2 =
                    min_distance_cell_to_point(coord_min, coord_max, bhLoc[1]);

                if (rp1 < rp2) {
                    punct_id = 0;
                } else {
                    punct_id = 1;
                }
                const double rp = std::min(rp1, rp2);

                pMesh->getUnzipElementalNodalValues(
                    unzippedcVec[varId_grad_grad2_chi_expression], blk, ele,
                    eVecTmp.data(), true);

                // compute the wavelet coefficients of the chi constraint.
                // BSSN_REL_ERR_MIN is the relative-error floor, passed as the
                // trailing arg to compute_wavelets_3D. This is intentionally
                // unguarded: dendrolib MUST provide a compute_wavelets_3D that
                // accepts it (build against the dendrolib fork — see RESYNC.md
                // Step 8). If it does not, the build FAILS here by design, so
                // rel_min can never be silently dropped to the stock 4-arg form.
                wrefEl->compute_wavelets_3D((double*)(eVecTmp.data()), isz, wCout,
                                            isBdyOct, bssn::BSSN_REL_ERR_MIN);
                wtol_val = (normL2(wCout.data(), wCout.size())) /
                           sqrt(wCout.size());
                constraint_error_ptr[ele] = wtol_val;

                {
                    const unsigned int ln2 =
                        1u << (m_uiMaxDepth - pNodes[ele].getLevel());
                    const double hxm    = ln2 / (double)(eOrder);
                    const double x      = pNodes[ele].minX() + eOrder / 2 * hxm;
                    const double y      = pNodes[ele].minY() + eOrder / 2 * hxm;
                    const double z      = pNodes[ele].minZ() + eOrder / 2 * hxm;
                    const Point oct_mid = Point(x, y, z);
                    Point tmp;
                    pMesh->octCoordToDomainCoord(oct_mid, tmp);
                    const double rad2 =
                        tmp.x() * tmp.x() + tmp.y() * tmp.y() + tmp.z() * tmp.z();
                    if (rad2 > 0.8 * bssn::BSSN_CURRENT_RK_COORD_TIME *
                                   bssn::BSSN_CURRENT_RK_COORD_TIME ||
                        rp < bssn::BSSN_INNER_SIS_REGION_OUTER_BOUND) {
                        refine_flags[(ele - eleLocalBegin)] = OCT_IGNORE;
                        continue;
                    }
                }

                unsigned int refine_flag_temp;
                const double l_max = wtol_val;
                if (l_max > tol_ele) {
                    refine_flag_temp = OCT_SPLIT;
                } else if (l_max < amr_coarse_fac * tol_ele) {
                    refine_flag_temp = OCT_COARSE;
                } else {
                    refine_flag_temp = OCT_NO_CHANGE;
                }

                int level_difference = 0;
                for (int level = 0; level < (int)bssn::BSSN_BOX_NUM_LEVELS[punct_id];
                     level++) {
                    if (rp >= bssn_box_radii_at[punct_id][level]) {
                        level_difference =
                            (int)(pNodes[(ele)].getLevel() + MAXDEAPTH_LEVEL_DIFF +
                                  1) -
                            (int)(bssn::BSSN_MINDEPTH_SIS + level);
                        break;
                    }
                }

                // SiS limits how far WAMR may de/refine relative to the SiS level.
                if (level_difference < -1) {
                    refine_flag_temp = OCT_SPLIT;
                }
                if (level_difference > 1) {
                    refine_flag_temp = OCT_COARSE;
                }
                if (level_difference == -1 && refine_flag_temp == OCT_COARSE) {
                    refine_flag_temp = OCT_NO_CHANGE;
                }
                if (level_difference == 1 && refine_flag_temp == OCT_SPLIT) {
                    refine_flag_temp = OCT_NO_CHANGE;
                }

                refine_flags[(ele - eleLocalBegin)] = refine_flag_temp;
            }
        }

        // optional diagnostic: dump the per-element wavelet error to VTU.
        if (bssn::BSSN_RIT_DUMP_WAVELET_ERROR &&
            (BSSN_CURRENT_RK_STEP == 0 ||
             BSSN_CURRENT_RK_STEP % BSSN_IO_OUTPUT_FREQ == 0)) {
            const char* cell_data_names[]    = {"wtol_error"};
            unsigned int num_cell_vars       = 1;
            const double* cell_data_pointers[] = {constraint_error_ptr};
            std::ostringstream filename;
            filename << BSSN_VTU_FILE_PREFIX << "_wavelet_error_"
                     << std::setfill('0') << std::setw(5) << TEMP_BSSN_STEP_VAL;
            io::vtk::mesh2vtuFine(pMesh, filename.str().c_str(), 0, NULL, NULL, 0,
                                  NULL, NULL, num_cell_vars, cell_data_names,
                                  cell_data_pointers, false);
        }

        delete wrefEl;
    }

    pMesh->destroyVector(constraint_error_ptr);
    return refine_flags;
}

// ---------------------------------------------------------------------------
// CONSTRAINT : value-based refinement on log10|grad2_chi/chi^2|.
// ---------------------------------------------------------------------------
bool isRemeshConstraint(ot::Mesh* pMesh, const Point* bhLoc,
                        const double** unzippedcVec,
                        const unsigned int varId_grad2_chi,
                        const double** unzippedVec,
                        const unsigned int varId_chi) {
    bool isOctChange                 = false;
    bool isOctChange_g               = false;
    const unsigned int eleLocalBegin = pMesh->getElementLocalBegin();
    const unsigned int eleLocalEnd   = pMesh->getElementLocalEnd();

    std::vector<unsigned int> refine_flags;

    if (pMesh->isActive()) {
        // TODO(rit): suspected bug flagged by student. The condition below uses
        // '=' (assignment) not '==', which also clobbers the global
        // BSSN_CURRENT_RK_STEP. Preserved verbatim from chi-tests-SiS-limit
        // pending separate review; do NOT "fix" without verifying intent.
        if (bssn::BSSN_CURRENT_RK_STEP = 0) {
            refine_flags = bssn::isRemeshSinSInitHelper(pMesh, bhLoc);
        } else {
            refine_flags = bssn::isRemeshConstraintHelper(
                pMesh, bhLoc, unzippedcVec, varId_grad2_chi, unzippedVec,
                varId_chi);
        }
        isOctChange = pMesh->setMeshRefinementFlags(refine_flags);
    }
    MPI_Allreduce(&isOctChange, &isOctChange_g, 1, MPI_CXX_BOOL, MPI_LOR,
                  pMesh->getMPIGlobalCommunicator());
    return isOctChange_g;
}

std::vector<unsigned int> isRemeshConstraintHelper(
    ot::Mesh* pMesh, const Point* bhLoc, const double** unzippedcVec,
    const unsigned int varId_grad2_chi, const double** unzippedVec,
    unsigned int varId_chi) {
    const unsigned int eleLocalBegin = pMesh->getElementLocalBegin();
    bool isOctChange                 = false;
    bool isOctChange_g               = false;

    std::vector<unsigned int> refine_flags;
    if (pMesh->isActive()) {
        ot::TreeNode* pNodes =
            (ot::TreeNode*)&(*(pMesh->getAllElements().begin()));

        const std::vector<ot::Block>& blkList = pMesh->getLocalBlockList();
        unsigned int sz[3];
        unsigned int ei[3];
        refine_flags.resize(pMesh->getNumLocalMeshElements(), OCT_NO_CHANGE);
        const unsigned int eOrder = pMesh->getElementOrder();
        for (unsigned int b = 0; b < blkList.size(); b++) {
            const ot::TreeNode blkNode = blkList[b].getBlockNode();

            sz[0] = blkList[b].getAllocationSzX();
            sz[1] = blkList[b].getAllocationSzY();
            sz[2] = blkList[b].getAllocationSzZ();

            const unsigned int bflag  = blkList[b].getBlkNodeFlag();
            const unsigned int offset = blkList[b].getOffset();

            const unsigned int regLev = blkList[b].getRegularGridLev();
            const unsigned int eleIndexMax =
                (1u << (regLev - blkNode.getLevel())) - 1;
            const unsigned int eleIndexMin = 0;
            for (unsigned int ele = blkList[b].getLocalElementBegin();
                 ele < blkList[b].getLocalElementEnd(); ele++) {
                ei[0] = (pNodes[ele].getX() - blkNode.getX()) >>
                        (m_uiMaxDepth - regLev);
                ei[1] = (pNodes[ele].getY() - blkNode.getY()) >>
                        (m_uiMaxDepth - regLev);
                ei[2] = (pNodes[ele].getZ() - blkNode.getZ()) >>
                        (m_uiMaxDepth - regLev);

                if ((bflag & (1u << OCT_DIR_LEFT)) && ei[0] == eleIndexMin)
                    continue;
                if ((bflag & (1u << OCT_DIR_DOWN)) && ei[1] == eleIndexMin)
                    continue;
                if ((bflag & (1u << OCT_DIR_BACK)) && ei[2] == eleIndexMin)
                    continue;

                if ((bflag & (1u << OCT_DIR_RIGHT)) && ei[0] == eleIndexMax)
                    continue;
                if ((bflag & (1u << OCT_DIR_UP)) && ei[1] == eleIndexMax)
                    continue;
                if ((bflag & (1u << OCT_DIR_FRONT)) && ei[2] == eleIndexMax)
                    continue;

                int level = 1 + pNodes[ele].getLevel() + MAXDEAPTH_LEVEL_DIFF;

                int num_coarse     = 0;
                int num_no_change  = 0;
                int num_split      = 0;
                unsigned int rf    = OCT_COARSE;
                for (unsigned int k = 3; k < eOrder + 1 + 3; k++)
                    for (unsigned int j = 3; j < eOrder + 1 + 3; j++)
                        for (unsigned int i = 3; i < eOrder + 1 + 3; i++) {
                            double D2Chi =
                                unzippedcVec[varId_grad2_chi]
                                            [offset +
                                             (ei[2] * eOrder + k) * sz[0] * sz[1] +
                                             (ei[1] * eOrder + j) * sz[0] +
                                             (ei[0] * eOrder + i)];
                            double chi =
                                unzippedVec[varId_chi]
                                           [offset +
                                            (ei[2] * eOrder + k) * sz[0] * sz[1] +
                                            (ei[1] * eOrder + j) * sz[0] +
                                            (ei[0] * eOrder + i)];
                            double LogAbsChiExpression =
                                log10(fabs(D2Chi / pow(chi, 2)));
                            int chi_index = bssn::BSSN_CHI_NUM_VALUES;
                            while (true) {
                                chi_index--;
                                if (LogAbsChiExpression >
                                    bssn::BSSN_CHI_VALUES[chi_index]) {
                                    break;
                                }
                                if (chi_index < 0) {
                                    std::cerr
                                        << "chi reference index is negative: "
                                        << chi_index << std::endl;
                                    MPI_Abort(MPI_COMM_WORLD, -1);
                                }
                            }
                            if (level < chi_index + (int)bssn::BSSN_MINDEPTH_SIS) {
                                num_split++;
                            } else if (level ==
                                       chi_index + (int)bssn::BSSN_MINDEPTH_SIS) {
                                num_no_change++;
                            } else {
                                num_coarse++;
                            }
                        }
                if (num_no_change >= num_coarse & num_no_change > num_split) {
                    rf = OCT_NO_CHANGE;
                } else if (num_split >= num_no_change & num_split >= num_coarse) {
                    rf = OCT_SPLIT;
                }
                refine_flags.at(ele - eleLocalBegin) = rf;
            }
        }
    }
    return refine_flags;
}

// ---------------------------------------------------------------------------
// Shared SiS helper: per-element flags about the punctures (no mesh update).
// ---------------------------------------------------------------------------
std::vector<unsigned int> isRemeshSinSInitHelper(ot::Mesh* pMesh,
                                                 const Point* bhLoc) {
    const unsigned int eleLocalBegin = pMesh->getElementLocalBegin();
    const unsigned int eleLocalEnd   = pMesh->getElementLocalEnd();

    const double* bssn_box_radii_at[] = {bssn::BSSN_BOX_RADII_1,
                                         bssn::BSSN_BOX_RADII_2};

    std::vector<unsigned int> refine_flags;
    if (pMesh->isActive()) {
        const ot::TreeNode* pNodes = pMesh->getAllElements().data();

        refine_flags.resize(pMesh->getNumLocalMeshElements(), OCT_NO_CHANGE);

        for (unsigned int ele = eleLocalBegin; ele < eleLocalEnd; ele++) {
            const unsigned int ln = 1u << (m_uiMaxDepth - pNodes[ele].getLevel());
            unsigned int punct_id = 0;

            const double x_min = pNodes[ele].minX();
            const double y_min = pNodes[ele].minY();
            const double z_min = pNodes[ele].minZ();

            const double x_max  = pNodes[ele].minX() + ln;
            const double y_max  = pNodes[ele].minY() + ln;
            const double z_max  = pNodes[ele].minZ() + ln;
            const Point oct_min = Point(x_min, y_min, z_min);
            const Point oct_max = Point(x_max, y_max, z_max);
            Point coord_min;
            Point coord_max;
            pMesh->octCoordToDomainCoord(oct_min, coord_min);
            pMesh->octCoordToDomainCoord(oct_max, coord_max);

            const double rp1 =
                min_distance_cell_to_point(coord_min, coord_max, bhLoc[0]);
            const double rp2 =
                min_distance_cell_to_point(coord_min, coord_max, bhLoc[1]);

            if (rp1 < rp2) {
                punct_id = 0;
            } else {
                punct_id = 1;
            }
            const double rp = std::min(rp1, rp2);

            for (int level = 0; level < (int)bssn::BSSN_BOX_NUM_LEVELS[punct_id];
                 level++) {
                if (rp >= bssn_box_radii_at[punct_id][level]) {
                    if ((pNodes[ele].getLevel() + MAXDEAPTH_LEVEL_DIFF + 1) >
                        bssn::BSSN_MINDEPTH_SIS + level) {
                        refine_flags[ele - eleLocalBegin] = OCT_COARSE;
                    } else if ((pNodes[ele].getLevel() + MAXDEAPTH_LEVEL_DIFF +
                                1) < bssn::BSSN_MINDEPTH_SIS + level) {
                        refine_flags[ele - eleLocalBegin] = OCT_SPLIT;
                    } else {
                        refine_flags[ele - eleLocalBegin] = OCT_NO_CHANGE;
                    }
                    break;
                }
            }
        }
    }
    return refine_flags;
}

// ---------------------------------------------------------------------------
// SPHERE_IN_SPHERE / Box-in-Box static refinement.
// ---------------------------------------------------------------------------
bool isRemeshSinS(ot::Mesh* pMesh, const Point* bhLoc) {
    const unsigned int eleLocalBegin = pMesh->getElementLocalBegin();
    const unsigned int eleLocalEnd   = pMesh->getElementLocalEnd();
    bool isOctChange                 = false;

    const double* bssn_box_radii_at[] = {bssn::BSSN_BOX_RADII_1,
                                         bssn::BSSN_BOX_RADII_2};

    std::vector<unsigned int> refine_flags;
    if (pMesh->isActive()) {
        const ot::TreeNode* pNodes = pMesh->getAllElements().data();

        refine_flags.resize(pMesh->getNumLocalMeshElements(), OCT_NO_CHANGE);

        for (unsigned int ele = eleLocalBegin; ele < eleLocalEnd; ele++) {
            const unsigned int ln = 1u << (m_uiMaxDepth - pNodes[ele].getLevel());
            unsigned int punct_id = 0;

            const double x_min = pNodes[ele].minX();
            const double y_min = pNodes[ele].minY();
            const double z_min = pNodes[ele].minZ();

            const double x_max  = pNodes[ele].minX() + ln;
            const double y_max  = pNodes[ele].minY() + ln;
            const double z_max  = pNodes[ele].minZ() + ln;
            const Point oct_min = Point(x_min, y_min, z_min);
            const Point oct_max = Point(x_max, y_max, z_max);
            Point coord_min;
            Point coord_max;
            pMesh->octCoordToDomainCoord(oct_min, coord_min);
            pMesh->octCoordToDomainCoord(oct_max, coord_max);

            const double rp1 =
                min_distance_cell_to_point(coord_min, coord_max, bhLoc[0]);
            const double rp2 =
                min_distance_cell_to_point(coord_min, coord_max, bhLoc[1]);

            if (rp1 < rp2) {
                punct_id = 0;
            } else {
                punct_id = 1;
            }
            const double rp = std::min(rp1, rp2);

            for (int level = 0; level < (int)bssn::BSSN_BOX_NUM_LEVELS[punct_id];
                 level++) {
                if (rp >= bssn_box_radii_at[punct_id][level]) {
                    if ((pNodes[ele].getLevel() + MAXDEAPTH_LEVEL_DIFF + 1) >
                        bssn::BSSN_MINDEPTH_SIS + level) {
                        refine_flags[ele - eleLocalBegin] = OCT_COARSE;
                    } else if ((pNodes[ele].getLevel() + MAXDEAPTH_LEVEL_DIFF +
                                1) < bssn::BSSN_MINDEPTH_SIS + level) {
                        refine_flags[ele - eleLocalBegin] = OCT_SPLIT;
                    } else {
                        refine_flags[ele - eleLocalBegin] = OCT_NO_CHANGE;
                    }
                    break;
                }
            }
        }
        isOctChange = pMesh->setMeshRefinementFlags(refine_flags);
    }
    bool isOctChanged_g;
    MPI_Allreduce(&isOctChange, &isOctChanged_g, 1, MPI_CXX_BOOL, MPI_LOR,
                  pMesh->getMPIGlobalCommunicator());
    return isOctChanged_g;
}

}  // namespace bssn
