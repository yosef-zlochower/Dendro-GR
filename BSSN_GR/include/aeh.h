/**
 * @brief main apparent event horizon solvers code
 * @version 1.0
 * @date 2023-06-23
 * @author Milinda Fernando (milinda@oden.utexas.edu)
 * @copyright Copyright (c) 2019
 */

#pragma once

#include "mesh.h"
#include "mpi.h"
#include "grDef.h"
#include "daUtils.h"
#include "gsl/gsl_specfunc.h"
#include "gsl/gsl_integration.h"
#include "parameters.h"
#include "dvec.h"

namespace aeh
{
    enum AEHErrorType {SUCCESS, MAX_ITERATIONS_REACHED, FAIL};
    typedef std::pair<int, int> LM_MODE;
    typedef ot::DVector<DendroScalar, unsigned int> DVec;


    typedef struct
    {
        DVec grad_gt;
        DVec grad_chi;
        
        DVec gt;
        DVec At;
        DVec chi;
        DVec K;

    } AEH_VARS;


    
    template<typename Ctx, typename T>
    class SpectralAEHSolver
    {
        private:
            Ctx * m_ctx  = nullptr;

            std::vector<LM_MODE> m_sph_modes;
            
            unsigned int m_num_theta;
            
            unsigned int m_num_phi;
            
            gsl_integration_fixed_workspace * m_quad_theta = nullptr;
            
            gsl_integration_fixed_workspace * m_quad_phi   = nullptr;

            AEH_VARS * m_aeh_vars;

            std::vector<T> m_sin_qtheta;
            std::vector<T> m_cos_qtheta;

            std::vector<T> m_sin_qphi;
            std::vector<T> m_cos_qphi;

            
        public:
            /**
             * @brief Construct a new Spectral AEH solver object
             * 
             * @param ctx   : Application Ctx, i.e., GR spatial variables. 
             * @param l_max : Spectral expansion using l=0 to l=l_max, and corresponding m-modes 
             * @param azimuthal_symmetry : Use azimuthal symmetry or not. (m=0) modes
             */
            SpectralAEHSolver(Ctx* ctx, unsigned int l_max, unsigned int q_theta, unsigned int q_phi, bool azimuthal_symmetry=false);
            
            /**@brief deconstructor for AEH solver object*/
            ~SpectralAEHSolver();

            const std::vector<LM_MODE> & get_lm_modes() const {return m_sph_modes;}

            unsigned int get_num_lm_modes() const {return m_sph_modes.size();}

            /**
             * @brief Computes the real spherical harmonics
             * 
             * @param l l modes 
             * @param m  m mode
             * @param theta : polar angle (0,pi) 
             * @param phi : azimuthal angle (0, 2pi)
             * @return Y_lm(theta, phi)
             */
            inline static double real_spherical_harmonic(int l, int m, T theta, T phi)
            {
                assert(abs(m)<=l);
                if(m==0)
                    return gsl_sf_legendre_sphPlm(l,m, cos(theta));
                else if (m>0)
                    return ((-1)^m) *sqrt(2) * gsl_sf_legendre_sphPlm(l,m, cos(theta)) * cos(m * phi);
                else
                    return ((-1)^m) *sqrt(2) * gsl_sf_legendre_sphPlm(l, abs(m), cos(theta)) * sin( abs(m) * phi);
               
            }

            /**
             * @brief Level set function computation for the AEH surface. 
             * @param m_uiMesh 
             * @param aeh_f_ptr 
             * @param h_qs 
             * @return int 
             */
            int eval_aeh_level_set(const Point& origin, ot::Mesh* m_uiMesh, T* const aeh_f_ptr, const T* const h_qs, DendroScalar r_max);

            /**
             * @brief main AEH solver update
             * @param ctx 
             * @param r_init 
             * @param max_iter 
             * @return AEHErrorType 
             */
            AEHErrorType solve(const Point& origin, Ctx * ctx, const T* const h_init, T*  h_qs, unsigned int max_iter, DendroScalar rel_eps, DendroScalar abs_eps, DendroScalar r_max);


        private:
            DendroScalar integral_S2_h(const Point& origin, Ctx* ctx, const T* const h_qs, DVec& aeh_f, DVec& aeh_h, std::vector<T>& interp_coords, DendroScalar r_max)
            {
                ot::Mesh* m_uiMesh = ctx->get_mesh();
                if(!(m_uiMesh->isActive()))
                    return 0.0;
                

                T * aeh_f_ptr = aeh_f.get_vec_ptr();
                T * aeh_h_ptr = aeh_h.get_vec_ptr();
                
                Point   grid_limits[2];
                Point domain_limits[2];

                grid_limits[0]   = Point(bssn::BSSN_OCTREE_MIN[0], bssn::BSSN_OCTREE_MIN[1], bssn::BSSN_OCTREE_MIN[2]);
                grid_limits[1]   = Point(bssn::BSSN_OCTREE_MAX[0], bssn::BSSN_OCTREE_MAX[1], bssn::BSSN_OCTREE_MAX[2]);

                domain_limits[0] = Point(bssn::BSSN_COMPD_MIN[0], bssn::BSSN_COMPD_MIN[1], bssn::BSSN_COMPD_MIN[2]);
                domain_limits[1] = Point(bssn::BSSN_COMPD_MAX[0], bssn::BSSN_COMPD_MAX[1], bssn::BSSN_COMPD_MAX[2]);

                this->eval_aeh_level_set(origin, m_uiMesh, aeh_f_ptr, h_qs, r_max);
                ctx->aeh_expansion(origin, m_aeh_vars, aeh_f, aeh_h);

                m_uiMesh->readFromGhostBegin(aeh_h.get_vec_ptr(), 1);

                unsigned int num_angular_pts   = m_num_theta * m_num_phi;
                const double * const m_qtheta  = m_quad_theta->x;
                const double * const m_qphi    = m_quad_phi->x;

                for(unsigned int qt=0; qt < m_num_theta; qt++)
                    for(unsigned int qp=0; qp < m_num_phi; qp++)
                    {
                        const unsigned int q_idx = qt * m_num_phi + qp;

                        T r_val = (T)0;
                        for(unsigned int lm_idx=0; lm_idx < m_sph_modes.size(); lm_idx++)
                            r_val+= real_spherical_harmonic(m_sph_modes[lm_idx].first, m_sph_modes[lm_idx].second, m_qtheta[qt], m_qphi[qp]) * h_qs[lm_idx];
                        
                        interp_coords[3 * q_idx + 0] = origin.x() + (r_val) * sin(m_qtheta[qt]) * cos(m_qphi[qp]); //m_sin_qtheta[qt] * m_cos_qphi[qp];
                        interp_coords[3 * q_idx + 1] = origin.y() + (r_val) * sin(m_qtheta[qt]) * sin(m_qphi[qp]); //m_sin_qtheta[qt] * m_sin_qphi[qp];
                        interp_coords[3 * q_idx + 2] = origin.z() + (r_val) * cos(m_qtheta[qt]);                   //m_cos_qtheta[qt];
                        
                    }

                m_uiMesh->readFromGhostEnd(aeh_h.get_vec_ptr(), 1);

                std::vector<unsigned int> valid_idx;
                valid_idx.clear();

                std::vector<T> aeh_h_inp;
                aeh_h_inp.resize(num_angular_pts);
                ot::da::interpolateToCoords(m_uiMesh, aeh_h_ptr, interp_coords.data(), interp_coords.size(), grid_limits, domain_limits, aeh_h_inp.data(), valid_idx);

                DendroScalar result   = 0;
                DendroScalar result_g = 0;
                for(unsigned int idx=0;idx < valid_idx.size(); idx++)
                {
                    const unsigned int qp = valid_idx[idx] % m_num_phi;
                    const unsigned int qt = (valid_idx[idx] -qp) / m_num_phi;

                    DendroScalar s = aeh_h_inp[valid_idx[idx]];
                    result+= (s) * m_quad_theta->weights[qt] * m_quad_phi->weights[qp];
                }

                par::Mpi_Allreduce(&result, &result_g, 1, MPI_SUM, m_uiMesh->getMPICommunicator());
                return result_g;

            }

            DendroScalar rhs_00(const Point& origin, Ctx* ctx, DendroScalar a, T* h_qs, DVec& aeh_f, DVec& aeh_h, std::vector<T>& interp_coords, DendroScalar r_max)
            {

                ot::Mesh* m_uiMesh = ctx->get_mesh();
                if(!(m_uiMesh->isActive()))
                    return -1;

                T * aeh_f_ptr = aeh_f.get_vec_ptr();
                T * aeh_h_ptr = aeh_h.get_vec_ptr();
                
                h_qs[0] = a;    
                Point grid_limits[2];
                Point domain_limits[2];
                grid_limits[0]   = Point(bssn::BSSN_OCTREE_MIN[0], bssn::BSSN_OCTREE_MIN[1], bssn::BSSN_OCTREE_MIN[2]);
                grid_limits[1]   = Point(bssn::BSSN_OCTREE_MAX[0], bssn::BSSN_OCTREE_MAX[1], bssn::BSSN_OCTREE_MAX[2]);

                domain_limits[0] = Point(bssn::BSSN_COMPD_MIN[0], bssn::BSSN_COMPD_MIN[1], bssn::BSSN_COMPD_MIN[2]);
                domain_limits[1] = Point(bssn::BSSN_COMPD_MAX[0], bssn::BSSN_COMPD_MAX[1], bssn::BSSN_COMPD_MAX[2]);

                
                
                this->eval_aeh_level_set(origin, m_uiMesh, aeh_f_ptr, h_qs, r_max);
                ctx->aeh_expansion(origin, m_aeh_vars, aeh_f, aeh_h);

                m_uiMesh->readFromGhostBegin(aeh_h.get_vec_ptr(), 1);

                unsigned int num_angular_pts   = m_num_theta * m_num_phi;
                const double * const m_qtheta  = m_quad_theta->x;
                const double * const m_qphi    = m_quad_phi->x;

                for(unsigned int qt=0; qt < m_num_theta; qt++)
                    for(unsigned int qp=0; qp < m_num_phi; qp++)
                    {
                        const unsigned int q_idx = qt * m_num_phi + qp;

                        T r_val = (T)0;
                        for(unsigned int lm_idx=0; lm_idx < m_sph_modes.size(); lm_idx++)
                            r_val+= real_spherical_harmonic(m_sph_modes[lm_idx].first, m_sph_modes[lm_idx].second, m_qtheta[qt], m_qphi[qp]) * h_qs[lm_idx];
                        
                        interp_coords[3 * q_idx + 0] = origin.x() + (r_val) * sin(m_qtheta[qt]) * cos(m_qphi[qp]); //m_sin_qtheta[qt] * m_cos_qphi[qp];
                        interp_coords[3 * q_idx + 1] = origin.y() + (r_val) * sin(m_qtheta[qt]) * sin(m_qphi[qp]); //m_sin_qtheta[qt] * m_sin_qphi[qp];
                        interp_coords[3 * q_idx + 2] = origin.z() + (r_val) * cos(m_qtheta[qt]);                   //m_cos_qtheta[qt];
                        
                    }

                m_uiMesh->readFromGhostEnd(aeh_h.get_vec_ptr(), 1);

                std::vector<unsigned int> valid_idx;
                valid_idx.clear();

                std::vector<T> aeh_h_inp;
                aeh_h_inp.resize(num_angular_pts);

                ot::da::interpolateToCoords(m_uiMesh, aeh_h_ptr, interp_coords.data(), interp_coords.size(), grid_limits, domain_limits, aeh_h_inp.data(), valid_idx);
                // printf("rhs_{00}\n");
                // printArray_1D(interp_coords.data(), interp_coords.size());
                // printArray_1D(aeh_h_inp.data(), aeh_h_inp.size());

                DendroScalar result   = 0;
                DendroScalar result_g = 0;
                for(unsigned int idx=0;idx < valid_idx.size(); idx++)
                {
                    const unsigned int qp = valid_idx[idx] % m_num_phi;
                    const unsigned int qt = (valid_idx[idx] -qp) / m_num_phi;

                    DendroScalar s = aeh_h_inp[valid_idx[idx]];

                    for(unsigned int lm_idx=0; lm_idx < m_sph_modes.size(); lm_idx++)
                    {
                        int l  = m_sph_modes[lm_idx].first;
                        int m  = m_sph_modes[lm_idx].second;

                        s += -(l)*(l+1) * real_spherical_harmonic(l, m, m_qtheta[qt],  m_qphi[qp]) * h_qs[lm_idx];
                        
                    }
                    
                    result+= (s) * m_quad_theta->weights[qt] * m_quad_phi->weights[qp];
                }

                par::Mpi_Allreduce(&result, &result_g, 1, MPI_SUM, m_uiMesh->getMPICommunicator());
                return result_g;

            }

            DendroScalar solve_00_bisection(const Point& origin, DendroScalar a, DendroScalar b, Ctx * ctx, T*  h_qs, DVec& aeh_f, DVec& aeh_h, std::vector<T>& interp_coords, DendroScalar eps, DendroScalar r_max)
            {
                ot::Mesh * m_uiMesh = ctx->get_mesh();
                if(!(m_uiMesh->isActive()))
                    return -1.0;

                int rank = m_uiMesh->getMPIRank();
                int npes = m_uiMesh->getMPICommSize();
                
                DendroScalar h_00 = h_qs[0];

                DendroScalar f_a  = rhs_00(origin, ctx, a, h_qs, aeh_f, aeh_h, interp_coords, r_max);
                DendroScalar f_b  = rhs_00(origin, ctx, b, h_qs, aeh_f, aeh_h, interp_coords, r_max);
                
                if (f_a * f_b >= 0)
                {
                    if(!rank)
                        printf("lm=(0,0) initial interval (a,b) = (%.8E, %.8E) = (f(a), f(b)) = (%.8E, %.8E)\n", a, b, f_a, f_b);
                    
                    return -1.0;
                }

                DendroScalar   c = a;
                while ((b-a) >= eps)
                {
                    // Find middle point
                    c   = (a+b)/2;
                    DendroScalar f_c = rhs_00(origin, ctx, c, h_qs, aeh_f, aeh_h, interp_coords, r_max);  

                    if(!rank)
                        printf("lm=(0,0)  h_00 = %.8E ,  abs(H(h)) = %.8E\n", c, abs(f_c));
                        
                    // Check if middle point is root
                    if (abs(f_c) < eps )
                    {
                        return c;
                    }
                    DendroScalar f_a = rhs_00(origin, ctx, a, h_qs, aeh_f, aeh_h, interp_coords, r_max);  
                    // Decide the side to repeat the steps
                    if (f_c * f_a < 0)
                        b = c;
                    else
                        a = c;
                }
                
                return c;
            }

            DendroScalar solve_00_newton(const Point& origin, Ctx * ctx, T*  h_qs, DVec& aeh_f, DVec& aeh_h, std::vector<T>& interp_coords, DendroScalar eps, DendroScalar r_max)
            {
                ot::Mesh * m_uiMesh = ctx->get_mesh();
                if(!(m_uiMesh->isActive()))
                    return -1.0;

                int rank = m_uiMesh->getMPIRank();
                int npes = m_uiMesh->getMPICommSize();
                
                DendroScalar h_00 = h_qs[0];
                DendroScalar x    = h_00;

                DendroScalar relative_error = 0.0;
                unsigned int iter     =  0;
                unsigned int max_iter = 20;

                do
                {
                    DendroScalar dh  = sqrt(std::numeric_limits<DendroScalar>::epsilon()) * x;
                    DendroScalar fp  = rhs_00(origin, ctx, x + dh, h_qs, aeh_f, aeh_h, interp_coords, r_max);
                    DendroScalar fa  = rhs_00(origin, ctx, x     , h_qs, aeh_f, aeh_h, interp_coords, r_max);
                    DendroScalar fm  = rhs_00(origin, ctx, x - dh, h_qs, aeh_f, aeh_h, interp_coords, r_max);

                    DendroScalar grad_f = 0.5 * (fp - fm) / dh;
                    DendroScalar alpha  = 1.0;

                    DendroScalar b, fb;
                    
                    do
                    {
                        b     = x - alpha * (fa/grad_f);
                        fb    = rhs_00(origin, ctx, b     , h_qs, aeh_f, aeh_h, interp_coords, r_max);
                        alpha = alpha * 0.1;

                    }while((alpha > 1e-3) && ((abs(fb) > abs(fa)) || (b<0)) );
                    
                    relative_error      = abs(b-x)/abs(x);

                    if(!rank)
                        printf("  Newton iter=%03d\t h_00=%.8E\t abs(f)=%.8E\t relative_error=%.8E\n", iter, x, abs(fa), relative_error);

                    
                    iter+=1;
                    x= b;

                    
                }while((iter < max_iter) && (relative_error > eps));

                return x;
                
            }

    };

    template<typename Ctx, typename T>
    SpectralAEHSolver<Ctx, T>::SpectralAEHSolver(Ctx* ctx, unsigned int l_max, unsigned int q_theta, unsigned int q_phi, bool azimuthal_symmetry)
    {
        m_ctx = ctx;
        ot::Mesh * m_uiMesh = ctx->get_mesh();

        if(!m_uiMesh->isActive())
            return;

        const int rank = m_uiMesh->getMPIRank();
        
        if(azimuthal_symmetry)
        {
            for(unsigned int l=0; l <=l_max; l++)
                m_sph_modes.push_back(LM_MODE(l,0));
        }else
        {
            for(int l=0; l <=l_max; l++)
                for(int m=-l; m <=l; m++)
                    m_sph_modes.push_back(LM_MODE(l,m));
        }

        // need to initialize the quadrature grid
        m_num_theta = q_theta;
        m_num_phi   = q_phi;

        // need to initialize the q_theta based on the Gauss-Legendre quadrature
        m_quad_theta = gsl_integration_fixed_alloc(gsl_integration_fixed_legendre, m_num_theta, -1, 1      , 0.0, 0.0);
        m_quad_phi   = gsl_integration_fixed_alloc(gsl_integration_fixed_legendre, m_num_phi  , 0 , 2*M_PI , 0.0, 0.0);

        for(unsigned int i=0; i < m_num_theta; i++)
            m_quad_theta->x[i] = acos(m_quad_theta->x[i]);

        m_sin_qtheta.resize(m_num_theta);
        m_cos_qtheta.resize(m_num_theta);

        m_sin_qphi.resize(m_num_phi);
        m_cos_qphi.resize(m_num_phi);

        const double* const m_qtheta = m_quad_theta->x;
        const double* const m_qphi   = m_quad_phi->x;
        
        for(unsigned int qt=0; qt < m_num_theta; qt++)
        {
            m_sin_qtheta[qt] = sin(m_qtheta[qt]);
            m_cos_qtheta[qt] = cos(m_qtheta[qt]);
        }

        for(unsigned int qp=0; qp < m_num_phi; qp++)
        {
            m_sin_qphi[qp] = sin(m_qphi[qp]);
            m_cos_qphi[qp] = cos(m_qphi[qp]);
        }

        // printArray_1D(m_quad_theta->x, m_num_theta);
        // printArray_1D(m_quad_theta->weights, m_num_theta);

        // printArray_1D(m_quad_phi->x, m_num_phi);
        // printArray_1D(m_quad_phi->weights, m_num_phi);

        m_aeh_vars = new AEH_VARS();

        m_aeh_vars->grad_gt.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST, 18, true);
        m_aeh_vars->grad_chi.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST, 3, true);
        
        m_aeh_vars->gt.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,  6, true);
        m_aeh_vars->At.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,  6, true);
        m_aeh_vars->chi.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST, 1, true);
        m_aeh_vars->K.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_LOCAL_WITH_PADDING, ot::DVEC_LOC::HOST,   1, true);
        

        DVec& gr_vars              = ctx->get_evolution_vars();
        DendroScalar * gr_vars_ptr = gr_vars.get_vec_ptr();
    
        const unsigned int cg_sz   = m_uiMesh->getDegOfFreedom();
        const unsigned int uz_sz   = m_uiMesh->getDegOfFreedomUnZip();
        
        DendroScalar * gt          = &gr_vars_ptr[VAR::U_SYMGT0 * cg_sz];
        DendroScalar * At          = &gr_vars_ptr[VAR::U_SYMAT0 * cg_sz];
        DendroScalar * chi         = &gr_vars_ptr[VAR::U_CHI    * cg_sz];
        DendroScalar * K           = &gr_vars_ptr[VAR::U_K      * cg_sz];


        DendroScalar * gt_uz  = m_aeh_vars->gt.get_vec_ptr();
        DendroScalar * At_uz  = m_aeh_vars->At.get_vec_ptr();
        DendroScalar * chi_uz = m_aeh_vars->chi.get_vec_ptr();
        DendroScalar * K_uz   = m_aeh_vars->K.get_vec_ptr();
        
        

        m_uiMesh->readFromGhostBegin<DendroScalar>(gt , 6);
        m_uiMesh->readFromGhostBegin<DendroScalar>(At , 6);
        m_uiMesh->readFromGhostBegin<DendroScalar>(chi, 1);
        m_uiMesh->readFromGhostBegin<DendroScalar>(K  , 1);

        
        m_uiMesh->readFromGhostEnd<DendroScalar>(gt,  6);
        m_uiMesh->unzip(gt , gt_uz, 6);

        m_uiMesh->readFromGhostEnd<DendroScalar>(At,  6);
        m_uiMesh->unzip(At , At_uz, 6);


        m_uiMesh->readFromGhostEnd<DendroScalar>(chi, 1);
        m_uiMesh->readFromGhostEnd<DendroScalar>(K,   1);

        m_uiMesh->unzip(chi, chi_uz, 1);
        m_uiMesh->unzip(K  , K_uz  , 1);


        const Point pt_min(bssn::BSSN_COMPD_MIN[0],bssn::BSSN_COMPD_MIN[1],bssn::BSSN_COMPD_MIN[2]);
        const Point pt_max(bssn::BSSN_COMPD_MAX[0],bssn::BSSN_COMPD_MAX[1],bssn::BSSN_COMPD_MAX[2]);
        const unsigned int PW=bssn::BSSN_PADDING_WIDTH;

        const ot::Block* blkList     = m_uiMesh->getLocalBlockList().data();
        const unsigned int numBlocks = m_uiMesh->getLocalBlockList().size();

        for(unsigned int blk=0; blk<numBlocks; blk++)
        {
            const unsigned int offset = blkList[blk].getOffset();
            const unsigned int sz[3]={blkList[blk].getAllocationSzX(), blkList[blk].getAllocationSzY(), blkList[blk].getAllocationSzZ()};
            const unsigned int bflag=blkList[blk].getBlkNodeFlag();

            const DendroScalar dx = blkList[blk].computeDx(pt_min,pt_max);
            const DendroScalar dy = blkList[blk].computeDy(pt_min,pt_max);
            const DendroScalar dz = blkList[blk].computeDz(pt_min,pt_max);

            DendroScalar ptmin[3], ptmax[3];

            ptmin[0]=GRIDX_TO_X(blkList[blk].getBlockNode().minX())-PW*dx;
            ptmin[1]=GRIDY_TO_Y(blkList[blk].getBlockNode().minY())-PW*dy;
            ptmin[2]=GRIDZ_TO_Z(blkList[blk].getBlockNode().minZ())-PW*dz;

            ptmax[0]=GRIDX_TO_X(blkList[blk].getBlockNode().maxX())+PW*dx;
            ptmax[1]=GRIDY_TO_Y(blkList[blk].getBlockNode().maxY())+PW*dy;
            ptmax[2]=GRIDZ_TO_Z(blkList[blk].getBlockNode().maxZ())+PW*dz;

            const unsigned int n = sz[0]*sz[1]*sz[2];
            const unsigned int BLK_SZ=n;
            
            const unsigned int nx = sz[0];
            const unsigned int ny = sz[1];
            const unsigned int nz = sz[2];

            const double hx = (ptmax[0] - ptmin[0]) / (nx - 1);
            const double hy = (ptmax[1] - ptmin[1]) / (ny - 1);
            const double hz = (ptmax[2] - ptmin[2]) / (nz - 1);

            double * grad_0_chi = &m_aeh_vars->grad_chi.get_vec_ptr()[0 * uz_sz + offset];
            double * grad_1_chi = &m_aeh_vars->grad_chi.get_vec_ptr()[1 * uz_sz + offset];
            double * grad_2_chi = &m_aeh_vars->grad_chi.get_vec_ptr()[2 * uz_sz + offset];

            double * grad_0_gt0 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 0 * 3 *uz_sz + 0 * uz_sz + offset];
            double * grad_1_gt0 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 0 * 3 *uz_sz + 1 * uz_sz + offset];
            double * grad_2_gt0 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 0 * 3 *uz_sz + 2 * uz_sz + offset];

            double * grad_0_gt1 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 1 * 3 *uz_sz + 0 * uz_sz + offset];;
            double * grad_1_gt1 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 1 * 3 *uz_sz + 1 * uz_sz + offset];;
            double * grad_2_gt1 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 1 * 3 *uz_sz + 2 * uz_sz + offset];;

            double * grad_0_gt2 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 2 * 3 *uz_sz + 0 * uz_sz + offset];
            double * grad_1_gt2 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 2 * 3 *uz_sz + 1 * uz_sz + offset];
            double * grad_2_gt2 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 2 * 3 *uz_sz + 2 * uz_sz + offset];
            
            double * grad_0_gt3 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 3 * 3 *uz_sz + 0 * uz_sz + offset];
            double * grad_1_gt3 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 3 * 3 *uz_sz + 1 * uz_sz + offset];
            double * grad_2_gt3 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 3 * 3 *uz_sz + 2 * uz_sz + offset];
            
            double * grad_0_gt4 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 4 * 3 *uz_sz + 0 * uz_sz + offset];
            double * grad_1_gt4 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 4 * 3 *uz_sz + 1 * uz_sz + offset];
            double * grad_2_gt4 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 4 * 3 *uz_sz + 2 * uz_sz + offset];
            
            double * grad_0_gt5 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 5 * 3 *uz_sz + 0 * uz_sz + offset];
            double * grad_1_gt5 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 5 * 3 *uz_sz + 1 * uz_sz + offset];
            double * grad_2_gt5 = &m_aeh_vars->grad_gt.get_vec_ptr()[ 5 * 3 *uz_sz + 2 * uz_sz + offset];
            

            const double * const gt0 = &gt_uz[0 * uz_sz + offset];
            const double * const gt1 = &gt_uz[1 * uz_sz + offset];
            const double * const gt2 = &gt_uz[2 * uz_sz + offset];
            const double * const gt3 = &gt_uz[3 * uz_sz + offset];
            const double * const gt4 = &gt_uz[4 * uz_sz + offset];
            const double * const gt5 = &gt_uz[5 * uz_sz + offset];

            const double * const chi = &chi_uz[0 * uz_sz + offset];
            
            
            deriv_x(grad_0_gt0, gt0, hx, sz, bflag);
            deriv_y(grad_1_gt0, gt0, hy, sz, bflag);
            deriv_z(grad_2_gt0, gt0, hz, sz, bflag);
            
            deriv_x(grad_0_gt1, gt1, hx, sz, bflag);
            deriv_y(grad_1_gt1, gt1, hy, sz, bflag);
            deriv_z(grad_2_gt1, gt1, hz, sz, bflag);
            
            deriv_x(grad_0_gt2, gt2, hx, sz, bflag);
            deriv_y(grad_1_gt2, gt2, hy, sz, bflag);
            deriv_z(grad_2_gt2, gt2, hz, sz, bflag);
            
            deriv_x(grad_0_gt3, gt3, hx, sz, bflag);
            deriv_y(grad_1_gt3, gt3, hy, sz, bflag);
            deriv_z(grad_2_gt3, gt3, hz, sz, bflag);
            
            deriv_x(grad_0_gt4, gt4, hx, sz, bflag);
            deriv_y(grad_1_gt4, gt4, hy, sz, bflag);
            deriv_z(grad_2_gt4, gt4, hz, sz, bflag);
            
            deriv_x(grad_0_gt5, gt5, hx, sz, bflag);
            deriv_y(grad_1_gt5, gt5, hy, sz, bflag);
            deriv_z(grad_2_gt5, gt5, hz, sz, bflag);

            deriv_x(grad_0_chi, chi, hx, sz, bflag);
            deriv_y(grad_1_chi, chi, hy, sz, bflag);
            deriv_z(grad_2_chi, chi, hz, sz, bflag);

        }
        
        
        if(!rank)
        {
            std::cout<<"====================================================================================================="<<std::endl;
            std::cout<<"                      AEH Solver Initialized                                                         "<<std::endl;
            std::cout<<" lm : ";
            for (unsigned int lm_idx=0; lm_idx<m_sph_modes.size(); lm_idx++)
                std::cout<<"( "<<m_sph_modes[lm_idx].first<<", "<<m_sph_modes[lm_idx].second<<" )  ";
            std::cout<<std::endl;
            std::cout<<"quad theta : "<<m_num_theta<<std::endl;
            std::cout<<"quad phi   : "<<m_num_phi  <<std::endl;
            std::cout<<"====================================================================================================="<<std::endl;
        }


    }
    template<typename Ctx, typename T>
    SpectralAEHSolver<Ctx, T>::~SpectralAEHSolver()
    {
        m_sph_modes.clear();
        gsl_integration_fixed_free(m_quad_theta);
        gsl_integration_fixed_free(m_quad_phi);

        m_aeh_vars->gt.destroy_vector();
        m_aeh_vars->At.destroy_vector();
        m_aeh_vars->chi.destroy_vector();
        m_aeh_vars->K.destroy_vector();
        
        m_aeh_vars->grad_gt.destroy_vector();
        m_aeh_vars->grad_chi.destroy_vector();

        delete m_aeh_vars;

        m_sin_qtheta.clear();
        m_cos_qtheta.clear();

        m_sin_qphi.clear();
        m_cos_qphi.clear();

        return;
    }

    template<typename Ctx, typename T>
    int SpectralAEHSolver<Ctx, T>::eval_aeh_level_set(const Point& origin, ot::Mesh* m_uiMesh, T* const aeh_f_ptr, const T* const h_qs, DendroScalar r_max)
    {
        if(!(m_uiMesh->isActive()))
            return 0;

        const ot::TreeNode* pNodes = &(*(m_uiMesh->getAllElements().begin()));
        const unsigned int eleOrder = m_uiMesh->getElementOrder();
        const unsigned int* e2n_cg = &(*(m_uiMesh->getE2NMapping().begin()));
        const unsigned int* e2n_dg = &(*(m_uiMesh->getE2NMapping_DG().begin()));
        const unsigned int nPe = m_uiMesh->getNumNodesPerElement();
        const unsigned int nodeLocalBegin = m_uiMesh->getNodeLocalBegin();
        const unsigned int nodeLocalEnd = m_uiMesh->getNodeLocalEnd();

        for (unsigned int elem = m_uiMesh->getElementLocalBegin(); elem < m_uiMesh->getElementLocalEnd(); elem++) {
            for (unsigned int k = 0; k < (eleOrder + 1); k++)
                for (unsigned int j = 0; j < (eleOrder + 1); j++)
                    for (unsigned int i = 0; i < (eleOrder + 1); i++) {

                        const unsigned int idx = elem * nPe +  k * (eleOrder + 1) * (eleOrder + 1) +  j * (eleOrder + 1) + i;
                        const unsigned int nodeLookUp_CG =  e2n_cg[idx];
                        
                        if (nodeLookUp_CG >= nodeLocalBegin &&
                            nodeLookUp_CG < nodeLocalEnd) {

                            const unsigned int nodeLookUp_DG = e2n_dg[idx];
                            unsigned int ownerID, ii_x, jj_y, kk_z;

                            m_uiMesh->dg2eijk(nodeLookUp_DG, ownerID, ii_x, jj_y, kk_z);
                            const DendroScalar len = (double)(1u << (m_uiMaxDepth - pNodes[ownerID].getLevel()));

                            const DendroScalar x = pNodes[ownerID].getX() + ii_x * (len / (eleOrder));
                            const DendroScalar y = pNodes[ownerID].getY() + jj_y * (len / (eleOrder));
                            const DendroScalar z = pNodes[ownerID].getZ() + kk_z * (len / (eleOrder));

                            const DendroScalar xx = GRIDX_TO_X(x) - origin.x();
                            const DendroScalar yy = GRIDY_TO_Y(y) - origin.y();
                            const DendroScalar zz = GRIDZ_TO_Z(z) - origin.z();

                            const DendroScalar p_rr = sqrt(xx * xx + yy * yy  + zz * zz);
                            
                            DendroScalar h_tp = (T)0;
                            if(p_rr < 1e-10)
                            {
                                
                                for(unsigned int lm_idx =0; lm_idx < m_sph_modes.size(); lm_idx++)
                                    h_tp+= h_qs[lm_idx] * real_spherical_harmonic(m_sph_modes[lm_idx].first, m_sph_modes[lm_idx].second, 0, 0);

                            }else if(p_rr < r_max)
                            {
                                const DendroScalar p_pt = acos(zz/p_rr);
                                const DendroScalar p_ph = std::fmod(atan2(yy,xx), 2 * M_PI);
                                for(unsigned int lm_idx =0; lm_idx < m_sph_modes.size(); lm_idx++)
                                    h_tp+= h_qs[lm_idx] * real_spherical_harmonic(m_sph_modes[lm_idx].first, m_sph_modes[lm_idx].second, p_pt, p_ph);
                                
                            }

                            aeh_f_ptr[nodeLookUp_CG] = p_rr - h_tp;
                            
                        }
                    }
        }
        
        return 0;

    }

    
    template<typename Ctx, typename T>
    AEHErrorType SpectralAEHSolver<Ctx, T>::solve(const Point& origin, Ctx * ctx, const T* const h_init, T* h_qs, unsigned int max_iter, double rel_eps, double abs_eps, double r_max)
    {
        ot::Mesh * m_uiMesh = ctx->get_mesh();
        if(!m_uiMesh->isActive())
            return AEHErrorType::SUCCESS;
        
        const int rank = m_uiMesh->getMPIRank();
        const unsigned int num_lm = m_sph_modes.size();
        const unsigned int cg_sz  = m_uiMesh->getDegOfFreedom();

        DendroScalar* h_qs0 = new DendroScalar[num_lm];
        DendroScalar* h_qs1 = new DendroScalar[num_lm];
        
        std::memcpy(h_qs0, h_init, sizeof(T) * num_lm);
        std::memcpy(h_qs1, h_init, sizeof(T) * num_lm);
        
        DVec aeh_v;
        aeh_v.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST, 2, true);

        DVec aeh_f;
        DVec aeh_h;


        DendroScalar * aeh_v_ptr_0 = aeh_v.get_vec_ptr();
        DendroScalar * aeh_v_ptr_1 = aeh_v.get_vec_ptr() + cg_sz;
        
        // aeh_f.set_vec_ptr(aeh_v_ptr_0 , m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST, 1, true);
        // aeh_h.set_vec_ptr(aeh_v_ptr_1 , m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST, 1, true);

        aeh_f.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST, 1, true);
        aeh_h.create_vector(m_uiMesh, ot::DVEC_TYPE::OCT_SHARED_NODES, ot::DVEC_LOC::HOST, 1, true);

        T * aeh_h_ptr = aeh_h.get_vec_ptr();
        T * aeh_f_ptr = aeh_f.get_vec_ptr();

        unsigned int num_angular_pts = m_num_theta * m_num_phi;

        std::vector<T> interp_coords;
        interp_coords.resize(3 * num_angular_pts);

        std::vector<T> aeh_r;
        aeh_r.resize(num_angular_pts);

        Point grid_limits[2];
        Point domain_limits[2];
        grid_limits[0]   = Point(bssn::BSSN_OCTREE_MIN[0], bssn::BSSN_OCTREE_MIN[1], bssn::BSSN_OCTREE_MIN[2]);
        grid_limits[1]   = Point(bssn::BSSN_OCTREE_MAX[0], bssn::BSSN_OCTREE_MAX[1], bssn::BSSN_OCTREE_MAX[2]);

        domain_limits[0] = Point(bssn::BSSN_COMPD_MIN[0], bssn::BSSN_COMPD_MIN[1], bssn::BSSN_COMPD_MIN[2]);
        domain_limits[1] = Point(bssn::BSSN_COMPD_MAX[0], bssn::BSSN_COMPD_MAX[1], bssn::BSSN_COMPD_MAX[2]);

        unsigned int iter=0;
        const double * const m_qtheta  = m_quad_theta->x;
        const double * const m_qphi    = m_quad_phi->x;
        
        std::vector<T> aeh_h_inp;
        aeh_h_inp.resize(num_angular_pts);
        double relative_error=0;
        double absolute_error=0;
        do
        {
            
            //printArray_1D(interp_coords.data(), interp_coords.size());
            this->eval_aeh_level_set(origin, m_uiMesh, aeh_f_ptr, h_qs0, r_max);
            ctx->aeh_expansion(origin, m_aeh_vars, aeh_f, aeh_h);
            
            m_uiMesh->readFromGhostBegin(aeh_h.get_vec_ptr(), 1);

            for(unsigned int qt=0; qt < m_num_theta; qt++)
                for(unsigned int qp=0; qp < m_num_phi; qp++)
                {
                    const unsigned int q_idx = qt * m_num_phi + qp;
                    T r_val = (T)0;
                    for(unsigned int lm_idx=0; lm_idx < m_sph_modes.size(); lm_idx++)
                        r_val+= real_spherical_harmonic(m_sph_modes[lm_idx].first, m_sph_modes[lm_idx].second, m_qtheta[qt], m_qphi[qp]) * h_qs0[lm_idx];

                    aeh_r[q_idx] = r_val;
                    
                    interp_coords[3 * q_idx + 0] = origin.x() + (r_val) * sin(m_qtheta[qt]) * cos(m_qphi[qp]); //m_sin_qtheta[qt] * m_cos_qphi[qp];
                    interp_coords[3 * q_idx + 1] = origin.y() + (r_val) * sin(m_qtheta[qt]) * sin(m_qphi[qp]); //m_sin_qtheta[qt] * m_sin_qphi[qp];
                    interp_coords[3 * q_idx + 2] = origin.z() + (r_val) * cos(m_qtheta[qt]);                   //m_cos_qtheta[qt];
                    
                }
            
            m_uiMesh->readFromGhostEnd(aeh_h.get_vec_ptr(), 1);

            std::vector<unsigned int> valid_idx;
            valid_idx.clear();

            ot::da::interpolateToCoords(m_uiMesh, aeh_h_ptr, interp_coords.data(), interp_coords.size(), grid_limits, domain_limits, aeh_h_inp.data(), valid_idx);
            
            for(unsigned int lm_idx=1; lm_idx < m_sph_modes.size(); lm_idx++)
            {
                DendroScalar tmp  = 0.0;
                DendroScalar a_qs = 0.0;

                int l = m_sph_modes[lm_idx].first;
                int m = m_sph_modes[lm_idx].second;
                
                for(unsigned int idx=0;idx < valid_idx.size(); idx++)
                {
                    const unsigned int qp = valid_idx[idx] % m_num_phi;
                    const unsigned int qt = (valid_idx[idx] -qp) / m_num_phi;
                    tmp+= aeh_h_inp[valid_idx[idx]] *  real_spherical_harmonic(l, m, m_qtheta[qt],  m_qphi[qp]) * m_quad_theta->weights[qt] * m_quad_phi->weights[qp];
                    //printf("idx : %d value = %.12E\n", valid_idx[idx], aeh_h_inp[valid_idx[idx]]);
                }

                par::Mpi_Allreduce(&tmp, &a_qs, 1, MPI_SUM, m_uiMesh->getMPICommunicator());
                // if(a_qs < rel_eps)
                //     a_qs=0.0;

                h_qs1[lm_idx] = h_qs0[lm_idx] - (1.0 / (DendroScalar)( l * (l + 1))) * a_qs;

            }
            
            h_qs1[0] = h_qs0[0];
            //h_qs1[0] = this->solve_00_bisection(origin, 0, r_max, ctx, h_qs1, aeh_f, aeh_h, interp_coords, abs_eps, r_max);
            h_qs1[0] = this->solve_00_newton(origin, ctx, h_qs1, aeh_f, aeh_h, interp_coords, 1e-12, r_max);
            if(h_qs1[0]==-1.0)
                return AEHErrorType::FAIL;

            // printArray_1D(h_qs0,num_lm);
            // printArray_1D(h_qs1,num_lm);
            absolute_error = abs(integral_S2_h(origin, ctx, h_qs1, aeh_f, aeh_h, interp_coords, r_max));
            relative_error = normL2(h_qs1, h_qs0, m_sph_modes.size())/ normL2(h_qs0, m_sph_modes.size());
            //absolute_error = this->rhs_00(origin, ctx, h_qs1[0], h_qs1, aeh_f, aeh_h, interp_coords);
            if(!rank)
            {
                printf("AEH solver iteration = %d / %d relative error = %.8E  absolute error = %.8E \n",iter + 1, max_iter, relative_error, absolute_error);
                for(unsigned int lm_idx=0; lm_idx < m_sph_modes.size(); lm_idx++)
                    printf("h_{%d, %d} = %.8E ,",m_sph_modes[lm_idx].first, m_sph_modes[lm_idx].second, h_qs1[lm_idx]);
                printf("\n");
            }
                
            
            std::swap(h_qs0, h_qs1);
            iter+=1;

            {
                DendroScalar* pData[16];

                this->eval_aeh_level_set(origin, m_uiMesh, aeh_f_ptr, h_qs0, r_max);
                ctx->aeh_expansion(origin, m_aeh_vars, aeh_f, aeh_h);
            
                m_uiMesh->readFromGhostBegin(aeh_h.get_vec_ptr(), 1);
                m_uiMesh->readFromGhostEnd(aeh_h.get_vec_ptr(), 1);

                pData[0] = aeh_f.get_vec_ptr();
                pData[1] = aeh_h.get_vec_ptr();

                DendroScalar* gr_evar  = ctx->get_evolution_vars().get_vec_ptr();
                pData[2] = &gr_evar[VAR::U_SYMGT0 * cg_sz];
                pData[3] = &gr_evar[VAR::U_SYMGT1 * cg_sz];
                pData[4] = &gr_evar[VAR::U_SYMGT2 * cg_sz];
                pData[5] = &gr_evar[VAR::U_SYMGT3 * cg_sz];
                pData[6] = &gr_evar[VAR::U_SYMGT4 * cg_sz];
                pData[7] = &gr_evar[VAR::U_SYMGT5 * cg_sz];

                pData[8]  = &gr_evar[VAR::U_SYMAT0 * cg_sz];
                pData[9]  = &gr_evar[VAR::U_SYMAT1 * cg_sz];
                pData[10] = &gr_evar[VAR::U_SYMAT2 * cg_sz];
                pData[11] = &gr_evar[VAR::U_SYMAT3 * cg_sz];
                pData[12] = &gr_evar[VAR::U_SYMAT4 * cg_sz];
                pData[13] = &gr_evar[VAR::U_SYMAT5 * cg_sz];

                pData[14] = &gr_evar[VAR::U_CHI * cg_sz];
                pData[15] = &gr_evar[VAR::U_K * cg_sz];


                const char * pNames[]={"F", "H", "gt0", "gt1", "gt2", "gt3", "gt4", "gt5", "At0", "At1", "At2", "At3", "At4", "At5", "At5", "chi", "K" };
                char fname[256];
                sprintf(fname, "%s_iter_%03d","vtu/aeh", iter);
                io::vtk::mesh2vtuFine(m_uiMesh, fname, 0, nullptr, nullptr, 16, pNames, (const double**)pData, 0, nullptr, nullptr, false);

            }


        }while((iter < max_iter) && (relative_error > rel_eps && absolute_error > abs_eps) );
        
        std::memcpy(h_qs, h_qs0, sizeof(T) * num_lm);
        
        delete [] h_qs1;
        delete [] h_qs0;

        aeh_f.destroy_vector();
        aeh_h.destroy_vector();
        aeh_v.destroy_vector();
        if(iter==max_iter && (relative_error > rel_eps))
            return AEHErrorType::FAIL;
        else
            return AEHErrorType::SUCCESS;

    }

}

