/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "force_eam.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "memory.h"
#include <immintrin.h>

#define MAXLINE 1024

#define MAX(a,b) (a>b?a:b)
#define MIN(a,b) (a<b?a:b)

/* ---------------------------------------------------------------------- */

ForceEAM::ForceEAM()
{
  cutforce = 0.0;
  cutforcesq = 0.0;
  use_oldcompute = 0;

  nmax = 0;

  rho = 0;
  fp = 0;
  style = FORCEEAM;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

ForceEAM::~ForceEAM()
{

}

void ForceEAM::setup()
{
  me = threads->mpi_me;
  coeff("Cu_u6.eam");
  init_style();
}


void ForceEAM::compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  if(neighbor.halfneigh) {
    if(threads->omp_num_threads > 1)
      return ;
    else
      return compute_halfneigh(atom, neighbor, comm, me);
  } else return compute_fullneigh(atom, neighbor, comm, me);

}
/* ---------------------------------------------------------------------- */

void ForceEAM::compute_halfneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{

  MMD_float evdwl = 0.0;

  virial = 0;
  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  if(atom.nmax > nmax) {
    nmax = atom.nmax;
    delete [] rho;
    delete [] fp;

    rho = new MMD_float[nmax];
    fp = new MMD_float[nmax];
  }

  MMD_float* x = &atom.x[0][0];
  MMD_float* f = &atom.f[0][0];
  const int nlocal = atom.nlocal;

  // zero out density

  for(int i = 0; i < atom.nlocal + atom.nghost; i++) {
    f[i * PAD + 0] = 0;
    f[i * PAD + 1] = 0;
    f[i * PAD + 2] = 0;
  }

  for(MMD_int i = 0; i < nlocal; i++) rho[i] = 0.0;

  // rho = density at each atom
  // loop over neighbors of my atoms

  for(MMD_int i = 0; i < nlocal; i++) {
    int* neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneigh = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    MMD_float rhoi = 0.0;
    int vf = 4;
    int upper_bound = numneigh / vf * vf;
    MMD_int jj = 0;
    for(; jj < upper_bound; jj += vf){
        const MMD_int j = neighs[jj];
        const MMD_float delx = xtmp - x[j * PAD + 0];
        const MMD_float dely = ytmp - x[j * PAD + 1];
        const MMD_float delz = ztmp - x[j * PAD + 2];
        const MMD_float rsq = delx * delx + dely * dely + delz * delz;

        const MMD_int j2 = neighs[jj+1];
        const MMD_float delx2 = xtmp - x[j2 * PAD + 0];
        const MMD_float dely2 = ytmp - x[j2 * PAD + 1];
        const MMD_float delz2 = ztmp - x[j2 * PAD + 2];
        const MMD_float rsq2 = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;

        const MMD_int j3 = neighs[jj+2];
        const MMD_float delx3 = xtmp - x[j3 * PAD + 0];
        const MMD_float dely3 = ytmp - x[j3 * PAD + 1];
        const MMD_float delz3 = ztmp - x[j3 * PAD + 2];
        const MMD_float rsq3 = delx3 * delx3 + dely3 * dely3 + delz3 * delz3;

        const MMD_int j4 = neighs[jj+3];
        const MMD_float delx4 = xtmp - x[j4 * PAD + 0];
        const MMD_float dely4 = ytmp - x[j4 * PAD + 1];
        const MMD_float delz4 = ztmp - x[j4 * PAD + 2];
        const MMD_float rsq4 = delx4 * delx4 + dely4 * dely4 + delz4 * delz4;
        //double rsqs_[4] = {rsq, rsq2, rsq3, rsq4};
        //__m256d cmp = _mm256_cmp_pd(_mm256_load_pd(rsqs_), _mm256_set1_pd(cutforcesq), _CMP_LT_OQ);
        //int mask = _mm256_movemask_pd(cmp);
        //if (mask == 15){
	if(rsq < cutforcesq && rsq2 < cutforcesq && rsq3 < cutforcesq && rsq4 < cutforcesq){
            MMD_float p = sqrt(rsq) * rdr + 1.0;
            MMD_float p2 = sqrt(rsq2) * rdr + 1.0;
            MMD_float p3 = sqrt(rsq3) * rdr + 1.0;
            MMD_float p4 = sqrt(rsq4) * rdr + 1.0;
            __m256d p_ = {p, p2, p3, p4};
            MMD_int m = static_cast<int>(p);
            MMD_int m2 = static_cast<int>(p2);
            MMD_int m3 = static_cast<int>(p3);
            MMD_int m4 = static_cast<int>(p4);
            __m256d m_ = {m, m2, m3, m4};
            __m256d min_ = _mm256_min_pd(m_, _mm256_set1_pd(nr - 1));
            p_ = p_ - min_;
            p_ = _mm256_min_pd(p_, _mm256_set1_pd(1.0));
            __m256d rho_spline_m = _mm256_load_pd(&rhor_spline[m*7+3]);
            double p_copy[4];
            _mm256_store_pd(p_copy, p_);
            __m256d p1_ = {p_copy[0]*p_copy[0]*p_copy[0], p_copy[0]*p_copy[0], p_copy[0], 1};
            __m256d rho_spline_m2 = _mm256_load_pd(&rhor_spline[m2*7+3]);
            __m256d p2_ = {p_copy[1]*p_copy[1]*p_copy[1], p_copy[1]*p_copy[1], p_copy[1], 1};
            __m256d rho_spline_m3 = _mm256_load_pd(&rhor_spline[m3*7+3]);
            __m256d p3_ = {p_copy[2]*p_copy[2]*p_copy[2], p_copy[2]*p_copy[2], p_copy[2], 1};
            __m256d rho_spline_m4 = _mm256_load_pd(&rhor_spline[m4*7+3]);
            __m256d p4_ = {p_copy[3]*p_copy[3]*p_copy[3], p_copy[3]*p_copy[3], p_copy[3], 1};
            double rho1_[4];
            double rho1 = 0;
            _mm256_store_pd(rho1_, rho_spline_m * p1_);
            rho1 += rho1_[0];
            rho1 += rho1_[1];
            rho1 += rho1_[2];
            rho1 += rho1_[3];
            double rho2_[4];
            double rho2 = 0;
            _mm256_store_pd(rho2_, rho_spline_m2 * p2_);
            rho2 += rho2_[0];
            rho2 += rho2_[1];
            rho2 += rho2_[2];
            rho2 += rho2_[3];
            double rho3_[4];
            double rho3 = 0;
            _mm256_store_pd(rho3_, rho_spline_m3 * p3_);
            rho3 += rho3_[0];
            rho3 += rho3_[1];
            rho3 += rho3_[2];
            rho3 += rho3_[3];
            double rho4_[4];
            double rho4 = 0;
            _mm256_store_pd(rho4_, rho_spline_m4 * p4_);
            rho4 += rho4_[0];
            rho4 += rho4_[1];
            rho4 += rho4_[2];
            rho4 += rho4_[3];
            rhoi += (rho1 + rho2 + rho3 + rho4);
            //__m256d j_s = {j, j2, j3, j4};
            //__m256d cmp2 = _mm256_cmp_pd(j_s, _mm256_set1_pd(nlocal), _CMP_LT_OQ);
            //int mask2 = _mm256_movemask_pd(cmp2);
            //if(mask2 == 15){
	    if(j < nlocal && j2 < nlocal && j3 < nlocal && j4 < nlocal){
                rho[j] += rho1;
                rho[j2] += rho2;
                rho[j3] += rho3;
                rho[j4] += rho4;
            }
            //else if(mask2 == 0){
	    else if(!(j < nlocal) && !(j2 < nlocal) && !(j3 < nlocal) && !(j4 < nlocal)){
            }
            else{
                if(j < nlocal) {
                    rho[j] += rho1;
                }
                if(j2 < nlocal) {
                    rho[j2] += rho2;
                }
                if(j3 < nlocal) {
                    rho[j3] += rho3;
                }
                if(j4 < nlocal) {
                    rho[j4] += rho4;
                }
            }
        }
        //else if (mask == 0){
	else if(!(rsq < cutforcesq) && !(rsq2 < cutforcesq) && !(rsq3 < cutforcesq) && !(rsq4 < cutforcesq)){
        }
        else{
            if(rsq < cutforcesq) {
                MMD_float p = sqrt(rsq) * rdr + 1.0;
                MMD_int m = static_cast<int>(p);
                m = m < nr - 1 ? m : nr - 1;
                p -= m;
                p = p < 1.0 ? p : 1.0;
                rhoi += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
                if(j < nlocal) {
                    rho[j] += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
                }
            }
            if(rsq2 < cutforcesq) {
                MMD_float p = sqrt(rsq2) * rdr + 1.0;
                MMD_int m = static_cast<int>(p);
                m = m < nr - 1 ? m : nr - 1;
                p -= m;
                p = p < 1.0 ? p : 1.0;
                rhoi += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
                if(j2 < nlocal) {
                    rho[j2] += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
                }
            }
            if(rsq3 < cutforcesq) {
                MMD_float p = sqrt(rsq3) * rdr + 1.0;
                MMD_int m = static_cast<int>(p);
                m = m < nr - 1 ? m : nr - 1;
                p -= m;
                p = p < 1.0 ? p : 1.0;
                rhoi += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
                if(j3 < nlocal) {
                    rho[j3] += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
                }
            }
            if(rsq4 < cutforcesq) {
                MMD_float p = sqrt(rsq4) * rdr + 1.0;
                MMD_int m = static_cast<int>(p);
                m = m < nr - 1 ? m : nr - 1;
                p -= m;
                p = p < 1.0 ? p : 1.0;
                rhoi += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
                if(j4 < nlocal) {
                    rho[j4] += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
                }
            }
        }
    }
    for(;jj<numneigh;jj++){
        const MMD_int j = neighs[jj];
        const MMD_float delx = xtmp - x[j * PAD + 0];
        const MMD_float dely = ytmp - x[j * PAD + 1];
        const MMD_float delz = ztmp - x[j * PAD + 2];
        const MMD_float rsq = delx * delx + dely * dely + delz * delz;
        if(rsq < cutforcesq) {
            MMD_float p = sqrt(rsq) * rdr + 1.0;
            MMD_int m = static_cast<int>(p);
            m = m < nr - 1 ? m : nr - 1;
            p -= m;
            p = p < 1.0 ? p : 1.0;
            rhoi += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
            if(j < nlocal) {
                rho[j] += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
            }
        }
    }


    rho[i] += rhoi;
  }

  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom

  for(MMD_int i = 0; i < nlocal; i++) {
    MMD_float p = 1.0 * rho[i] * rdrho + 1.0;
    MMD_int m = static_cast<int>(p);
    m = MAX(1, MIN(m, nrho - 1));
    p -= m;
    p = MIN(p, 1.0);
    fp[i] = (frho_spline[m * 7 + 0] * p + frho_spline[m * 7 + 1]) * p + frho_spline[m * 7 + 2];

    // printf("fp: %lf %lf %lf %lf %lf %i %lf %lf\n",fp[i],p,frho_spline[m*7+0],frho_spline[m*7+1],frho_spline[m*7+2],m,rdrho,rho[i]);
    if(evflag) {
      evdwl += ((frho_spline[m * 7 + 3] * p + frho_spline[m * 7 + 4]) * p + frho_spline[m * 7 + 5]) * p + frho_spline[m * 7 + 6];
    }
  }

  // communicate derivative of embedding function

  communicate(atom, comm);


  // compute forces on each atom
  // loop over neighbors of my atoms
  for(MMD_int i = 0; i < nlocal; i++) {
    int* neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneigh = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    MMD_float fx = 0;
    MMD_float fy = 0;
    MMD_float fz = 0;

    //printf("Hallo %i %i %lf %lf\n",i,numneigh[i],sqrt(cutforcesq),neighbor.cutneigh);
    int vf = 4;
    MMD_int jj = 0;
    int upper_bound = numneigh / vf * vf;
    for(; jj < upper_bound; jj+=vf) {
      const MMD_int j = neighs[jj];
      const MMD_int j2 = neighs[jj+1];
      const MMD_int j3 = neighs[jj+2];
      const MMD_int j4 = neighs[jj+3];
      __m256d j_ = {j, j2, j3, j4};

      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      const MMD_float delx2 = xtmp - x[j2 * PAD + 0];
      const MMD_float dely2 = ytmp - x[j2 * PAD + 1];
      const MMD_float delz2 = ztmp - x[j2 * PAD + 2];
      const MMD_float rsq2 = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;

      const MMD_float delx3 = xtmp - x[j3 * PAD + 0];
      const MMD_float dely3 = ytmp - x[j3 * PAD + 1];
      const MMD_float delz3 = ztmp - x[j3 * PAD + 2];
      const MMD_float rsq3 = delx3 * delx3 + dely3 * dely3 + delz3 * delz3;

      const MMD_float delx4 = xtmp - x[j4 * PAD + 0];
      const MMD_float dely4 = ytmp - x[j4 * PAD + 1];
      const MMD_float delz4 = ztmp - x[j4 * PAD + 2];
      const MMD_float rsq4 = delx4 * delx4 + dely4 * dely4 + delz4 * delz4;

      __m256d delx_ = {delx, delx2, delx3, delx4};
      __m256d dely_ = {dely, dely2, dely3, dely4};
      __m256d delz_ = {delz, delz2, delz3 , delz4};
      __m256d rsq_ = {rsq, rsq2, rsq3, rsq4};
      //__m256d cmp = _mm256_cmp_pd(rsq_, _mm256_set1_pd(cutforcesq), _CMP_LT_OQ);
      //int mask = _mm256_movemask_pd(cmp);
      //printf("EAM: %i %i %lf %lf\n",i,j,rsq,cutforcesq);
      if(rsq < cutforcesq && rsq2 < cutforcesq && rsq3 < cutforcesq && rsq4 < cutforcesq){
      //if(mask==15) {
        __m256d r_ = _mm256_sqrt_pd(rsq_);
        __m256d p_ = _mm256_add_pd(_mm256_mul_pd(r_, _mm256_set1_pd(rdr)), _mm256_set1_pd(1.0));
        //MMD_float r = sqrt(rsq);
        //MMD_float r2 = sqrt(rsq2);
        //MMD_float r3 = sqrt(rsq3);
        //MMD_float r4 = sqrt(rsq4);
        //double temp_p[4];
        //_mm256_store_pd(temp_p, p_);

        MMD_float p = p_[0];
        MMD_float p2 = p_[1];
        MMD_float p3 = p_[2];
        MMD_float p4 = p_[3];
//        MMD_float p = r * rdr + 1.0;
//        MMD_float p2 = r2 *rdr + 1.0;
//        MMD_float p3 = r3 * rdr + 1.0;
//        MMD_float p4 = r4 * rdr + 1.0;

        MMD_int m = static_cast<int>(p);
        MMD_int m2 = static_cast<int>(p2);
        MMD_int m3 = static_cast<int>(p3);
        MMD_int m4 = static_cast<int>(p4);

        __m256d m_ = {m , m2, m3, m4};
        m_ = _mm256_min_pd(m_, _mm256_set1_pd(nr - 1));
        p_ = p_ - m_;
//        m = m < nr - 1 ? m : nr - 1;
//        m2 = m2 < nr - 1? m2 : nr - 1;
//        m3 = m3 < nr - 1? m3 : nr - 1;
//        m4 = m4 < nr - 1? m4 : nr - 1;

//        p -= m;
//        p2 -= m2;
//        p3 -= m3;
//        p4 -= m4;
        p_ = _mm256_min_pd(p_, _mm256_set1_pd(1.0));


        p = p_[0];
        p2 = p_[1];
        p3 = p_[2];
        p4 = p_[3];


//        p = p < 1.0 ? p : 1.0;
//        p2 = p2 < 1.0 ? p2 : 1.0;
//        p3 = p3 < 1.0 ? p3: 1.0;
//        p4 = p4 < 1.0 ? p4: 1.0;
        // rhoip = derivative of (density at atom j due to atom i)
        // rhojp = derivative of (density at atom i due to atom j)
        // phi = pair potential energy
        // phip = phi'
        // z2 = phi * r
        // z2p = (phi * r)' = (phi' r) + phi
        // psip needs both fp[i] and fp[j] terms since r_ij appears in two
        //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
        //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

        MMD_float rhoip = (rhor_spline[m * 7 + 0] * p + rhor_spline[m * 7 + 1]) * p + rhor_spline[m * 7 + 2];
        MMD_float rhoip2 = (rhor_spline[m2 * 7 + 0] * p2 + rhor_spline[m2 * 7 + 1]) * p2 + rhor_spline[m2 * 7 + 2];
        MMD_float rhoip3 = (rhor_spline[m3 * 7 + 0] * p3 + rhor_spline[m3 * 7 + 1]) * p3 + rhor_spline[m3 * 7 + 2];
        MMD_float rhoip4 = (rhor_spline[m4 * 7 + 0] * p4 + rhor_spline[m4 * 7 + 1]) * p4 + rhor_spline[m4 * 7 + 2];


        MMD_float z2p = (z2r_spline[m * 7 + 0] * p + z2r_spline[m * 7 + 1]) * p + z2r_spline[m * 7 + 2];
        MMD_float z2p2 = (z2r_spline[m2 * 7 + 0] * p2 + z2r_spline[m2 * 7 + 1]) * p2 + z2r_spline[m2 * 7 + 2];
        MMD_float z2p3 = (z2r_spline[m3 * 7 + 0] * p3 + z2r_spline[m3 * 7 + 1]) * p3 + z2r_spline[m3 * 7 + 2];
        MMD_float z2p4 = (z2r_spline[m4 * 7 + 0] * p4 + z2r_spline[m4 * 7 + 1]) * p4 + z2r_spline[m4 * 7 + 2];

        __m256d z2r_ = _mm256_load_pd(&z2r_spline[m*7+3]);
        __m256d z2r_p = {p * p * p, p * p, p , 1};
        z2r_ = z2r_ * z2r_p;
        MMD_float z2 = (z2r_[0] + z2r_[1] + z2r_[2] + z2r_[3]);

        //MMD_float z2 = ((z2r_spline[m * 7 + 3] * p + z2r_spline[m * 7 + 4]) * p + z2r_spline[m * 7 + 5]) * p + z2r_spline[m * 7 + 6];
        __m256d z2r2_ = _mm256_load_pd(&z2r_spline[m2*7+3]);
        __m256d z2r_p2 = {p2 * p2 * p2, p2 * p2, p2 , 1};
        z2r2_ = z2r2_ * z2r_p2;
        MMD_float z22 = (z2r2_[0] + z2r2_[1] + z2r2_[2] + z2r2_[3]);

        __m256d z2r3_ = _mm256_load_pd(&z2r_spline[m3*7+3]);
        __m256d z2r_p3 = {p3 * p3 * p3, p3 * p3, p3 , 1};
        z2r3_ = z2r3_ * z2r_p3;
        MMD_float z23 = (z2r3_[0] + z2r3_[1] + z2r3_[2] + z2r3_[3]);

        __m256d z2r4_ = _mm256_load_pd(&z2r_spline[m4*7+3]);
        __m256d z2r_p4 = {p4 * p4 * p4, p4 * p4, p4 , 1};
        z2r4_ = z2r4_ * z2r_p4;
        MMD_float z24 = (z2r4_[0] + z2r4_[1] + z2r4_[2] + z2r4_[3]);
        //MMD_float z22 = ((z2r_spline[m2 * 7 + 3] * p2 + z2r_spline[m2 * 7 + 4]) * p2 + z2r_spline[m2 * 7 + 5]) * p2 + z2r_spline[m2 * 7 + 6];
        //MMD_float z23 = ((z2r_spline[m3 * 7 + 3] * p3 + z2r_spline[m3 * 7 + 4]) * p3 + z2r_spline[m3 * 7 + 5]) * p3 + z2r_spline[m3 * 7 + 6];
        //MMD_float z24 = ((z2r_spline[m4 * 7 + 3] * p4 + z2r_spline[m4 * 7 + 4]) * p4 + z2r_spline[m4 * 7 + 5]) * p4 + z2r_spline[m4 * 7 + 6];


        __m256d recip_ = _mm256_div_pd(_mm256_set1_pd(1.0), r_);
//        double temp_recip[4];
//        _mm256_store_pd(temp_recip, recip_);
        MMD_float recip = recip_[0];
        MMD_float recip2 = recip_[1];
        MMD_float recip3 = recip_[2];
        MMD_float recip4 = recip_[3];

        MMD_float phi = z2 * recip;
        MMD_float phi2 = z22 * recip2;
        MMD_float phi3 = z23 * recip3;
        MMD_float phi4 = z24 * recip4;
        __m256d phi_ = {phi, phi2, phi3, phi4};

        MMD_float phip = z2p * recip - phi * recip;
        MMD_float phip2 = z2p2 * recip2 - phi2 * recip2;
        MMD_float phip3 = z2p3 * recip3 - phi3 * recip3;
        MMD_float phip4 = z2p4 * recip4 - phi4 * recip4;
        __m256d phip_ = {phip, phip2, phip3, phip4};


        MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
        MMD_float psip2 = fp[i] * rhoip2 + fp[j2] * rhoip2 + phip2;
        MMD_float psip3 = fp[i] * rhoip3 + fp[j3] * rhoip3 + phip3;
        MMD_float psip4 = fp[i] * rhoip4 + fp[j4] * rhoip4 + phip4;
        __m256d psip_ = {psip, psip2, psip3, psip4};

        __m256d fpair_ = _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-1), psip_), recip_);
        MMD_float fpair = fpair_[0];
        MMD_float fpair2 = fpair_[1];
        MMD_float fpair3 = fpair_[2];
        MMD_float fpair4 = fpair_[3];

        __m256d fx_res = delx_ * fpair_;
        __m256d fy_res = dely_ * fpair_;
        __m256d fz_res = delz_ * fpair_;

        fx += (fx_res[0] + fx_res[1] + fx_res[2] + fx_res[3]);
        fy += (fy_res[0] + fy_res[1] + fy_res[2] + fy_res[3]);
        fz += (fz_res[0] + fz_res[1] + fz_res[2] + fz_res[3]);





        //  	if(i==0&&j<20)
        //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
        if(j < nlocal) {
          f[j * PAD + 0] -= delx * fpair;
          f[j * PAD + 1] -= dely * fpair;
          f[j * PAD + 2] -= delz * fpair;
        } else fpair *= 0.5;

        if(j2 < nlocal) {
            f[j2 * PAD + 0] -= delx2 * fpair2;
            f[j2 * PAD + 1] -= dely2 * fpair2;
            f[j2 * PAD + 2] -= delz2 * fpair2;
          } else fpair2 *= 0.5;

          if(j3 < nlocal) {
              f[j3 * PAD + 0] -= delx3 * fpair3;
              f[j3 * PAD + 1] -= dely3 * fpair3;
              f[j3 * PAD + 2] -= delz3 * fpair3;
          } else fpair3 *= 0.5;

          if(j4 < nlocal) {
              f[j4 * PAD + 0] -= delx4 * fpair4;
              f[j4 * PAD + 1] -= dely4 * fpair4;
              f[j4 * PAD + 2] -= delz4 * fpair4;
          } else fpair4 *= 0.5;

        if(evflag) {
          __m256d res = (delx_ * delx_ * fpair_) + (dely_ * dely_ * fpair_) + (delz_ * delz_ * fpair_);
          virial += (res[0] + res[1] + res[2] + res[3]);
        }
        if(j < nlocal) evdwl += phi;
        else evdwl += 0.5 * phi;

        if(j2 < nlocal) evdwl += phi2;
        else evdwl += 0.5 * phi2;

        if(j3 < nlocal) evdwl += phi3;
        else evdwl += 0.5 * phi3;

        if(j4 < nlocal) evdwl += phi4;
        else evdwl += 0.5 * phi4;


      }
      //else if(mask==0){
      else if(!(rsq < cutforcesq) && !(rsq2 < cutforcesq) && !(rsq3 < cutforcesq) && !(rsq4 < cutforcesq)){
      }
      else{
          if(rsq < cutforcesq) {
              MMD_float r = sqrt(rsq);
              MMD_float p = r * rdr + 1.0;
              MMD_int m = static_cast<int>(p);
              m = m < nr - 1 ? m : nr - 1;
              p -= m;
              p = p < 1.0 ? p : 1.0;


              // rhoip = derivative of (density at atom j due to atom i)
              // rhojp = derivative of (density at atom i due to atom j)
              // phi = pair potential energy
              // phip = phi'
              // z2 = phi * r
              // z2p = (phi * r)' = (phi' r) + phi
              // psip needs both fp[i] and fp[j] terms since r_ij appears in two
              //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
              //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

              MMD_float rhoip = (rhor_spline[m * 7 + 0] * p + rhor_spline[m * 7 + 1]) * p + rhor_spline[m * 7 + 2];
              MMD_float z2p = (z2r_spline[m * 7 + 0] * p + z2r_spline[m * 7 + 1]) * p + z2r_spline[m * 7 + 2];
              MMD_float z2 = ((z2r_spline[m * 7 + 3] * p + z2r_spline[m * 7 + 4]) * p + z2r_spline[m * 7 + 5]) * p + z2r_spline[m * 7 + 6];

              MMD_float recip = 1.0 / r;
              MMD_float phi = z2 * recip;
              MMD_float phip = z2p * recip - phi * recip;
              MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
              MMD_float fpair = -psip * recip;

              fx += delx * fpair;
              fy += dely * fpair;
              fz += delz * fpair;

              //  	if(i==0&&j<20)
              //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
              if(j < nlocal) {
                  f[j * PAD + 0] -= delx * fpair;
                  f[j * PAD + 1] -= dely * fpair;
                  f[j * PAD + 2] -= delz * fpair;
              } else fpair *= 0.5;

              if(evflag) {
                  virial += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
              }

              if(j < nlocal) evdwl += phi;
              else evdwl += 0.5 * phi;
          }
          if(rsq2 < cutforcesq) {
              MMD_float r = sqrt(rsq2);
              MMD_float p = r * rdr + 1.0;
              MMD_int m = static_cast<int>(p);
              m = m < nr - 1 ? m : nr - 1;
              p -= m;
              p = p < 1.0 ? p : 1.0;


              // rhoip = derivative of (density at atom j due to atom i)
              // rhojp = derivative of (density at atom i due to atom j)
              // phi = pair potential energy
              // phip = phi'
              // z2 = phi * r
              // z2p = (phi * r)' = (phi' r) + phi
              // psip needs both fp[i] and fp[j] terms since r_ij appears in two
              //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
              //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

              MMD_float rhoip = (rhor_spline[m * 7 + 0] * p + rhor_spline[m * 7 + 1]) * p + rhor_spline[m * 7 + 2];
              MMD_float z2p = (z2r_spline[m * 7 + 0] * p + z2r_spline[m * 7 + 1]) * p + z2r_spline[m * 7 + 2];
              MMD_float z2 = ((z2r_spline[m * 7 + 3] * p + z2r_spline[m * 7 + 4]) * p + z2r_spline[m * 7 + 5]) * p + z2r_spline[m * 7 + 6];

              MMD_float recip = 1.0 / r;
              MMD_float phi = z2 * recip;
              MMD_float phip = z2p * recip - phi * recip;
              MMD_float psip = fp[i] * rhoip + fp[j2] * rhoip + phip;
              MMD_float fpair = -psip * recip;

              fx += delx2 * fpair;
              fy += dely2 * fpair;
              fz += delz2 * fpair;

              //  	if(i==0&&j<20)
              //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
              if(j2 < nlocal) {
                  f[j2 * PAD + 0] -= delx2 * fpair;
                  f[j2 * PAD + 1] -= dely2 * fpair;
                  f[j2 * PAD + 2] -= delz2 * fpair;
              } else fpair *= 0.5;

              if(evflag) {
                  virial += delx2 * delx2 * fpair + dely2 * dely2 * fpair + delz2 * delz2 * fpair;
              }

              if(j2 < nlocal) evdwl += phi;
              else evdwl += 0.5 * phi;
          }
          if(rsq3 < cutforcesq) {
              MMD_float r = sqrt(rsq3);
              MMD_float p = r * rdr + 1.0;
              MMD_int m = static_cast<int>(p);
              m = m < nr - 1 ? m : nr - 1;
              p -= m;
              p = p < 1.0 ? p : 1.0;


              // rhoip = derivative of (density at atom j due to atom i)
              // rhojp = derivative of (density at atom i due to atom j)
              // phi = pair potential energy
              // phip = phi'
              // z2 = phi * r
              // z2p = (phi * r)' = (phi' r) + phi
              // psip needs both fp[i] and fp[j] terms since r_ij appears in two
              //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
              //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

              MMD_float rhoip = (rhor_spline[m * 7 + 0] * p + rhor_spline[m * 7 + 1]) * p + rhor_spline[m * 7 + 2];
              MMD_float z2p = (z2r_spline[m * 7 + 0] * p + z2r_spline[m * 7 + 1]) * p + z2r_spline[m * 7 + 2];
              MMD_float z2 = ((z2r_spline[m * 7 + 3] * p + z2r_spline[m * 7 + 4]) * p + z2r_spline[m * 7 + 5]) * p + z2r_spline[m * 7 + 6];

              MMD_float recip = 1.0 / r;
              MMD_float phi = z2 * recip;
              MMD_float phip = z2p * recip - phi * recip;
              MMD_float psip = fp[i] * rhoip + fp[j3] * rhoip + phip;
              MMD_float fpair = -psip * recip;

              fx += delx3 * fpair;
              fy += dely3 * fpair;
              fz += delz3 * fpair;

              //  	if(i==0&&j<20)
              //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
              if(j3 < nlocal) {
                  f[j3 * PAD + 0] -= delx3 * fpair;
                  f[j3 * PAD + 1] -= dely3 * fpair;
                  f[j3 * PAD + 2] -= delz3 * fpair;
              } else fpair *= 0.5;

              if(evflag) {
                  virial += delx3 * delx3 * fpair + dely3 * dely3 * fpair + delz3 * delz3 * fpair;
              }

              if(j3 < nlocal) evdwl += phi;
              else evdwl += 0.5 * phi;
          }
          if(rsq4 < cutforcesq) {
              MMD_float r = sqrt(rsq4);
              MMD_float p = r * rdr + 1.0;
              MMD_int m = static_cast<int>(p);
              m = m < nr - 1 ? m : nr - 1;
              p -= m;
              p = p < 1.0 ? p : 1.0;


              // rhoip = derivative of (density at atom j due to atom i)
              // rhojp = derivative of (density at atom i due to atom j)
              // phi = pair potential energy
              // phip = phi'
              // z2 = phi * r
              // z2p = (phi * r)' = (phi' r) + phi
              // psip needs both fp[i] and fp[j] terms since r_ij appears in two
              //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
              //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

              MMD_float rhoip = (rhor_spline[m * 7 + 0] * p + rhor_spline[m * 7 + 1]) * p + rhor_spline[m * 7 + 2];
              MMD_float z2p = (z2r_spline[m * 7 + 0] * p + z2r_spline[m * 7 + 1]) * p + z2r_spline[m * 7 + 2];
              MMD_float z2 = ((z2r_spline[m * 7 + 3] * p + z2r_spline[m * 7 + 4]) * p + z2r_spline[m * 7 + 5]) * p + z2r_spline[m * 7 + 6];

              MMD_float recip = 1.0 / r;
              MMD_float phi = z2 * recip;
              MMD_float phip = z2p * recip - phi * recip;
              MMD_float psip = fp[i] * rhoip + fp[j4] * rhoip + phip;
              MMD_float fpair = -psip * recip;

              fx += delx4 * fpair;
              fy += dely4 * fpair;
              fz += delz4 * fpair;

              //  	if(i==0&&j<20)
              //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
              if(j4 < nlocal) {
                  f[j4 * PAD + 0] -= delx4 * fpair;
                  f[j4 * PAD + 1] -= dely4 * fpair;
                  f[j4 * PAD + 2] -= delz4 * fpair;
              } else fpair *= 0.5;

              if(evflag) {
                  virial += delx4 * delx4 * fpair + dely4 * dely4 * fpair + delz4 * delz4 * fpair;
              }

              if(j4 < nlocal) evdwl += phi;
              else evdwl += 0.5 * phi;
          }
      }
    }
    for(;jj<numneigh;jj++){
        const MMD_int j = neighs[jj];

        const MMD_float delx = xtmp - x[j * PAD + 0];
        const MMD_float dely = ytmp - x[j * PAD + 1];
        const MMD_float delz = ztmp - x[j * PAD + 2];
        const MMD_float rsq = delx * delx + dely * dely + delz * delz;

        //printf("EAM: %i %i %lf %lf\n",i,j,rsq,cutforcesq);
        if(rsq < cutforcesq) {
            MMD_float r = sqrt(rsq);
            MMD_float p = r * rdr + 1.0;
            MMD_int m = static_cast<int>(p);
            m = m < nr - 1 ? m : nr - 1;
            p -= m;
            p = p < 1.0 ? p : 1.0;


            // rhoip = derivative of (density at atom j due to atom i)
            // rhojp = derivative of (density at atom i due to atom j)
            // phi = pair potential energy
            // phip = phi'
            // z2 = phi * r
            // z2p = (phi * r)' = (phi' r) + phi
            // psip needs both fp[i] and fp[j] terms since r_ij appears in two
            //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
            //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

            MMD_float rhoip = (rhor_spline[m * 7 + 0] * p + rhor_spline[m * 7 + 1]) * p + rhor_spline[m * 7 + 2];
            MMD_float z2p = (z2r_spline[m * 7 + 0] * p + z2r_spline[m * 7 + 1]) * p + z2r_spline[m * 7 + 2];
            MMD_float z2 = ((z2r_spline[m * 7 + 3] * p + z2r_spline[m * 7 + 4]) * p + z2r_spline[m * 7 + 5]) * p + z2r_spline[m * 7 + 6];

            MMD_float recip = 1.0 / r;
            MMD_float phi = z2 * recip;
            MMD_float phip = z2p * recip - phi * recip;
            MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
            MMD_float fpair = -psip * recip;

            fx += delx * fpair;
            fy += dely * fpair;
            fz += delz * fpair;

            //  	if(i==0&&j<20)
            //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
            if(j < nlocal) {
                f[j * PAD + 0] -= delx * fpair;
                f[j * PAD + 1] -= dely * fpair;
                f[j * PAD + 2] -= delz * fpair;
            } else fpair *= 0.5;

            if(evflag) {
                virial += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
            }

            if(j < nlocal) evdwl += phi;
            else evdwl += 0.5 * phi;
        }
    }

    f[i * PAD + 0] += fx;
    f[i * PAD + 1] += fy;
    f[i * PAD + 2] += fz;
  }

  eng_vdwl = evdwl;
}

/* ---------------------------------------------------------------------- */

void ForceEAM::compute_fullneigh(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{

  MMD_float evdwl = 0.0;

  virial = 0;
  // grow energy and fp arrays if necessary
  // need to be atom->nmax in length

  //#pragma omp master
  {
    if(atom.nmax > nmax) {
      nmax = atom.nmax;
      rho = new MMD_float[nmax];
      fp = new MMD_float[nmax];
    }
  }

  //#pragma omp barrier
  MMD_float* x = &atom.x[0][0];
  MMD_float* f = &atom.f[0][0];
  const int nlocal = atom.nlocal;

  // zero out density

  // rho = density at each atom
  // loop over neighbors of my atoms

  OMPFORSCHEDULE
  for(MMD_int i = 0; i < nlocal; i++) {
    int* neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int jnum = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    MMD_float rhoi = 0;

    //#pragma ivdep
    for(MMD_int jj = 0; jj < jnum; jj++) {
      const MMD_int j = neighs[jj];

      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      if(rsq < cutforcesq) {
        MMD_float p = sqrt(rsq) * rdr + 1.0;
        MMD_int m = static_cast<int>(p);
        m = m < nr - 1 ? m : nr - 1;
        p -= m;
        p = p < 1.0 ? p : 1.0;

        rhoi += ((rhor_spline[m * 7 + 3] * p + rhor_spline[m * 7 + 4]) * p + rhor_spline[m * 7 + 5]) * p + rhor_spline[m * 7 + 6];
      }
    }

    MMD_float p = 1.0 * rhoi * rdrho + 1.0;
    MMD_int m = static_cast<int>(p);
    m = MAX(1, MIN(m, nrho - 1));
    p -= m;
    p = MIN(p, 1.0);
    fp[i] = (frho_spline[m * 7 + 0] * p + frho_spline[m * 7 + 1]) * p + frho_spline[m * 7 + 2];

    // printf("fp: %lf %lf %lf %lf %lf %i %lf %lf\n",fp[i],p,frho_spline[m*7+0],frho_spline[m*7+1],frho_spline[m*7+2],m,rdrho,rho[i]);
    if(evflag) {
      evdwl += ((frho_spline[m * 7 + 3] * p + frho_spline[m * 7 + 4]) * p + frho_spline[m * 7 + 5]) * p + frho_spline[m * 7 + 6];
    }

  }

  // //#pragma omp barrier
  // fp = derivative of embedding energy at each atom
  // phi = embedding energy at each atom

  // communicate derivative of embedding function

  //#pragma omp master
  {
    communicate(atom, comm);
  }

  //#pragma omp barrier

  MMD_float t_virial = 0;
  // compute forces on each atom
  // loop over neighbors of my atoms

  OMPFORSCHEDULE
  for(MMD_int i = 0; i < nlocal; i++) {
    int* neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneigh = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];

    MMD_float fx = 0.0;
    MMD_float fy = 0.0;
    MMD_float fz = 0.0;

    //#pragma ivdep
    for(MMD_int jj = 0; jj < numneigh; jj++) {
      const MMD_int j = neighs[jj];

      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;
      //printf("EAM: %i %i %lf %lf // %lf %lf\n",i,j,rsq,cutforcesq,fp[i],fp[j]);

      if(rsq < cutforcesq) {
        MMD_float r = sqrt(rsq);
        MMD_float p = r * rdr + 1.0;
        MMD_int m = static_cast<int>(p);
        m = m < nr - 1 ? m : nr - 1;
        p -= m;
        p = p < 1.0 ? p : 1.0;


        // rhoip = derivative of (density at atom j due to atom i)
        // rhojp = derivative of (density at atom i due to atom j)
        // phi = pair potential energy
        // phip = phi'
        // z2 = phi * r
        // z2p = (phi * r)' = (phi' r) + phi
        // psip needs both fp[i] and fp[j] terms since r_ij appears in two
        //   terms of embed eng: Fi(sum rho_ij) and Fj(sum rho_ji)
        //   hence embed' = Fi(sum rho_ij) rhojp + Fj(sum rho_ji) rhoip

        MMD_float rhoip = (rhor_spline[m * 7 + 0] * p + rhor_spline[m * 7 + 1]) * p + rhor_spline[m * 7 + 2];
        MMD_float z2p = (z2r_spline[m * 7 + 0] * p + z2r_spline[m * 7 + 1]) * p + z2r_spline[m * 7 + 2];
        MMD_float z2 = ((z2r_spline[m * 7 + 3] * p + z2r_spline[m * 7 + 4]) * p + z2r_spline[m * 7 + 5]) * p + z2r_spline[m * 7 + 6];

        MMD_float recip = 1.0 / r;
        MMD_float phi = z2 * recip;
        MMD_float phip = z2p * recip - phi * recip;
        MMD_float psip = fp[i] * rhoip + fp[j] * rhoip + phip;
        MMD_float fpair = -psip * recip;

        fx += delx * fpair;
        fy += dely * fpair;
        fz += delz * fpair;
        //  	if(i==0&&j<20)
        //      printf("fpair: %i %i %lf %lf %lf %lf\n",i,j,fpair,delx,dely,delz);
        fpair *= 0.5;

        if(evflag) {
          t_virial += delx * delx * fpair + dely * dely * fpair + delz * delz * fpair;
          evdwl += 0.5 * phi;
        }

      }
    }

    f[i * PAD + 0] = fx;
    f[i * PAD + 1] = fy;
    f[i * PAD + 2] = fz;

  }

  //#pragma omp atomic
  virial += t_virial;
  //#pragma omp atomic
  eng_vdwl += 2.0 * evdwl;

  //#pragma omp barrier
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void ForceEAM::coeff(const char* arg)
{



  // read funcfl file if hasn't already been read
  // store filename in Funcfl data struct


  read_file(arg);
  int n = strlen(arg) + 1;
  funcfl.file = new char[n];

  // set setflag and map only for i,i type pairs
  // set mass of atom type if i = j

  //atom->mass = funcfl.mass;
  cutmax = funcfl.cut;

  cutforcesq = cutmax * cutmax;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void ForceEAM::init_style()
{
  // convert read-in file(s) to arrays and spline them

  file2array();
  array2spline();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */



/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
------------------------------------------------------------------------- */

void ForceEAM::read_file(const char* filename)
{
  Funcfl* file = &funcfl;

  //me = 0;
  FILE* fptr;
  char line[MAXLINE];

  int flag = 0;

  if(me == 0) {
    fptr = fopen(filename, "r");

    if(fptr == NULL) {
      printf("Can't open EAM Potential file: %s\n", filename);
      flag = 1;
    }
  }

  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(flag) {
    MPI_Finalize();
    exit(0);
  }

  int tmp;

  if(me == 0) {
    fgets(line, MAXLINE, fptr);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%d %lg", &tmp, &file->mass);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%d %lg %d %lg %lg",
           &file->nrho, &file->drho, &file->nr, &file->dr, &file->cut);
  }

  //printf("Read: %lf %i %lf %i %lf %lf\n",file->mass,file->nrho,file->drho,file->nr,file->dr,file->cut);
  MPI_Bcast(&file->mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->nrho, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->drho, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->nr, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->dr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&file->cut, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  mass = file->mass;
  file->frho = new MMD_float[file->nrho + 1];
  file->rhor = new MMD_float[file->nr + 1];
  file->zr = new MMD_float[file->nr + 1];

  if(me == 0) grab(fptr, file->nrho, file->frho);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->frho, file->nrho, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->frho, file->nrho, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(me == 0) grab(fptr, file->nr, file->zr);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->zr, file->nr, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->zr, file->nr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if(me == 0) grab(fptr, file->nr, file->rhor);

  if(sizeof(MMD_float) == 4)
    MPI_Bcast(file->rhor, file->nr, MPI_FLOAT, 0, MPI_COMM_WORLD);
  else
    MPI_Bcast(file->rhor, file->nr, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  for(int i = file->nrho; i > 0; i--) file->frho[i] = file->frho[i - 1];

  for(int i = file->nr; i > 0; i--) file->rhor[i] = file->rhor[i - 1];

  for(int i = file->nr; i > 0; i--) file->zr[i] = file->zr[i - 1];

  if(me == 0) fclose(fptr);
}

/* ----------------------------------------------------------------------
   convert read-in funcfl potential(s) to standard array format
   interpolate all file values to a single grid and cutoff
------------------------------------------------------------------------- */

void ForceEAM::file2array()
{
  int i, j, k, m, n;
  int ntypes = 1;
  double sixth = 1.0 / 6.0;

  // determine max function params from all active funcfl files
  // active means some element is pointing at it via map

  int active;
  double rmax, rhomax;
  dr = drho = rmax = rhomax = 0.0;

  active = 0;
  Funcfl* file = &funcfl;
  dr = MAX(dr, file->dr);
  drho = MAX(drho, file->drho);
  rmax = MAX(rmax, (file->nr - 1) * file->dr);
  rhomax = MAX(rhomax, (file->nrho - 1) * file->drho);

  // set nr,nrho from cutoff and spacings
  // 0.5 is for round-off in divide

  nr = static_cast<int>(rmax / dr + 0.5);
  nrho = static_cast<int>(rhomax / drho + 0.5);

  // ------------------------------------------------------------------
  // setup frho arrays
  // ------------------------------------------------------------------

  // allocate frho arrays
  // nfrho = # of funcfl files + 1 for zero array

  frho = new MMD_float[nrho + 1];

  // interpolate each file's frho to a single grid and cutoff

  double r, p, cof1, cof2, cof3, cof4;

  n = 0;

  for(m = 1; m <= nrho; m++) {
    r = (m - 1) * drho;
    p = r / file->drho + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file->nrho - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    frho[m] = cof1 * file->frho[k - 1] + cof2 * file->frho[k] +
              cof3 * file->frho[k + 1] + cof4 * file->frho[k + 2];
  }


  // ------------------------------------------------------------------
  // setup rhor arrays
  // ------------------------------------------------------------------

  // allocate rhor arrays
  // nrhor = # of funcfl files

  rhor = new MMD_float[nr + 1];

  // interpolate each file's rhor to a single grid and cutoff

  for(m = 1; m <= nr; m++) {
    r = (m - 1) * dr;
    p = r / file->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, file->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    rhor[m] = cof1 * file->rhor[k - 1] + cof2 * file->rhor[k] +
              cof3 * file->rhor[k + 1] + cof4 * file->rhor[k + 2];
    //if(m==119)printf("BuildRho: %e %e %e %e %e %e\n",rhor[m],cof1,cof2,cof3,cof4,file->rhor[k]);
  }

  // type2rhor[i][j] = which rhor array (0 to nrhor-1) each type pair maps to
  // for funcfl files, I,J mapping only depends on I
  // OK if map = -1 (non-EAM atom in pair hybrid) b/c type2rhor not used

  // ------------------------------------------------------------------
  // setup z2r arrays
  // ------------------------------------------------------------------

  // allocate z2r arrays
  // nz2r = N*(N+1)/2 where N = # of funcfl files

  z2r = new MMD_float[nr + 1];

  // create a z2r array for each file against other files, only for I >= J
  // interpolate zri and zrj to a single grid and cutoff

  double zri, zrj;

  Funcfl* ifile = &funcfl;
  Funcfl* jfile = &funcfl;

  for(m = 1; m <= nr; m++) {
    r = (m - 1) * dr;

    p = r / ifile->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, ifile->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    zri = cof1 * ifile->zr[k - 1] + cof2 * ifile->zr[k] +
          cof3 * ifile->zr[k + 1] + cof4 * ifile->zr[k + 2];

    p = r / jfile->dr + 1.0;
    k = static_cast<int>(p);
    k = MIN(k, jfile->nr - 2);
    k = MAX(k, 2);
    p -= k;
    p = MIN(p, 2.0);
    cof1 = -sixth * p * (p - 1.0) * (p - 2.0);
    cof2 = 0.5 * (p * p - 1.0) * (p - 2.0);
    cof3 = -0.5 * p * (p + 1.0) * (p - 2.0);
    cof4 = sixth * p * (p * p - 1.0);
    zrj = cof1 * jfile->zr[k - 1] + cof2 * jfile->zr[k] +
          cof3 * jfile->zr[k + 1] + cof4 * jfile->zr[k + 2];

    z2r[m] = 27.2 * 0.529 * zri * zrj;
  }

}

/* ---------------------------------------------------------------------- */

void ForceEAM::array2spline()
{
  rdr = 1.0 / dr;
  rdrho = 1.0 / drho;

  frho_spline = new MMD_float[(nrho + 1) * 7];
  rhor_spline = new MMD_float[(nr + 1) * 7];
  z2r_spline = new MMD_float[(nr + 1) * 7];

  interpolate(nrho, drho, frho, frho_spline);

  interpolate(nr, dr, rhor, rhor_spline);

  // printf("Rhor: %lf\n",rhor(119));

  interpolate(nr, dr, z2r, z2r_spline);

  //printf("RhorSpline: %e %e %e %e\n",rhor_spline(119,3),rhor_spline(119,4),rhor_spline(119,5),rhor_spline(119,6));
  //printf("FrhoSpline: %e %e %e %e\n",frho_spline(119,3),frho_spline(119,4),frho_spline(119,5),frho_spline(119,6));

}

/* ---------------------------------------------------------------------- */

void ForceEAM::interpolate(MMD_int n, MMD_float delta, MMD_float* f, MMD_float* spline)
{
  for(int m = 1; m <= n; m++) spline[m * 7 + 6] = f[m];

  spline[1 * 7 + 5] = spline[2 * 7 + 6] - spline[1 * 7 + 6];
  spline[2 * 7 + 5] = 0.5 * (spline[3 * 7 + 6] - spline[1 * 7 + 6]);
  spline[(n - 1) * 7 + 5] = 0.5 * (spline[n * 7 + 6] - spline[(n - 2) * 7 + 6]);
  spline[n * 7 + 5] = spline[n * 7 + 6] - spline[(n - 1) * 7 + 6];

  for(int m = 3; m <= n - 2; m++)
    spline[m * 7 + 5] = ((spline[(m - 2) * 7 + 6] - spline[(m + 2) * 7 + 6]) +
                         8.0 * (spline[(m + 1) * 7 + 6] - spline[(m - 1) * 7 + 6])) / 12.0;

  for(int m = 1; m <= n - 1; m++) {
    spline[m * 7 + 4] = 3.0 * (spline[(m + 1) * 7 + 6] - spline[m * 7 + 6]) -
                        2.0 * spline[m * 7 + 5] - spline[(m + 1) * 7 + 5];
    spline[m * 7 + 3] = spline[m * 7 + 5] + spline[(m + 1) * 7 + 5] -
                        2.0 * (spline[(m + 1) * 7 + 6] - spline[m * 7 + 6]);
  }

  spline[n * 7 + 4] = 0.0;
  spline[n * 7 + 3] = 0.0;

  for(int m = 1; m <= n; m++) {
    spline[m * 7 + 2] = spline[m * 7 + 5] / delta;
    spline[m * 7 + 1] = 2.0 * spline[m * 7 + 4] / delta;
    spline[m * 7 + 0] = 3.0 * spline[m * 7 + 3] / delta;
  }
}

/* ----------------------------------------------------------------------
   grab n values from file fp and put them in list
   values can be several to a line
   only called by proc 0
------------------------------------------------------------------------- */

void ForceEAM::grab(FILE* fptr, MMD_int n, MMD_float* list)
{
  char* ptr;
  char line[MAXLINE];

  int i = 0;

  while(i < n) {
    fgets(line, MAXLINE, fptr);
    ptr = strtok(line, " \t\n\r\f");
    list[i++] = atof(ptr);

    while(ptr = strtok(NULL, " \t\n\r\f")) list[i++] = atof(ptr);
  }
}

/* ---------------------------------------------------------------------- */

MMD_float ForceEAM::single(int i, int j, int itype, int jtype,
                           MMD_float rsq, MMD_float factor_coul, MMD_float factor_lj,
                           MMD_float &fforce)
{
  int m;
  MMD_float r, p, rhoip, rhojp, z2, z2p, recip, phi, phip, psip;
  MMD_float* coeff;

  r = sqrt(rsq);
  p = r * rdr + 1.0;
  m = static_cast<int>(p);
  m = MIN(m, nr - 1);
  p -= m;
  p = MIN(p, 1.0);

  coeff = &rhor_spline[m * 7 + 0];
  rhoip = (coeff[0] * p + coeff[1]) * p + coeff[2];
  coeff = &rhor_spline[m * 7 + 0];
  rhojp = (coeff[0] * p + coeff[1]) * p + coeff[2];
  coeff = &z2r_spline[m * 7 + 0];
  z2p = (coeff[0] * p + coeff[1]) * p + coeff[2];
  z2 = ((coeff[3] * p + coeff[4]) * p + coeff[5]) * p + coeff[6];

  recip = 1.0 / r;
  phi = z2 * recip;
  phip = z2p * recip - phi * recip;
  psip = fp[i] * rhojp + fp[j] * rhoip + phip;
  fforce = -psip * recip;

  return phi;
}

void ForceEAM::communicate(Atom &atom, Comm &comm)
{

  int iswap;
  int pbc_flags[4];
  MMD_float* buf;
  MPI_Request request;
  MPI_Status status;

  for(iswap = 0; iswap < comm.nswap; iswap++) {

    /* pack buffer */

    pbc_flags[0] = comm.pbc_any[iswap];
    pbc_flags[1] = comm.pbc_flagx[iswap];
    pbc_flags[2] = comm.pbc_flagy[iswap];
    pbc_flags[3] = comm.pbc_flagz[iswap];
    //timer->stamp_extra_start();

    int size = pack_comm(comm.sendnum[iswap], iswap, comm.buf_send, comm.sendlist);
    //timer->stamp_extra_stop(TIME_TEST);


    /* exchange with another proc
       if self, set recv buffer to send buffer */

    if(comm.sendproc[iswap] != me) {
      if(sizeof(MMD_float) == 4) {
        MPI_Irecv(comm.buf_recv, comm.comm_recv_size[iswap], MPI_FLOAT,
                  comm.recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(comm.buf_send, comm.comm_send_size[iswap], MPI_FLOAT,
                 comm.sendproc[iswap], 0, MPI_COMM_WORLD);
      } else {
        MPI_Irecv(comm.buf_recv, comm.comm_recv_size[iswap], MPI_DOUBLE,
                  comm.recvproc[iswap], 0, MPI_COMM_WORLD, &request);
        MPI_Send(comm.buf_send, comm.comm_send_size[iswap], MPI_DOUBLE,
                 comm.sendproc[iswap], 0, MPI_COMM_WORLD);
      }

      MPI_Wait(&request, &status);
      buf = comm.buf_recv;
    } else buf = comm.buf_send;

    /* unpack buffer */

    unpack_comm(comm.recvnum[iswap], comm.firstrecv[iswap], buf);
  }
}
/* ---------------------------------------------------------------------- */

int ForceEAM::pack_comm(int n, int iswap, MMD_float* buf, int** asendlist)
{
  int i, j, m;

  m = 0;

  for(i = 0; i < n; i++) {
    j = asendlist[iswap][i];
    buf[i] = fp[j];
  }

  return 1;
}

/* ---------------------------------------------------------------------- */

void ForceEAM::unpack_comm(int n, int first, MMD_float* buf)
{
  int i, m, last;

  m = 0;
  last = first + n;

  for(i = first; i < last; i++) fp[i] = buf[m++];
}

/* ---------------------------------------------------------------------- */

int ForceEAM::pack_reverse_comm(int n, int first, MMD_float* buf)
{
  int i, m, last;

  m = 0;
  last = first + n;

  for(i = first; i < last; i++) buf[m++] = rho[i];

  return 1;
}

/* ---------------------------------------------------------------------- */

void ForceEAM::unpack_reverse_comm(int n, int* list, MMD_float* buf)
{
  int i, j, m;

  m = 0;

  for(i = 0; i < n; i++) {
    j = list[i];
    rho[j] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

MMD_float ForceEAM::memory_usage()
{
  MMD_int bytes = 2 * nmax * sizeof(MMD_float);
  return bytes;
}


void ForceEAM::bounds(char* str, int nmax, int &nlo, int &nhi)
{
  char* ptr = strchr(str, '*');

  if(ptr == NULL) {
    nlo = nhi = atoi(str);
  } else if(strlen(str) == 1) {
    nlo = 1;
    nhi = nmax;
  } else if(ptr == str) {
    nlo = 1;
    nhi = atoi(ptr + 1);
  } else if(strlen(ptr + 1) == 0) {
    nlo = atoi(str);
    nhi = nmax;
  } else {
    nlo = atoi(str);
    nhi = atoi(ptr + 1);
  }

  if(nlo < 1 || nhi > nmax) printf("Numeric index is out of bounds");
}
