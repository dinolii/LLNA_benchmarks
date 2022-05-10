/* $Id: lbm.c,v 1.7 2004/05/11 08:45:02 pohlt Exp $ */

/*############################################################################*/

#include "lbm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
#include <immintrin.h>

#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
#include <omp.h>
#endif

/*############################################################################*/

#define DFL1 (1.0/ 3.0)
#define DFL2 (1.0/18.0)
#define DFL3 (1.0/36.0)

/*############################################################################*/

void LBM_allocateGrid( double** ptr ) {
	const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES,
	             size   = sizeof( LBM_Grid ) + 2*margin*sizeof( double );

	*ptr = malloc( size );
	if( ! *ptr ) {
		printf( "LBM_allocateGrid: could not allocate %.1f MByte\n",
		        size / (1024.0*1024.0) );
		exit( 1 );
	}
#ifndef SPEC
	printf( "LBM_allocateGrid: allocated %.1f MByte\n",
	        size / (1024.0*1024.0) );
#endif
	*ptr += margin;
}

/*############################################################################*/

void LBM_freeGrid( double** ptr ) {
	const size_t margin = 2*SIZE_X*SIZE_Y*N_CELL_ENTRIES;

	free( *ptr-margin );
	*ptr = NULL;
}

/*############################################################################*/

void LBM_initializeGrid( LBM_Grid grid ) {
	SWEEP_VAR

	/*voption indep*/
#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
#pragma omp parallel for
#endif
	SWEEP_START( 0, 0, -2, 0, 0, SIZE_Z+2 )
		LOCAL( grid, C  ) = DFL1;
		LOCAL( grid, N  ) = DFL2;
		LOCAL( grid, S  ) = DFL2;
		LOCAL( grid, E  ) = DFL2;
		LOCAL( grid, W  ) = DFL2;
		LOCAL( grid, T  ) = DFL2;
		LOCAL( grid, B  ) = DFL2;
		LOCAL( grid, NE ) = DFL3;
		LOCAL( grid, NW ) = DFL3;
		LOCAL( grid, SE ) = DFL3;
		LOCAL( grid, SW ) = DFL3;
		LOCAL( grid, NT ) = DFL3;
		LOCAL( grid, NB ) = DFL3;
		LOCAL( grid, ST ) = DFL3;
		LOCAL( grid, SB ) = DFL3;
		LOCAL( grid, ET ) = DFL3;
		LOCAL( grid, EB ) = DFL3;
		LOCAL( grid, WT ) = DFL3;
		LOCAL( grid, WB ) = DFL3;

		CLEAR_ALL_FLAGS_SWEEP( grid );
	SWEEP_END
}

/*############################################################################*/

void LBM_swapGrids( LBM_GridPtr* grid1, LBM_GridPtr* grid2 ) {
	LBM_GridPtr aux = *grid1;
	*grid1 = *grid2;
	*grid2 = aux;
}

/*############################################################################*/

void LBM_loadObstacleFile( LBM_Grid grid, const char* filename ) {
	int x,  y,  z;

	FILE* file = fopen( filename, "rb" );

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( fgetc( file ) != '.' ) SET_FLAG( grid, x, y, z, OBSTACLE );
			}
			fgetc( file );
		}
		fgetc( file );
	}

	fclose( file );
}

/*############################################################################*/

void LBM_initializeSpecialCellsForLDC( LBM_Grid grid ) {
	int x,  y,  z;

	/*voption indep*/
#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
#pragma omp parallel for private( x, y )
#endif
	for( z = -2; z < SIZE_Z+2; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( x == 0 || x == SIZE_X-1 ||
				    y == 0 || y == SIZE_Y-1 ||
				    z == 0 || z == SIZE_Z-1 ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );
				}
				else {
					if( (z == 1 || z == SIZE_Z-2) &&
					     x > 1 && x < SIZE_X-2 &&
					     y > 1 && y < SIZE_Y-2 ) {
						SET_FLAG( grid, x, y, z, ACCEL );
					}
				}
			}
		}
	}
}

/*############################################################################*/

void LBM_initializeSpecialCellsForChannel( LBM_Grid grid ) {
	int x,  y,  z;

	/*voption indep*/
#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
#pragma omp parallel for private( x, y )
#endif
	for( z = -2; z < SIZE_Z+2; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				if( x == 0 || x == SIZE_X-1 ||
				    y == 0 || y == SIZE_Y-1 ) {
					SET_FLAG( grid, x, y, z, OBSTACLE );

					if( (z == 0 || z == SIZE_Z-1) &&
					    ! TEST_FLAG( grid, x, y, z, OBSTACLE ))
						SET_FLAG( grid, x, y, z, IN_OUT_FLOW );
				}
			}
		}
	}
}

/*############################################################################*/

void LBM_performStreamCollideBGK( LBM_Grid srcGrid, LBM_Grid dstGrid ) {
	SWEEP_VAR

	double ux, uy, uz, u2, rho;

	/*voption indep*/
#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
#pragma omp parallel for private( ux, uy, uz, u2, rho )
#endif
	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
			DST_C ( dstGrid ) = SRC_C ( srcGrid );
			DST_S ( dstGrid ) = SRC_N ( srcGrid );
			DST_N ( dstGrid ) = SRC_S ( srcGrid );
			DST_W ( dstGrid ) = SRC_E ( srcGrid );
			DST_E ( dstGrid ) = SRC_W ( srcGrid );
			DST_B ( dstGrid ) = SRC_T ( srcGrid );
			DST_T ( dstGrid ) = SRC_B ( srcGrid );
			DST_SW( dstGrid ) = SRC_NE( srcGrid );
			DST_SE( dstGrid ) = SRC_NW( srcGrid );
			DST_NW( dstGrid ) = SRC_SE( srcGrid );
			DST_NE( dstGrid ) = SRC_SW( srcGrid );
			DST_SB( dstGrid ) = SRC_NT( srcGrid );
			DST_ST( dstGrid ) = SRC_NB( srcGrid );
			DST_NB( dstGrid ) = SRC_ST( srcGrid );
			DST_NT( dstGrid ) = SRC_SB( srcGrid );
			DST_WB( dstGrid ) = SRC_ET( srcGrid );
			DST_WT( dstGrid ) = SRC_EB( srcGrid );
			DST_EB( dstGrid ) = SRC_WT( srcGrid );
			DST_ET( dstGrid ) = SRC_WB( srcGrid );
			continue;
		}

		rho = + SRC_C ( srcGrid ) + SRC_N ( srcGrid )
		      + SRC_S ( srcGrid ) + SRC_E ( srcGrid )
		      + SRC_W ( srcGrid ) + SRC_T ( srcGrid )
		      + SRC_B ( srcGrid ) + SRC_NE( srcGrid )
		      + SRC_NW( srcGrid ) + SRC_SE( srcGrid )
		      + SRC_SW( srcGrid ) + SRC_NT( srcGrid )
		      + SRC_NB( srcGrid ) + SRC_ST( srcGrid )
		      + SRC_SB( srcGrid ) + SRC_ET( srcGrid )
		      + SRC_EB( srcGrid ) + SRC_WT( srcGrid )
		      + SRC_WB( srcGrid );

		ux = + SRC_E ( srcGrid ) - SRC_W ( srcGrid )
		     + SRC_NE( srcGrid ) - SRC_NW( srcGrid )
		     + SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_ET( srcGrid ) + SRC_EB( srcGrid )
		     - SRC_WT( srcGrid ) - SRC_WB( srcGrid );
		uy = + SRC_N ( srcGrid ) - SRC_S ( srcGrid )
		     + SRC_NE( srcGrid ) + SRC_NW( srcGrid )
		     - SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_NT( srcGrid ) + SRC_NB( srcGrid )
		     - SRC_ST( srcGrid ) - SRC_SB( srcGrid );
		uz = + SRC_T ( srcGrid ) - SRC_B ( srcGrid )
		     + SRC_NT( srcGrid ) - SRC_NB( srcGrid )
		     + SRC_ST( srcGrid ) - SRC_SB( srcGrid )
		     + SRC_ET( srcGrid ) - SRC_EB( srcGrid )
		     + SRC_WT( srcGrid ) - SRC_WB( srcGrid );

		ux /= rho;
		uy /= rho;
		uz /= rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005;
			uy = 0.002;
			uz = 0.000;
		}

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);
		DST_C ( dstGrid ) = (1.0-OMEGA)*SRC_C ( srcGrid ) + DFL1*OMEGA*rho*(1.0                                 - u2);

		DST_N ( dstGrid ) = (1.0-OMEGA)*SRC_N ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
		DST_S ( dstGrid ) = (1.0-OMEGA)*SRC_S ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
		DST_E ( dstGrid ) = (1.0-OMEGA)*SRC_E ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
		DST_W ( dstGrid ) = (1.0-OMEGA)*SRC_W ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
		DST_T ( dstGrid ) = (1.0-OMEGA)*SRC_T ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
		DST_B ( dstGrid ) = (1.0-OMEGA)*SRC_B ( srcGrid ) + DFL2*OMEGA*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

		DST_NE( dstGrid ) = (1.0-OMEGA)*SRC_NE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		DST_NW( dstGrid ) = (1.0-OMEGA)*SRC_NW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		DST_SE( dstGrid ) = (1.0-OMEGA)*SRC_SE( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		DST_SW( dstGrid ) = (1.0-OMEGA)*SRC_SW( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		DST_NT( dstGrid ) = (1.0-OMEGA)*SRC_NT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		DST_NB( dstGrid ) = (1.0-OMEGA)*SRC_NB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		DST_ST( dstGrid ) = (1.0-OMEGA)*SRC_ST( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		DST_SB( dstGrid ) = (1.0-OMEGA)*SRC_SB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		DST_ET( dstGrid ) = (1.0-OMEGA)*SRC_ET( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		DST_EB( dstGrid ) = (1.0-OMEGA)*SRC_EB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		DST_WT( dstGrid ) = (1.0-OMEGA)*SRC_WT( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		DST_WB( dstGrid ) = (1.0-OMEGA)*SRC_WB( srcGrid ) + DFL3*OMEGA*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
	SWEEP_END
}


//void LBM_performStreamCollideTRT_Vec_RC( LBM_Grid srcGrid, LBM_Grid dstGrid ) {
//    SWEEP_VAR
//
//    double ux, uy, uz, u2, rho;
//    int count_tt=0; int count_ff=0; int count=0;
//
//    const double lambda0 = 1.0/(0.5+3.0/(16.0*(1.0/OMEGA-0.5)));
//    double fs[N_CELL_ENTRIES], fa[N_CELL_ENTRIES],
//            feqs[N_CELL_ENTRIES], feqa[N_CELL_ENTRIES];
//#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
//#pragma omp parallel for private( ux, uy, uz, u2, rho, fs, fa, feqs, feqa )
//#endif
//    SWEEP_2STARTV1(0, 0, 0, 0, 0, SIZE_Z)
//        //printf("DEBUG: i3=%d\n", i);
//        if (!TEST_FLAG_SWEEP_NEXT3(srcGrid, OBSTACLE)  && !TEST_FLAG_SWEEP_NEXT2(srcGrid, OBSTACLE)  &&!TEST_FLAG_SWEEP_NEXT(srcGrid, OBSTACLE) && !TEST_FLAG_SWEEP(srcGrid, OBSTACLE)) {
//            /// Packing Data for Vectorization
//            printf("Debug i2:%d\n", i);
//            printf("Debug i2:%d\n", i + 1 * N_CELL_ENTRIES);
//            printf("Debug i2:%d\n", i + 2 * N_CELL_ENTRIES);
//            printf("Debug i2:%d\n", i + 3 * N_CELL_ENTRIES);
//            __m256d V_SRC_C = _mm256_set_pd(SRC_C_NEXT3(srcGrid), SRC_C_NEXT2(srcGrid), SRC_C_NEXT(srcGrid), SRC_C(srcGrid));
//            __m256d V_SRC_N = _mm256_set_pd(SRC_N_NEXT3(srcGrid), SRC_N_NEXT2(srcGrid), SRC_N_NEXT(srcGrid), SRC_N(srcGrid));
//            __m256d V_SRC_S = _mm256_set_pd(SRC_S_NEXT3(srcGrid), SRC_S_NEXT2(srcGrid), SRC_S_NEXT(srcGrid), SRC_S(srcGrid));
//            __m256d V_SRC_E = _mm256_set_pd(SRC_E_NEXT3(srcGrid), SRC_E_NEXT2(srcGrid), SRC_E_NEXT(srcGrid), SRC_E(srcGrid));
//            __m256d V_SRC_W = _mm256_set_pd(SRC_W_NEXT3(srcGrid), SRC_W_NEXT2(srcGrid), SRC_W_NEXT(srcGrid), SRC_W(srcGrid));
//            __m256d V_SRC_T = _mm256_set_pd(SRC_T_NEXT3(srcGrid), SRC_T_NEXT2(srcGrid), SRC_T_NEXT(srcGrid), SRC_T(srcGrid));
//            __m256d V_SRC_B = _mm256_set_pd(SRC_B_NEXT3(srcGrid), SRC_B_NEXT2(srcGrid), SRC_B_NEXT(srcGrid), SRC_B(srcGrid));
//            __m256d V_SRC_NE = _mm256_set_pd(SRC_NE_NEXT3(srcGrid), SRC_NE_NEXT2(srcGrid), SRC_NE_NEXT(srcGrid), SRC_NE(srcGrid));
//
//            __m256d V_SRC_NW = _mm256_set_pd(SRC_NW_NEXT3(srcGrid), SRC_NW_NEXT2(srcGrid), SRC_NW_NEXT(srcGrid), SRC_NW(srcGrid));
//            __m256d V_SRC_SE = _mm256_set_pd(SRC_SE_NEXT3(srcGrid), SRC_SE_NEXT2(srcGrid), SRC_SE_NEXT(srcGrid), SRC_SE(srcGrid));
//            __m256d V_SRC_SW = _mm256_set_pd(SRC_SW_NEXT3(srcGrid), SRC_SW_NEXT2(srcGrid), SRC_SW_NEXT(srcGrid), SRC_SW(srcGrid));
//            __m256d V_SRC_NT = _mm256_set_pd(SRC_NT_NEXT3(srcGrid), SRC_NT_NEXT2(srcGrid), SRC_NT_NEXT(srcGrid), SRC_NT(srcGrid));
//            __m256d V_SRC_NB = _mm256_set_pd(SRC_NB_NEXT3(srcGrid), SRC_NB_NEXT2(srcGrid), SRC_NB_NEXT(srcGrid), SRC_NB(srcGrid));
//            __m256d V_SRC_ST = _mm256_set_pd(SRC_ST_NEXT3(srcGrid), SRC_ST_NEXT2(srcGrid), SRC_ST_NEXT(srcGrid), SRC_ST(srcGrid));
//            __m256d V_SRC_SB = _mm256_set_pd(SRC_SB_NEXT3(srcGrid), SRC_SB_NEXT2(srcGrid), SRC_SB_NEXT(srcGrid), SRC_SB(srcGrid));
//            __m256d V_SRC_ET = _mm256_set_pd(SRC_ET_NEXT3(srcGrid), SRC_ET_NEXT2(srcGrid), SRC_ET_NEXT(srcGrid), SRC_ET(srcGrid));
//            __m256d V_SRC_EB = _mm256_set_pd(SRC_EB_NEXT3(srcGrid), SRC_EB_NEXT2(srcGrid), SRC_EB_NEXT(srcGrid), SRC_EB(srcGrid));
//            __m256d V_SRC_WT = _mm256_set_pd(SRC_WT_NEXT3(srcGrid), SRC_WT_NEXT2(srcGrid), SRC_WT_NEXT(srcGrid), SRC_WT(srcGrid));
//            __m256d V_SRC_WB = _mm256_set_pd(SRC_WB_NEXT3(srcGrid), SRC_WB_NEXT2(srcGrid), SRC_WB_NEXT(srcGrid), SRC_WB(srcGrid));
//
//            __m256d V_rho = +V_SRC_C + V_SRC_N
//                            + V_SRC_S + V_SRC_E
//                            + V_SRC_W + V_SRC_T
//                            + V_SRC_B + V_SRC_NE
//                            + V_SRC_NW + V_SRC_SE
//                            + V_SRC_SW + V_SRC_NT
//                            + V_SRC_NB + V_SRC_ST
//                            + V_SRC_SB + V_SRC_ET
//                            + V_SRC_EB + V_SRC_WT
//                            + V_SRC_WB;
//
//            __m256d V_ux = +V_SRC_E - V_SRC_W
//                           + V_SRC_NE - V_SRC_NW
//                           + V_SRC_SE - V_SRC_SW
//                           + V_SRC_ET + V_SRC_EB
//                           - V_SRC_WT - V_SRC_WB;
//
//            __m256d V_uy = +V_SRC_N - V_SRC_S
//                           + V_SRC_NE + V_SRC_NW
//                           - V_SRC_SE - V_SRC_SW
//                           + V_SRC_NT + V_SRC_NB
//                           - V_SRC_ST - V_SRC_SB;
//
//            __m256d V_uz = +V_SRC_T - V_SRC_B
//                           + V_SRC_NT - V_SRC_NB
//                           + V_SRC_ST - V_SRC_SB
//                           + V_SRC_ET - V_SRC_EB
//                           + V_SRC_WT - V_SRC_WB;
//
//            V_ux = V_ux / V_rho;
//            V_uy = V_uy / V_rho;
//            V_uz = V_uz / V_rho;
//
//            if (TEST_FLAG_SWEEP(srcGrid, ACCEL)) {
//                V_ux[0] = 0.005;
//                V_uy[0] = 0.002;
//                V_uz[0] = 0.000;
//            }
//
//            if (TEST_FLAG_SWEEP_NEXT(srcGrid, ACCEL)) {
//                V_ux[1] = 0.005;
//                V_uy[1] = 0.002;
//                V_uz[1] = 0.000;
//            }
//            if (TEST_FLAG_SWEEP_NEXT2(srcGrid, ACCEL)) {
//                V_ux[2] = 0.005;
//                V_uy[2] = 0.002;
//                V_uz[2] = 0.000;
//            }
//            if (TEST_FLAG_SWEEP_NEXT3(srcGrid, ACCEL)) {
//                V_ux[2] = 0.005;
//                V_uy[2] = 0.002;
//                V_uz[2] = 0.000;
//            }
//
//            __m256d V_u2 = 1.5 * (V_ux * V_ux + V_uy * V_uy + V_uz * V_uz);
//
//            __m256d V_feqs_C = DFL1 * V_rho * (1.0 - V_u2);
//            __m256d V_feqs_N = DFL2 * V_rho * (1.0 + 4.5 * (+V_uy) * (+V_uy) - V_u2);
//            __m256d V_feqs_S = V_feqs_N;
//            __m256d V_feqs_E = DFL2 * V_rho * (1.0 + 4.5 * (+V_ux) * (+V_ux) - V_u2);
//            __m256d V_feqs_W = V_feqs_E;
//            __m256d V_feqs_T = DFL2 * V_rho * (1.0 + 4.5 * (+V_uz) * (+V_uz) - V_u2);
//            __m256d V_feqs_B = V_feqs_T;
//            __m256d V_feqs_NE = DFL3 * V_rho * (1.0 + 4.5 * (+V_ux + V_uy) * (+V_ux + V_uy) - V_u2);
//            __m256d V_feqs_SW = V_feqs_NE;
//            __m256d V_feqs_NW = DFL3 * V_rho * (1.0 + 4.5 * (-V_ux + V_uy) * (-V_ux + V_uy) - V_u2);
//            __m256d V_feqs_SE = V_feqs_NW;
//            __m256d V_feqs_NT = DFL3 * V_rho * (1.0 + 4.5 * (+V_uy + V_uz) * (+V_uy + V_uz) - V_u2);
//            __m256d V_feqs_SB = V_feqs_NT;
//            __m256d V_feqs_NB = DFL3 * V_rho * (1.0 + 4.5 * (+V_uy - V_uz) * (+V_uy - V_uz) - V_u2);
//            __m256d V_feqs_ST = V_feqs_NB;
//            __m256d V_feqs_ET = DFL3 * V_rho * (1.0 + 4.5 * (+V_ux + V_uz) * (+V_ux + V_uz) - V_u2);
//            __m256d V_feqs_WB = V_feqs_ET;
//            __m256d V_feqs_EB = DFL3 * V_rho * (1.0 + 4.5 * (+V_ux - V_uz) * (+V_ux - V_uz) - V_u2);
//            __m256d V_feqs_WT = V_feqs_EB;
//
//            __m256d V_feqa_C = _mm256_set1_pd(0.0);
//            __m256d
//                    V_feqa_N = DFL2 * V_rho * 3.0 * (+V_uy);
//            __m256d
//                    V_feqa_E = DFL2 * V_rho * 3.0 * (+V_ux);
//            __m256d
//                    V_feqa_T = DFL2 * V_rho * 3.0 * (+V_uz);
//            __m256d
//                    V_feqa_NE = DFL3 * V_rho * 3.0 * (+V_ux + V_uy);
//            __m256d
//                    V_feqa_NW = DFL3 * V_rho * 3.0 * (-V_ux + V_uy);
//            __m256d
//                    V_feqa_NT = DFL3 * V_rho * 3.0 * (+V_uy + V_uz);
//            __m256d
//                    V_feqa_NB = DFL3 * V_rho * 3.0 * (+V_uy - V_uz);
//            __m256d
//                    V_feqa_ET = DFL3 * V_rho * 3.0 * (+V_ux + V_uz);
//            __m256d
//                    V_feqa_EB = DFL3 * V_rho * 3.0 * (+V_ux - V_uz);
//
//            __m256d V_feqa_S = -V_feqa_N;
//            __m256d V_feqa_W = -V_feqa_E;
//            __m256d V_feqa_B = -V_feqa_T;
//            __m256d V_feqa_SW = -V_feqa_NE;
//            __m256d V_feqa_SE = -V_feqa_NW;
//            __m256d V_feqa_SB = -V_feqa_NT;
//            __m256d V_feqa_ST = -V_feqa_NB;
//            __m256d V_feqa_WB = -V_feqa_ET;
//            __m256d V_feqa_WT = -V_feqa_EB;
//
//            __m256d V_fs_C = V_SRC_C;
//            __m256d V_fs_N = 0.5 * (V_SRC_N + V_SRC_S);
//            __m256d V_fs_E = 0.5 * (V_SRC_E + V_SRC_W);
//            __m256d V_fs_T = 0.5 * (V_SRC_T + V_SRC_B);
//            __m256d V_fs_NE = 0.5 * (V_SRC_NE + V_SRC_SW);
//            __m256d V_fs_NW = 0.5 * (V_SRC_NW + V_SRC_SE);
//            __m256d V_fs_NT = 0.5 * (V_SRC_NT + V_SRC_SB);
//            __m256d V_fs_NB = 0.5 * (V_SRC_NB + V_SRC_ST);
//            __m256d V_fs_ET = 0.5 * (V_SRC_ET + V_SRC_WB);
//            __m256d V_fs_EB = 0.5 * (V_SRC_EB + V_SRC_WT);
//
//            __m256d V_fs_S = V_fs_N;
//            __m256d V_fs_W = V_fs_E;
//            __m256d V_fs_B = V_fs_T;
//            __m256d V_fs_SW = V_fs_NE;
//            __m256d V_fs_SE = V_fs_NW;
//            __m256d V_fs_SB = V_fs_NT;
//            __m256d V_fs_ST = V_fs_NB;
//            __m256d V_fs_WB = V_fs_ET;
//            __m256d V_fs_WT = V_fs_EB;
//
//
//            __m256d V_fa_C = _mm256_set1_pd(0.0);
//            __m256d V_fa_N = 0.5 * (V_SRC_N - V_SRC_S);
//            __m256d V_fa_E = 0.5 * (V_SRC_E - V_SRC_W);
//            __m256d V_fa_T = 0.5 * (V_SRC_T - V_SRC_B);
//            __m256d V_fa_NE = 0.5 * (V_SRC_NE - V_SRC_SW);
//            __m256d V_fa_NW = 0.5 * (V_SRC_NW - V_SRC_SE);
//            __m256d V_fa_NT = 0.5 * (V_SRC_NT - V_SRC_SB);
//            __m256d V_fa_NB = 0.5 * (V_SRC_NB - V_SRC_ST);
//            __m256d V_fa_ET = 0.5 * (V_SRC_ET - V_SRC_WB);
//            __m256d V_fa_EB = 0.5 * (V_SRC_EB - V_SRC_WT);
//
//            __m256d V_fa_S = -V_fa_N;
//            __m256d V_fa_W = -V_fa_E;
//            __m256d V_fa_B = -V_fa_T;
//            __m256d V_fa_SW = -V_fa_NE;
//            __m256d V_fa_SE = -V_fa_NW;
//            __m256d V_fa_SB = -V_fa_NT;
//            __m256d V_fa_ST = -V_fa_NB;
//            __m256d V_fa_WB = -V_fa_ET;
//            __m256d V_fa_WT = -V_fa_EB;
//
//            __m256d V_DST_C = V_SRC_C - OMEGA * (V_fs_C - V_feqs_C);
//            __m256d V_DST_N = V_SRC_N - OMEGA * (V_fs_N - V_feqs_N) - lambda0 * (V_fa_N - V_feqa_N);
//            __m256d V_DST_S = V_SRC_S - OMEGA * (V_fs_S - V_feqs_S) - lambda0 * (V_fa_S - V_feqa_S);
//            __m256d V_DST_E = V_SRC_E - OMEGA * (V_fs_E - V_feqs_E) - lambda0 * (V_fa_E - V_feqa_E);
//            __m256d V_DST_W = V_SRC_W - OMEGA * (V_fs_W - V_feqs_W) - lambda0 * (V_fa_W - V_feqa_W);
//            __m256d V_DST_T = V_SRC_T - OMEGA * (V_fs_T - V_feqs_T) - lambda0 * (V_fa_T - V_feqa_T);
//            __m256d V_DST_B = V_SRC_B - OMEGA * (V_fs_B - V_feqs_B) - lambda0 * (V_fa_B - V_feqa_B);
//            __m256d V_DST_NE = V_SRC_NE - OMEGA * (V_fs_NE - V_feqs_NE) - lambda0 * (V_fa_NE - V_feqa_NE);
//            __m256d V_DST_NW = V_SRC_NW - OMEGA * (V_fs_NW - V_feqs_NW) - lambda0 * (V_fa_NW - V_feqa_NW);
//            __m256d V_DST_SE = V_SRC_SE - OMEGA * (V_fs_SE - V_feqs_SE) - lambda0 * (V_fa_SE - V_feqa_SE);
//            __m256d V_DST_SW = V_SRC_SW - OMEGA * (V_fs_SW - V_feqs_SW) - lambda0 * (V_fa_SW - V_feqa_SW);
//            __m256d V_DST_NT = V_SRC_NT - OMEGA * (V_fs_NT - V_feqs_NT) - lambda0 * (V_fa_NT - V_feqa_NT);
//            __m256d V_DST_NB = V_SRC_NB - OMEGA * (V_fs_NB - V_feqs_NB) - lambda0 * (V_fa_NB - V_feqa_NB);
//            __m256d V_DST_ST = V_SRC_ST - OMEGA * (V_fs_ST - V_feqs_ST) - lambda0 * (V_fa_ST - V_feqa_ST);
//            __m256d V_DST_SB = V_SRC_SB - OMEGA * (V_fs_SB - V_feqs_SB) - lambda0 * (V_fa_SB - V_feqa_SB);
//            __m256d V_DST_ET = V_SRC_ET - OMEGA * (V_fs_ET - V_feqs_ET) - lambda0 * (V_fa_ET - V_feqa_ET);
//            __m256d V_DST_EB = V_SRC_EB - OMEGA * (V_fs_EB - V_feqs_EB) - lambda0 * (V_fa_EB - V_feqa_EB);
//            __m256d V_DST_WT = V_SRC_WT - OMEGA * (V_fs_WT - V_feqs_WT) - lambda0 * (V_fa_WT - V_feqa_WT);
//            __m256d V_DST_WB = V_SRC_WB - OMEGA * (V_fs_WB - V_feqs_WB) - lambda0 * (V_fa_WB - V_feqa_WB);
//
//            DST_C(dstGrid) = V_DST_C[0];
//            DST_C_NEXT(dstGrid) = V_DST_C[1];
//            DST_C_NEXT2(dstGrid) = V_DST_C[2];
//            DST_C_NEXT3(dstGrid) = V_DST_C[3];
//
//            DST_N(dstGrid) = V_DST_N[0];
//            DST_N_NEXT(dstGrid) = V_DST_N[1];
//            DST_N_NEXT2(dstGrid) = V_DST_N[2];
//            DST_N_NEXT3(dstGrid) = V_DST_N[3];
//
//            DST_S(dstGrid) = V_DST_S[0];
//            DST_S_NEXT(dstGrid) = V_DST_S[1];
//            DST_S_NEXT2(dstGrid) = V_DST_S[2];
//            DST_S_NEXT3(dstGrid) = V_DST_S[3];
//            DST_E(dstGrid) = V_DST_E[0];
//            DST_E_NEXT(dstGrid) = V_DST_E[1];
//            DST_E_NEXT2(dstGrid) = V_DST_E[2];
//            DST_E_NEXT3(dstGrid) = V_DST_E[3];
//            DST_W(dstGrid) = V_DST_W[0];
//            DST_W_NEXT(dstGrid) = V_DST_W[1];
//            DST_W_NEXT2(dstGrid) = V_DST_W[2];
//            DST_W_NEXT3(dstGrid) = V_DST_W[3];
//            DST_T(dstGrid) = V_DST_T[0];
//            DST_T_NEXT(dstGrid) = V_DST_T[1];
//            DST_T_NEXT2(dstGrid) = V_DST_T[2];
//            DST_T_NEXT3(dstGrid) = V_DST_T[3];
//            DST_B(dstGrid) = V_DST_B[0];
//            DST_B_NEXT(dstGrid) = V_DST_B[1];
//            DST_B_NEXT2(dstGrid) = V_DST_B[2];
//            DST_B_NEXT3(dstGrid) = V_DST_B[3];
//            DST_NE(dstGrid) = V_DST_NE[0];
//            DST_NE_NEXT(dstGrid) = V_DST_NE[1];
//            DST_NE_NEXT2(dstGrid) = V_DST_NE[2];
//            DST_NE_NEXT3(dstGrid) = V_DST_NE[3];
//            DST_NW(dstGrid) = V_DST_NW[0];
//            DST_NW_NEXT(dstGrid) = V_DST_NW[1];
//            DST_NW_NEXT2(dstGrid) = V_DST_NW[2];
//            DST_NW_NEXT3(dstGrid) = V_DST_NW[3];
//            DST_SE(dstGrid) = V_DST_SE[0];
//            DST_SE_NEXT(dstGrid) = V_DST_SE[1];
//            DST_SE_NEXT2(dstGrid) = V_DST_SE[2];
//            DST_SE_NEXT3(dstGrid) = V_DST_SE[3];
//            DST_SW(dstGrid) = V_DST_SW[0];
//            DST_SW_NEXT(dstGrid) = V_DST_SW[1];
//            DST_SW_NEXT2(dstGrid) = V_DST_SW[2];
//            DST_SW_NEXT3(dstGrid) = V_DST_SW[3];
//            DST_NT(dstGrid) = V_DST_NT[0];
//            DST_NT_NEXT(dstGrid) = V_DST_NT[1];
//            DST_NT_NEXT2(dstGrid) = V_DST_NT[2];
//            DST_NT_NEXT3(dstGrid) = V_DST_NT[3];
//            DST_NB(dstGrid) = V_DST_NB[0];
//            DST_NB_NEXT(dstGrid) = V_DST_NB[1];
//            DST_NB_NEXT2(dstGrid) = V_DST_NB[2];
//            DST_NB_NEXT3(dstGrid) = V_DST_NB[3];
//            DST_ST(dstGrid) = V_DST_ST[0];
//            DST_ST_NEXT(dstGrid) = V_DST_ST[1];
//            DST_ST_NEXT2(dstGrid) = V_DST_ST[2];
//            DST_ST_NEXT3(dstGrid) = V_DST_ST[3];
//            DST_SB(dstGrid) = V_DST_SB[0];
//            DST_SB_NEXT(dstGrid) = V_DST_SB[1];
//            DST_SB_NEXT2(dstGrid) = V_DST_SB[2];
//            DST_SB_NEXT3(dstGrid) = V_DST_SB[3];
//            DST_ET(dstGrid) = V_DST_ET[0];
//            DST_ET_NEXT(dstGrid) = V_DST_ET[1];
//            DST_ET_NEXT2(dstGrid) = V_DST_ET[2];
//            DST_ET_NEXT3(dstGrid) = V_DST_ET[3];
//            DST_EB(dstGrid) = V_DST_EB[0];
//            DST_EB_NEXT(dstGrid) = V_DST_EB[1];
//            DST_EB_NEXT2(dstGrid) = V_DST_EB[2];
//            DST_EB_NEXT3(dstGrid) = V_DST_EB[3];
//            DST_WT(dstGrid) = V_DST_WT[0];
//            DST_WT_NEXT(dstGrid) = V_DST_WT[1];
//            DST_WT_NEXT2(dstGrid) = V_DST_WT[2];
//            DST_WT_NEXT3(dstGrid) = V_DST_WT[3];
//            DST_WB(dstGrid) = V_DST_WB[0];
//            DST_WB_NEXT(dstGrid) = V_DST_WB[1];
//            DST_WB_NEXT2(dstGrid) = V_DST_WB[2];
//            DST_WB_NEXT3(dstGrid) = V_DST_WB[3];
//            i += 4 * N_CELL_ENTRIES;
//            continue;
//        }
//        int start = i;
//        for (int count_=0; count_ < 4; count_++ ) {
//            i = start + (count_ * N_CELL_ENTRIES);
//            if (TEST_FLAG_SWEEP(srcGrid, OBSTACLE)) {
//                printf("Debug i1:%d\n", i);
//                DST_C (dstGrid) = SRC_C (srcGrid);
//                DST_S (dstGrid) = SRC_N (srcGrid);
//                DST_N (dstGrid) = SRC_S (srcGrid);
//                DST_W (dstGrid) = SRC_E (srcGrid);
//                DST_E (dstGrid) = SRC_W (srcGrid);
//                DST_B (dstGrid) = SRC_T (srcGrid);
//                DST_T (dstGrid) = SRC_B (srcGrid);
//                DST_SW(dstGrid) = SRC_NE(srcGrid);
//                DST_SE(dstGrid) = SRC_NW(srcGrid);
//                DST_NW(dstGrid) = SRC_SE(srcGrid);
//                DST_NE(dstGrid) = SRC_SW(srcGrid);
//                DST_SB(dstGrid) = SRC_NT(srcGrid);
//                DST_ST(dstGrid) = SRC_NB(srcGrid);
//                DST_NB(dstGrid) = SRC_ST(srcGrid);
//                DST_NT(dstGrid) = SRC_SB(srcGrid);
//                DST_WB(dstGrid) = SRC_ET(srcGrid);
//                DST_WT(dstGrid) = SRC_EB(srcGrid);
//                DST_EB(dstGrid) = SRC_WT(srcGrid);
//                DST_ET(dstGrid) = SRC_WB(srcGrid);
//                continue;
//            }
//            printf("Debug i2:%d\n", i);
//            rho = +SRC_C (srcGrid) + SRC_N (srcGrid)
//                  + SRC_S (srcGrid) + SRC_E (srcGrid)
//                  + SRC_W (srcGrid) + SRC_T (srcGrid)
//                  + SRC_B (srcGrid) + SRC_NE(srcGrid)
//                  + SRC_NW(srcGrid) + SRC_SE(srcGrid)
//                  + SRC_SW(srcGrid) + SRC_NT(srcGrid)
//                  + SRC_NB(srcGrid) + SRC_ST(srcGrid)
//                  + SRC_SB(srcGrid) + SRC_ET(srcGrid)
//                  + SRC_EB(srcGrid) + SRC_WT(srcGrid)
//                  + SRC_WB(srcGrid);
//
//            ux = +SRC_E (srcGrid) - SRC_W (srcGrid)
//                 + SRC_NE(srcGrid) - SRC_NW(srcGrid)
//                 + SRC_SE(srcGrid) - SRC_SW(srcGrid)
//                 + SRC_ET(srcGrid) + SRC_EB(srcGrid)
//                 - SRC_WT(srcGrid) - SRC_WB(srcGrid);
//            uy = +SRC_N (srcGrid) - SRC_S (srcGrid)
//                 + SRC_NE(srcGrid) + SRC_NW(srcGrid)
//                 - SRC_SE(srcGrid) - SRC_SW(srcGrid)
//                 + SRC_NT(srcGrid) + SRC_NB(srcGrid)
//                 - SRC_ST(srcGrid) - SRC_SB(srcGrid);
//            uz = +SRC_T (srcGrid) - SRC_B (srcGrid)
//                 + SRC_NT(srcGrid) - SRC_NB(srcGrid)
//                 + SRC_ST(srcGrid) - SRC_SB(srcGrid)
//                 + SRC_ET(srcGrid) - SRC_EB(srcGrid)
//                 + SRC_WT(srcGrid) - SRC_WB(srcGrid);
//
//            ux /= rho;
//            uy /= rho;
//            uz /= rho;
//
//            if (TEST_FLAG_SWEEP(srcGrid, ACCEL)) {
//                ux = 0.005;
//                uy = 0.002;
//                uz = 0.000;
//            }
//
//
//            u2 = 1.5 * (ux * ux + uy * uy + uz * uz);
//
//            feqs[C] = DFL1 * rho * (1.0 - u2);
//            feqs[N] = feqs[S] = DFL2 * rho * (1.0 + 4.5 * (+uy) * (+uy) - u2);
//            feqs[E] = feqs[W] = DFL2 * rho * (1.0 + 4.5 * (+ux) * (+ux) - u2);
//            feqs[T] = feqs[B] = DFL2 * rho * (1.0 + 4.5 * (+uz) * (+uz) - u2);
//            feqs[NE] = feqs[SW] = DFL3 * rho * (1.0 + 4.5 * (+ux + uy) * (+ux + uy) - u2);
//            feqs[NW] = feqs[SE] = DFL3 * rho * (1.0 + 4.5 * (-ux + uy) * (-ux + uy) - u2);
//            feqs[NT] = feqs[SB] = DFL3 * rho * (1.0 + 4.5 * (+uy + uz) * (+uy + uz) - u2);
//            feqs[NB] = feqs[ST] = DFL3 * rho * (1.0 + 4.5 * (+uy - uz) * (+uy - uz) - u2);
//            feqs[ET] = feqs[WB] = DFL3 * rho * (1.0 + 4.5 * (+ux + uz) * (+ux + uz) - u2);
//            feqs[EB] = feqs[WT] = DFL3 * rho * (1.0 + 4.5 * (+ux - uz) * (+ux - uz) - u2);
//
//            feqa[C] = 0.0;
//            feqa[S] = -(feqa[N] = DFL2 * rho * 3.0 * (+uy));
//            feqa[W] = -(feqa[E] = DFL2 * rho * 3.0 * (+ux));
//            feqa[B] = -(feqa[T] = DFL2 * rho * 3.0 * (+uz));
//            feqa[SW] = -(feqa[NE] = DFL3 * rho * 3.0 * (+ux + uy));
//            feqa[SE] = -(feqa[NW] = DFL3 * rho * 3.0 * (-ux + uy));
//            feqa[SB] = -(feqa[NT] = DFL3 * rho * 3.0 * (+uy + uz));
//            feqa[ST] = -(feqa[NB] = DFL3 * rho * 3.0 * (+uy - uz));
//            feqa[WB] = -(feqa[ET] = DFL3 * rho * 3.0 * (+ux + uz));
//            feqa[WT] = -(feqa[EB] = DFL3 * rho * 3.0 * (+ux - uz));
//
//            fs[C] = SRC_C (srcGrid);
//            fs[N] = fs[S] = 0.5 * (SRC_N (srcGrid) + SRC_S (srcGrid));
//            fs[E] = fs[W] = 0.5 * (SRC_E (srcGrid) + SRC_W (srcGrid));
//            fs[T] = fs[B] = 0.5 * (SRC_T (srcGrid) + SRC_B (srcGrid));
//            fs[NE] = fs[SW] = 0.5 * (SRC_NE(srcGrid) + SRC_SW(srcGrid));
//            fs[NW] = fs[SE] = 0.5 * (SRC_NW(srcGrid) + SRC_SE(srcGrid));
//            fs[NT] = fs[SB] = 0.5 * (SRC_NT(srcGrid) + SRC_SB(srcGrid));
//            fs[NB] = fs[ST] = 0.5 * (SRC_NB(srcGrid) + SRC_ST(srcGrid));
//            fs[ET] = fs[WB] = 0.5 * (SRC_ET(srcGrid) + SRC_WB(srcGrid));
//            fs[EB] = fs[WT] = 0.5 * (SRC_EB(srcGrid) + SRC_WT(srcGrid));
//
//            fa[C] = 0.0;
//            fa[S] = -(fa[N] = 0.5 * (SRC_N (srcGrid) - SRC_S (srcGrid)));
//            fa[W] = -(fa[E] = 0.5 * (SRC_E (srcGrid) - SRC_W (srcGrid)));
//            fa[B] = -(fa[T] = 0.5 * (SRC_T (srcGrid) - SRC_B (srcGrid)));
//            fa[SW] = -(fa[NE] = 0.5 * (SRC_NE(srcGrid) - SRC_SW(srcGrid)));
//            fa[SE] = -(fa[NW] = 0.5 * (SRC_NW(srcGrid) - SRC_SE(srcGrid)));
//            fa[SB] = -(fa[NT] = 0.5 * (SRC_NT(srcGrid) - SRC_SB(srcGrid)));
//            fa[ST] = -(fa[NB] = 0.5 * (SRC_NB(srcGrid) - SRC_ST(srcGrid)));
//            fa[WB] = -(fa[ET] = 0.5 * (SRC_ET(srcGrid) - SRC_WB(srcGrid)));
//            fa[WT] = -(fa[EB] = 0.5 * (SRC_EB(srcGrid) - SRC_WT(srcGrid)));
//
//            DST_C (dstGrid) = SRC_C (srcGrid) - OMEGA * (fs[C] - feqs[C]);
//            DST_N (dstGrid) = SRC_N (srcGrid) - OMEGA * (fs[N] - feqs[N]) - lambda0 * (fa[N] - feqa[N]);
//            DST_S (dstGrid) = SRC_S (srcGrid) - OMEGA * (fs[S] - feqs[S]) - lambda0 * (fa[S] - feqa[S]);
//            DST_E (dstGrid) = SRC_E (srcGrid) - OMEGA * (fs[E] - feqs[E]) - lambda0 * (fa[E] - feqa[E]);
//            DST_W (dstGrid) = SRC_W (srcGrid) - OMEGA * (fs[W] - feqs[W]) - lambda0 * (fa[W] - feqa[W]);
//            DST_T (dstGrid) = SRC_T (srcGrid) - OMEGA * (fs[T] - feqs[T]) - lambda0 * (fa[T] - feqa[T]);
//            DST_B (dstGrid) = SRC_B (srcGrid) - OMEGA * (fs[B] - feqs[B]) - lambda0 * (fa[B] - feqa[B]);
//            DST_NE(dstGrid) = SRC_NE(srcGrid) - OMEGA * (fs[NE] - feqs[NE]) - lambda0 * (fa[NE] - feqa[NE]);
//            DST_NW(dstGrid) = SRC_NW(srcGrid) - OMEGA * (fs[NW] - feqs[NW]) - lambda0 * (fa[NW] - feqa[NW]);
//            DST_SE(dstGrid) = SRC_SE(srcGrid) - OMEGA * (fs[SE] - feqs[SE]) - lambda0 * (fa[SE] - feqa[SE]);
//            DST_SW(dstGrid) = SRC_SW(srcGrid) - OMEGA * (fs[SW] - feqs[SW]) - lambda0 * (fa[SW] - feqa[SW]);
//            DST_NT(dstGrid) = SRC_NT(srcGrid) - OMEGA * (fs[NT] - feqs[NT]) - lambda0 * (fa[NT] - feqa[NT]);
//            DST_NB(dstGrid) = SRC_NB(srcGrid) - OMEGA * (fs[NB] - feqs[NB]) - lambda0 * (fa[NB] - feqa[NB]);
//            DST_ST(dstGrid) = SRC_ST(srcGrid) - OMEGA * (fs[ST] - feqs[ST]) - lambda0 * (fa[ST] - feqa[ST]);
//            DST_SB(dstGrid) = SRC_SB(srcGrid) - OMEGA * (fs[SB] - feqs[SB]) - lambda0 * (fa[SB] - feqa[SB]);
//            DST_ET(dstGrid) = SRC_ET(srcGrid) - OMEGA * (fs[ET] - feqs[ET]) - lambda0 * (fa[ET] - feqa[ET]);
//            DST_EB(dstGrid) = SRC_EB(srcGrid) - OMEGA * (fs[EB] - feqs[EB]) - lambda0 * (fa[EB] - feqa[EB]);
//            DST_WT(dstGrid) = SRC_WT(srcGrid) - OMEGA * (fs[WT] - feqs[WT]) - lambda0 * (fa[WT] - feqa[WT]);
//            DST_WB(dstGrid) = SRC_WB(srcGrid) - OMEGA * (fs[WB] - feqs[WB]) - lambda0 * (fa[WB] - feqa[WB]);
//        }
//        i+=N_CELL_ENTRIES;
//    SWEEP_END
//}



void LBM_performStreamCollideTRT( LBM_Grid srcGrid, LBM_Grid dstGrid ) {
	SWEEP_VAR

	double ux, uy, uz, u2, rho;
    int count_tt=0; int count_ff=0; int count=0;

	const double lambda0 = 1.0/(0.5+3.0/(16.0*(1.0/OMEGA-0.5)));
	double fs[N_CELL_ENTRIES], fa[N_CELL_ENTRIES],
		     feqs[N_CELL_ENTRIES], feqa[N_CELL_ENTRIES];
//    SWEEP_2START(0, 0, 0, 0, 0, SIZE_Z )
//        ++count;
//        if(TEST_FLAG_SWEEP(srcGrid, OBSTACLE) && TEST_FLAG_SWEEP_NEXT(srcGrid, OBSTACLE))
//        {
//            ++count_tt;
//            continue;
//        }
//
//        if(!TEST_FLAG_SWEEP_NEXT(srcGrid, OBSTACLE) && !TEST_FLAG_SWEEP(srcGrid, OBSTACLE))
//        {
//            ++count_ff;
//            continue;
//        }
//    SWEEP_END

//    SWEEP_4START(0, 0, 0, 0, 0, SIZE_Z )
//        ++count;
//        if (TEST_FLAG_SWEEP(srcGrid, OBSTACLE) && TEST_FLAG_SWEEP_NEXT(srcGrid, OBSTACLE) &&
//            TEST_FLAG_SWEEP_NEXT2(srcGrid, OBSTACLE) &&
//            TEST_FLAG_SWEEP_NEXT3(srcGrid, OBSTACLE)) {
//            ++count_tt;
//            continue;
//        }
//
//        if (!TEST_FLAG_SWEEP_NEXT(srcGrid, OBSTACLE) && !TEST_FLAG_SWEEP(srcGrid, OBSTACLE) &&
//            !TEST_FLAG_SWEEP_NEXT2(srcGrid, OBSTACLE) &&
//            !TEST_FLAG_SWEEP_NEXT3(srcGrid, OBSTACLE)) {
//            ++count_ff;
//            continue;
//        }
//    SWEEP_END


//    printf("tt ratio=%lf %d\t ff ratio=%lf %d\n", count_tt * 1.0/count,  count_tt, count_ff*1.0/count, count_ff);
	/*voption indep*/
#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
#pragma omp parallel for private( ux, uy, uz, u2, rho, fs, fa, feqs, feqa )
#endif
    SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
    //printf("Debug i3:%d\n", i);
		if( TEST_FLAG_SWEEP( srcGrid, OBSTACLE )) {
            //printf("Debug i1:%d\n", i);
			DST_C ( dstGrid ) = SRC_C ( srcGrid );
			DST_S ( dstGrid ) = SRC_N ( srcGrid );
			DST_N ( dstGrid ) = SRC_S ( srcGrid );
			DST_W ( dstGrid ) = SRC_E ( srcGrid );
			DST_E ( dstGrid ) = SRC_W ( srcGrid );
			DST_B ( dstGrid ) = SRC_T ( srcGrid );
			DST_T ( dstGrid ) = SRC_B ( srcGrid );
			DST_SW( dstGrid ) = SRC_NE( srcGrid );
			DST_SE( dstGrid ) = SRC_NW( srcGrid );
			DST_NW( dstGrid ) = SRC_SE( srcGrid );
			DST_NE( dstGrid ) = SRC_SW( srcGrid );
			DST_SB( dstGrid ) = SRC_NT( srcGrid );
			DST_ST( dstGrid ) = SRC_NB( srcGrid );
			DST_NB( dstGrid ) = SRC_ST( srcGrid );
			DST_NT( dstGrid ) = SRC_SB( srcGrid );
			DST_WB( dstGrid ) = SRC_ET( srcGrid );
			DST_WT( dstGrid ) = SRC_EB( srcGrid );
			DST_EB( dstGrid ) = SRC_WT( srcGrid );
			DST_ET( dstGrid ) = SRC_WB( srcGrid );
			continue;
		}
        //printf("Debug i2:%d\n", i);
		rho = + SRC_C ( srcGrid ) + SRC_N ( srcGrid )
		      + SRC_S ( srcGrid ) + SRC_E ( srcGrid )
		      + SRC_W ( srcGrid ) + SRC_T ( srcGrid )
		      + SRC_B ( srcGrid ) + SRC_NE( srcGrid )
		      + SRC_NW( srcGrid ) + SRC_SE( srcGrid )
		      + SRC_SW( srcGrid ) + SRC_NT( srcGrid )
		      + SRC_NB( srcGrid ) + SRC_ST( srcGrid )
		      + SRC_SB( srcGrid ) + SRC_ET( srcGrid )
		      + SRC_EB( srcGrid ) + SRC_WT( srcGrid )
		      + SRC_WB( srcGrid );

		ux = + SRC_E ( srcGrid ) - SRC_W ( srcGrid )
		     + SRC_NE( srcGrid ) - SRC_NW( srcGrid )
		     + SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_ET( srcGrid ) + SRC_EB( srcGrid )
		     - SRC_WT( srcGrid ) - SRC_WB( srcGrid );
		uy = + SRC_N ( srcGrid ) - SRC_S ( srcGrid )
		     + SRC_NE( srcGrid ) + SRC_NW( srcGrid )
		     - SRC_SE( srcGrid ) - SRC_SW( srcGrid )
		     + SRC_NT( srcGrid ) + SRC_NB( srcGrid )
		     - SRC_ST( srcGrid ) - SRC_SB( srcGrid );
		uz = + SRC_T ( srcGrid ) - SRC_B ( srcGrid )
		     + SRC_NT( srcGrid ) - SRC_NB( srcGrid )
		     + SRC_ST( srcGrid ) - SRC_SB( srcGrid )
		     + SRC_ET( srcGrid ) - SRC_EB( srcGrid )
		     + SRC_WT( srcGrid ) - SRC_WB( srcGrid );

		ux /= rho;
		uy /= rho;
		uz /= rho;

		if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
			ux = 0.005;
			uy = 0.002;
			uz = 0.000;
		}

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		feqs[C ] =            DFL1*rho*(1.0                         - u2);
		feqs[N ] = feqs[S ] = DFL2*rho*(1.0 + 4.5*(+uy   )*(+uy   ) - u2);
		feqs[E ] = feqs[W ] = DFL2*rho*(1.0 + 4.5*(+ux   )*(+ux   ) - u2);
		feqs[T ] = feqs[B ] = DFL2*rho*(1.0 + 4.5*(+uz   )*(+uz   ) - u2);
		feqs[NE] = feqs[SW] = DFL3*rho*(1.0 + 4.5*(+ux+uy)*(+ux+uy) - u2);
		feqs[NW] = feqs[SE] = DFL3*rho*(1.0 + 4.5*(-ux+uy)*(-ux+uy) - u2);
		feqs[NT] = feqs[SB] = DFL3*rho*(1.0 + 4.5*(+uy+uz)*(+uy+uz) - u2);
		feqs[NB] = feqs[ST] = DFL3*rho*(1.0 + 4.5*(+uy-uz)*(+uy-uz) - u2);
		feqs[ET] = feqs[WB] = DFL3*rho*(1.0 + 4.5*(+ux+uz)*(+ux+uz) - u2);
		feqs[EB] = feqs[WT] = DFL3*rho*(1.0 + 4.5*(+ux-uz)*(+ux-uz) - u2);

		feqa[C ] = 0.0;
		feqa[S ] = - (feqa[N ] = DFL2*rho*3.0*(+uy   ));
		feqa[W ] = - (feqa[E ] = DFL2*rho*3.0*(+ux   ));
		feqa[B ] = - (feqa[T ] = DFL2*rho*3.0*(+uz   ));
		feqa[SW] = - (feqa[NE] = DFL3*rho*3.0*(+ux+uy));
		feqa[SE] = - (feqa[NW] = DFL3*rho*3.0*(-ux+uy));
		feqa[SB] = - (feqa[NT] = DFL3*rho*3.0*(+uy+uz));
		feqa[ST] = - (feqa[NB] = DFL3*rho*3.0*(+uy-uz));
		feqa[WB] = - (feqa[ET] = DFL3*rho*3.0*(+ux+uz));
		feqa[WT] = - (feqa[EB] = DFL3*rho*3.0*(+ux-uz));

		fs[C ] =                 SRC_C ( srcGrid );
		fs[N ] = fs[S ] = 0.5 * (SRC_N ( srcGrid ) + SRC_S ( srcGrid ));
		fs[E ] = fs[W ] = 0.5 * (SRC_E ( srcGrid ) + SRC_W ( srcGrid ));
		fs[T ] = fs[B ] = 0.5 * (SRC_T ( srcGrid ) + SRC_B ( srcGrid ));
		fs[NE] = fs[SW] = 0.5 * (SRC_NE( srcGrid ) + SRC_SW( srcGrid ));
		fs[NW] = fs[SE] = 0.5 * (SRC_NW( srcGrid ) + SRC_SE( srcGrid ));
		fs[NT] = fs[SB] = 0.5 * (SRC_NT( srcGrid ) + SRC_SB( srcGrid ));
		fs[NB] = fs[ST] = 0.5 * (SRC_NB( srcGrid ) + SRC_ST( srcGrid ));
		fs[ET] = fs[WB] = 0.5 * (SRC_ET( srcGrid ) + SRC_WB( srcGrid ));
		fs[EB] = fs[WT] = 0.5 * (SRC_EB( srcGrid ) + SRC_WT( srcGrid ));

		fa[C ] = 0.0;
		fa[S ] = - (fa[N ] = 0.5 * (SRC_N ( srcGrid ) - SRC_S ( srcGrid )));
		fa[W ] = - (fa[E ] = 0.5 * (SRC_E ( srcGrid ) - SRC_W ( srcGrid )));
		fa[B ] = - (fa[T ] = 0.5 * (SRC_T ( srcGrid ) - SRC_B ( srcGrid )));
		fa[SW] = - (fa[NE] = 0.5 * (SRC_NE( srcGrid ) - SRC_SW( srcGrid )));
		fa[SE] = - (fa[NW] = 0.5 * (SRC_NW( srcGrid ) - SRC_SE( srcGrid )));
		fa[SB] = - (fa[NT] = 0.5 * (SRC_NT( srcGrid ) - SRC_SB( srcGrid )));
		fa[ST] = - (fa[NB] = 0.5 * (SRC_NB( srcGrid ) - SRC_ST( srcGrid )));
		fa[WB] = - (fa[ET] = 0.5 * (SRC_ET( srcGrid ) - SRC_WB( srcGrid )));
		fa[WT] = - (fa[EB] = 0.5 * (SRC_EB( srcGrid ) - SRC_WT( srcGrid )));

		DST_C ( dstGrid ) = SRC_C ( srcGrid ) - OMEGA * (fs[C ] - feqs[C ])                                ;
		DST_N ( dstGrid ) = SRC_N ( srcGrid ) - OMEGA * (fs[N ] - feqs[N ]) - lambda0 * (fa[N ] - feqa[N ]);
		DST_S ( dstGrid ) = SRC_S ( srcGrid ) - OMEGA * (fs[S ] - feqs[S ]) - lambda0 * (fa[S ] - feqa[S ]);
		DST_E ( dstGrid ) = SRC_E ( srcGrid ) - OMEGA * (fs[E ] - feqs[E ]) - lambda0 * (fa[E ] - feqa[E ]);
		DST_W ( dstGrid ) = SRC_W ( srcGrid ) - OMEGA * (fs[W ] - feqs[W ]) - lambda0 * (fa[W ] - feqa[W ]);
		DST_T ( dstGrid ) = SRC_T ( srcGrid ) - OMEGA * (fs[T ] - feqs[T ]) - lambda0 * (fa[T ] - feqa[T ]);
		DST_B ( dstGrid ) = SRC_B ( srcGrid ) - OMEGA * (fs[B ] - feqs[B ]) - lambda0 * (fa[B ] - feqa[B ]);
		DST_NE( dstGrid ) = SRC_NE( srcGrid ) - OMEGA * (fs[NE] - feqs[NE]) - lambda0 * (fa[NE] - feqa[NE]);
		DST_NW( dstGrid ) = SRC_NW( srcGrid ) - OMEGA * (fs[NW] - feqs[NW]) - lambda0 * (fa[NW] - feqa[NW]);
		DST_SE( dstGrid ) = SRC_SE( srcGrid ) - OMEGA * (fs[SE] - feqs[SE]) - lambda0 * (fa[SE] - feqa[SE]);
		DST_SW( dstGrid ) = SRC_SW( srcGrid ) - OMEGA * (fs[SW] - feqs[SW]) - lambda0 * (fa[SW] - feqa[SW]);
		DST_NT( dstGrid ) = SRC_NT( srcGrid ) - OMEGA * (fs[NT] - feqs[NT]) - lambda0 * (fa[NT] - feqa[NT]);
		DST_NB( dstGrid ) = SRC_NB( srcGrid ) - OMEGA * (fs[NB] - feqs[NB]) - lambda0 * (fa[NB] - feqa[NB]);
		DST_ST( dstGrid ) = SRC_ST( srcGrid ) - OMEGA * (fs[ST] - feqs[ST]) - lambda0 * (fa[ST] - feqa[ST]);
		DST_SB( dstGrid ) = SRC_SB( srcGrid ) - OMEGA * (fs[SB] - feqs[SB]) - lambda0 * (fa[SB] - feqa[SB]);
		DST_ET( dstGrid ) = SRC_ET( srcGrid ) - OMEGA * (fs[ET] - feqs[ET]) - lambda0 * (fa[ET] - feqa[ET]);
		DST_EB( dstGrid ) = SRC_EB( srcGrid ) - OMEGA * (fs[EB] - feqs[EB]) - lambda0 * (fa[EB] - feqa[EB]);
		DST_WT( dstGrid ) = SRC_WT( srcGrid ) - OMEGA * (fs[WT] - feqs[WT]) - lambda0 * (fa[WT] - feqa[WT]);
		DST_WB( dstGrid ) = SRC_WB( srcGrid ) - OMEGA * (fs[WB] - feqs[WB]) - lambda0 * (fa[WB] - feqa[WB]);
	SWEEP_END
}

void LBM_performStreamCollideTRT_Vec_RC( LBM_Grid srcGrid, LBM_Grid dstGrid) {
    SWEEP_VAR

    double ux, uy, uz, u2, rho;
    int count_tt=0; int count_ff=0; int count=0;

    const double lambda0 = 1.0/(0.5+3.0/(16.0*(1.0/OMEGA-0.5)));
    double fs[N_CELL_ENTRIES], fa[N_CELL_ENTRIES],
            feqs[N_CELL_ENTRIES], feqa[N_CELL_ENTRIES];



//    printf("tt ratio=%lf %d\t ff ratio=%lf %d\n", count_tt * 1.0/count,  count_tt, count_ff*1.0/count, count_ff);
    /*voption indep*/
#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
#pragma omp parallel for private( ux, uy, uz, u2, rho, fs, fa, feqs, feqa )
#endif
    SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
        //printf("Debug i3:%d\n", i);
        if( !TEST_FLAG_SWEEP( srcGrid, OBSTACLE ) && !TEST_FLAG_SWEEP_NEXT(srcGrid, OBSTACLE) && !TEST_FLAG_SWEEP_NEXT2(srcGrid, OBSTACLE) && !TEST_FLAG_SWEEP_NEXT3(srcGrid, OBSTACLE)){

            __m256d V_SRC_C = _mm256_set_pd(SRC_C_NEXT3(srcGrid), SRC_C_NEXT2(srcGrid), SRC_C_NEXT(srcGrid), SRC_C(srcGrid));
            __m256d V_SRC_N = _mm256_set_pd(SRC_N_NEXT3(srcGrid), SRC_N_NEXT2(srcGrid), SRC_N_NEXT(srcGrid), SRC_N(srcGrid));
            __m256d V_SRC_S = _mm256_set_pd(SRC_S_NEXT3(srcGrid), SRC_S_NEXT2(srcGrid), SRC_S_NEXT(srcGrid), SRC_S(srcGrid));
            __m256d V_SRC_E = _mm256_set_pd(SRC_E_NEXT3(srcGrid), SRC_E_NEXT2(srcGrid), SRC_E_NEXT(srcGrid), SRC_E(srcGrid));
            __m256d V_SRC_W = _mm256_set_pd(SRC_W_NEXT3(srcGrid), SRC_W_NEXT2(srcGrid), SRC_W_NEXT(srcGrid), SRC_W(srcGrid));
            __m256d V_SRC_T = _mm256_set_pd(SRC_T_NEXT3(srcGrid), SRC_T_NEXT2(srcGrid), SRC_T_NEXT(srcGrid), SRC_T(srcGrid));
            __m256d V_SRC_B = _mm256_set_pd(SRC_B_NEXT3(srcGrid), SRC_B_NEXT2(srcGrid), SRC_B_NEXT(srcGrid), SRC_B(srcGrid));
            __m256d V_SRC_NE = _mm256_set_pd(SRC_NE_NEXT3(srcGrid), SRC_NE_NEXT2(srcGrid), SRC_NE_NEXT(srcGrid), SRC_NE(srcGrid));

            __m256d V_SRC_NW = _mm256_set_pd(SRC_NW_NEXT3(srcGrid), SRC_NW_NEXT2(srcGrid), SRC_NW_NEXT(srcGrid), SRC_NW(srcGrid));
            __m256d V_SRC_SE = _mm256_set_pd(SRC_SE_NEXT3(srcGrid), SRC_SE_NEXT2(srcGrid), SRC_SE_NEXT(srcGrid), SRC_SE(srcGrid));
            __m256d V_SRC_SW = _mm256_set_pd(SRC_SW_NEXT3(srcGrid), SRC_SW_NEXT2(srcGrid), SRC_SW_NEXT(srcGrid), SRC_SW(srcGrid));
            __m256d V_SRC_NT = _mm256_set_pd(SRC_NT_NEXT3(srcGrid), SRC_NT_NEXT2(srcGrid), SRC_NT_NEXT(srcGrid), SRC_NT(srcGrid));
            __m256d V_SRC_NB = _mm256_set_pd(SRC_NB_NEXT3(srcGrid), SRC_NB_NEXT2(srcGrid), SRC_NB_NEXT(srcGrid), SRC_NB(srcGrid));
            __m256d V_SRC_ST = _mm256_set_pd(SRC_ST_NEXT3(srcGrid), SRC_ST_NEXT2(srcGrid), SRC_ST_NEXT(srcGrid), SRC_ST(srcGrid));
            __m256d V_SRC_SB = _mm256_set_pd(SRC_SB_NEXT3(srcGrid), SRC_SB_NEXT2(srcGrid), SRC_SB_NEXT(srcGrid), SRC_SB(srcGrid));
            __m256d V_SRC_ET = _mm256_set_pd(SRC_ET_NEXT3(srcGrid), SRC_ET_NEXT2(srcGrid), SRC_ET_NEXT(srcGrid), SRC_ET(srcGrid));
            __m256d V_SRC_EB = _mm256_set_pd(SRC_EB_NEXT3(srcGrid), SRC_EB_NEXT2(srcGrid), SRC_EB_NEXT(srcGrid), SRC_EB(srcGrid));
            __m256d V_SRC_WT = _mm256_set_pd(SRC_WT_NEXT3(srcGrid), SRC_WT_NEXT2(srcGrid), SRC_WT_NEXT(srcGrid), SRC_WT(srcGrid));
            __m256d V_SRC_WB = _mm256_set_pd(SRC_WB_NEXT3(srcGrid), SRC_WB_NEXT2(srcGrid), SRC_WB_NEXT(srcGrid), SRC_WB(srcGrid));

            __m256d V_rho = +V_SRC_C + V_SRC_N
                            + V_SRC_S + V_SRC_E
                            + V_SRC_W + V_SRC_T
                            + V_SRC_B + V_SRC_NE
                            + V_SRC_NW + V_SRC_SE
                            + V_SRC_SW + V_SRC_NT
                            + V_SRC_NB + V_SRC_ST
                            + V_SRC_SB + V_SRC_ET
                            + V_SRC_EB + V_SRC_WT
                            + V_SRC_WB;

            __m256d V_ux = +V_SRC_E - V_SRC_W
                           + V_SRC_NE - V_SRC_NW
                           + V_SRC_SE - V_SRC_SW
                           + V_SRC_ET + V_SRC_EB
                           - V_SRC_WT - V_SRC_WB;

            __m256d V_uy = +V_SRC_N - V_SRC_S
                           + V_SRC_NE + V_SRC_NW
                           - V_SRC_SE - V_SRC_SW
                           + V_SRC_NT + V_SRC_NB
                           - V_SRC_ST - V_SRC_SB;

            __m256d V_uz = +V_SRC_T - V_SRC_B
                           + V_SRC_NT - V_SRC_NB
                           + V_SRC_ST - V_SRC_SB
                           + V_SRC_ET - V_SRC_EB
                           + V_SRC_WT - V_SRC_WB;

            V_ux = V_ux / V_rho;
            V_uy = V_uy / V_rho;
            V_uz = V_uz / V_rho;

            if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
                V_ux[0] = 0.005;
                V_uy[0] = 0.002;
                V_uz[0] = 0.000;
            }
            if (TEST_FLAG_SWEEP_NEXT(srcGrid, ACCEL)) {
                V_ux[1] = 0.005;
                V_uy[1] = 0.002;
                V_uz[1] = 0.000;
            }
            if (TEST_FLAG_SWEEP_NEXT2(srcGrid, ACCEL)) {
                V_ux[2] = 0.005;
                V_uy[2] = 0.002;
                V_uz[2] = 0.000;
            }
            if (TEST_FLAG_SWEEP_NEXT3(srcGrid, ACCEL)) {
                V_ux[3] = 0.005;
                V_uy[3] = 0.002;
                V_uz[3] = 0.000;
            }

            __m256d V_u2 = 1.5 * (V_ux * V_ux + V_uy * V_uy + V_uz * V_uz);

            __m256d V_half = _mm256_set1_pd(0.5);
            __m256d V_one = _mm256_set1_pd(1.0);
            __m256d V_one_half = _mm256_set1_pd(1.5);
            __m256d V_three = _mm256_set1_pd(3.0);
            __m256d V_four_half = _mm256_set1_pd(4.5);
            __m256d V_DFL1 = _mm256_set1_pd(DFL1);
            __m256d V_DFL2 = _mm256_set1_pd(DFL2);
            __m256d V_DFL3 = _mm256_set1_pd(DFL3);
            __m256d V_OMEGA = _mm256_set1_pd(OMEGA);
            __m256d V_lambda0 = _mm256_set1_pd(lambda0);

            //u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

            __m256d V_feqs_C = V_DFL1 * V_rho * (V_one - V_u2);
            __m256d V_feqs_N = V_DFL2 * V_rho * (V_one + V_four_half * (+V_uy) * (+V_uy) - V_u2);
            __m256d V_feqs_S = V_feqs_N;
            __m256d V_feqs_E = V_DFL2 * V_rho * (V_one + V_four_half * (+V_ux) * (+V_ux) - V_u2);
            __m256d V_feqs_W = V_feqs_E;
            __m256d V_feqs_T = V_DFL2 * V_rho * (V_one + V_four_half * (+V_uz) * (+V_uz) - V_u2);
            __m256d V_feqs_B = V_feqs_T;
            __m256d V_feqs_NE = V_DFL3 * V_rho * (V_one + V_four_half * (+V_ux + V_uy) * (+V_ux + V_uy) - V_u2);
            __m256d V_feqs_SW = V_feqs_NE;
            __m256d V_feqs_NW = V_DFL3 * V_rho * (V_one + V_four_half * (-V_ux + V_uy) * (-V_ux + V_uy) - V_u2);
            __m256d V_feqs_SE = V_feqs_NW;
            __m256d V_feqs_NT = V_DFL3 * V_rho * (V_one + V_four_half * (+V_uy + V_uz) * (+V_uy + V_uz) - V_u2);
            __m256d V_feqs_SB = V_feqs_NT;
            __m256d V_feqs_NB = V_DFL3 * V_rho * (V_one + V_four_half * (+V_uy - V_uz) * (+V_uy - V_uz) - V_u2);
            __m256d V_feqs_ST = V_feqs_NB;
            __m256d V_feqs_ET = V_DFL3 * V_rho * (V_one + V_four_half * (+V_ux + V_uz) * (+V_ux + V_uz) - V_u2);
            __m256d V_feqs_WB = V_feqs_ET;
            __m256d V_feqs_EB = V_DFL3 * V_rho * (V_one + V_four_half * (+V_ux - V_uz) * (+V_ux - V_uz) - V_u2);
            __m256d V_feqs_WT = V_feqs_EB;


            __m256d V_feqa_C = _mm256_set1_pd(0.0);
            __m256d
                    V_feqa_N = V_DFL2 * V_rho * V_three * (+V_uy);
            __m256d
                    V_feqa_E = V_DFL2 * V_rho * V_three * (+V_ux);
            __m256d
                    V_feqa_T = V_DFL2 * V_rho * V_three * (+V_uz);
            __m256d
                    V_feqa_NE = V_DFL3 * V_rho * V_three * (+V_ux + V_uy);
            __m256d
                    V_feqa_NW = V_DFL3 * V_rho * V_three * (-V_ux + V_uy);
            __m256d
                    V_feqa_NT = V_DFL3 * V_rho * V_three * (+V_uy + V_uz);
            __m256d
                    V_feqa_NB = V_DFL3 * V_rho * V_three * (+V_uy - V_uz);
            __m256d
                    V_feqa_ET = V_DFL3 * V_rho * V_three * (+V_ux + V_uz);
            __m256d
                    V_feqa_EB = V_DFL3 * V_rho * V_three * (+V_ux - V_uz);

            __m256d V_feqa_S = -V_feqa_N;
            __m256d V_feqa_W = -V_feqa_E;
            __m256d V_feqa_B = -V_feqa_T;
            __m256d V_feqa_SW = -V_feqa_NE;
            __m256d V_feqa_SE = -V_feqa_NW;
            __m256d V_feqa_SB = -V_feqa_NT;
            __m256d V_feqa_ST = -V_feqa_NB;
            __m256d V_feqa_WB = -V_feqa_ET;
            __m256d V_feqa_WT = -V_feqa_EB;


            __m256d V_fs_C = V_SRC_C;
            __m256d V_fs_N = V_half * (V_SRC_N + V_SRC_S);
            __m256d V_fs_E = V_half * (V_SRC_E + V_SRC_W);
            __m256d V_fs_T = V_half * (V_SRC_T + V_SRC_B);
            __m256d V_fs_NE = V_half * (V_SRC_NE + V_SRC_SW);
            __m256d V_fs_NW = V_half * (V_SRC_NW + V_SRC_SE);
            __m256d V_fs_NT = V_half * (V_SRC_NT + V_SRC_SB);
            __m256d V_fs_NB = V_half * (V_SRC_NB + V_SRC_ST);
            __m256d V_fs_ET = V_half * (V_SRC_ET + V_SRC_WB);
            __m256d V_fs_EB = V_half * (V_SRC_EB + V_SRC_WT);

            __m256d V_fs_S = V_fs_N;
            __m256d V_fs_W = V_fs_E;
            __m256d V_fs_B = V_fs_T;
            __m256d V_fs_SW = V_fs_NE;
            __m256d V_fs_SE = V_fs_NW;
            __m256d V_fs_SB = V_fs_NT;
            __m256d V_fs_ST = V_fs_NB;
            __m256d V_fs_WB = V_fs_ET;
            __m256d V_fs_WT = V_fs_EB;

            __m256d V_fa_C = _mm256_set1_pd(0.0);
            __m256d V_fa_N = V_half * (V_SRC_N - V_SRC_S);
            __m256d V_fa_E = V_half * (V_SRC_E - V_SRC_W);
            __m256d V_fa_T = V_half * (V_SRC_T - V_SRC_B);
            __m256d V_fa_NE = V_half * (V_SRC_NE - V_SRC_SW);
            __m256d V_fa_NW = V_half * (V_SRC_NW - V_SRC_SE);
            __m256d V_fa_NT = V_half * (V_SRC_NT - V_SRC_SB);
            __m256d V_fa_NB = V_half * (V_SRC_NB - V_SRC_ST);
            __m256d V_fa_ET = V_half * (V_SRC_ET - V_SRC_WB);
            __m256d V_fa_EB = V_half * (V_SRC_EB - V_SRC_WT);

            __m256d V_fa_S = -V_fa_N;
            __m256d V_fa_W = -V_fa_E;
            __m256d V_fa_B = -V_fa_T;
            __m256d V_fa_SW = -V_fa_NE;
            __m256d V_fa_SE = -V_fa_NW;
            __m256d V_fa_SB = -V_fa_NT;
            __m256d V_fa_ST = -V_fa_NB;
            __m256d V_fa_WB = -V_fa_ET;
            __m256d V_fa_WT = -V_fa_EB;

            __m256d V_DST_C = V_SRC_C - V_OMEGA * (V_fs_C - V_feqs_C);
            __m256d V_DST_N = V_SRC_N - V_OMEGA * (V_fs_N - V_feqs_N) - V_lambda0 * (V_fa_N - V_feqa_N);
            __m256d V_DST_S = V_SRC_S - V_OMEGA * (V_fs_S - V_feqs_S) - V_lambda0 * (V_fa_S - V_feqa_S);
            __m256d V_DST_E = V_SRC_E - V_OMEGA * (V_fs_E - V_feqs_E) - V_lambda0 * (V_fa_E - V_feqa_E);
            __m256d V_DST_W = V_SRC_W - V_OMEGA * (V_fs_W - V_feqs_W) - V_lambda0 * (V_fa_W - V_feqa_W);
            __m256d V_DST_T = V_SRC_T - V_OMEGA * (V_fs_T - V_feqs_T) - V_lambda0 * (V_fa_T - V_feqa_T);
            __m256d V_DST_B = V_SRC_B - V_OMEGA * (V_fs_B - V_feqs_B) - V_lambda0 * (V_fa_B - V_feqa_B);
            __m256d V_DST_NE = V_SRC_NE - V_OMEGA * (V_fs_NE - V_feqs_NE) - V_lambda0 * (V_fa_NE - V_feqa_NE);
            __m256d V_DST_NW = V_SRC_NW - V_OMEGA * (V_fs_NW - V_feqs_NW) - V_lambda0 * (V_fa_NW - V_feqa_NW);
            __m256d V_DST_SE = V_SRC_SE - V_OMEGA * (V_fs_SE - V_feqs_SE) - V_lambda0 * (V_fa_SE - V_feqa_SE);
            __m256d V_DST_SW = V_SRC_SW - V_OMEGA * (V_fs_SW - V_feqs_SW) - V_lambda0 * (V_fa_SW - V_feqa_SW);
            __m256d V_DST_NT = V_SRC_NT - V_OMEGA * (V_fs_NT - V_feqs_NT) - V_lambda0 * (V_fa_NT - V_feqa_NT);
            __m256d V_DST_NB = V_SRC_NB - V_OMEGA * (V_fs_NB - V_feqs_NB) - V_lambda0 * (V_fa_NB - V_feqa_NB);
            __m256d V_DST_ST = V_SRC_ST - V_OMEGA * (V_fs_ST - V_feqs_ST) - V_lambda0 * (V_fa_ST - V_feqa_ST);
            __m256d V_DST_SB = V_SRC_SB - V_OMEGA * (V_fs_SB - V_feqs_SB) - V_lambda0 * (V_fa_SB - V_feqa_SB);
            __m256d V_DST_ET = V_SRC_ET - V_OMEGA * (V_fs_ET - V_feqs_ET) - V_lambda0 * (V_fa_ET - V_feqa_ET);
            __m256d V_DST_EB = V_SRC_EB - V_OMEGA * (V_fs_EB - V_feqs_EB) - V_lambda0 * (V_fa_EB - V_feqa_EB);
            __m256d V_DST_WT = V_SRC_WT - V_OMEGA * (V_fs_WT - V_feqs_WT) - V_lambda0 * (V_fa_WT - V_feqa_WT);
            __m256d V_DST_WB = V_SRC_WB - V_OMEGA * (V_fs_WB - V_feqs_WB) - V_lambda0 * (V_fa_WB - V_feqa_WB);

            DST_C(dstGrid) = V_DST_C[0];
            DST_C_NEXT(dstGrid) = V_DST_C[1];
            DST_C_NEXT2(dstGrid) = V_DST_C[2];
            DST_C_NEXT3(dstGrid) = V_DST_C[3];

            DST_N(dstGrid) = V_DST_N[0];
            DST_N_NEXT(dstGrid) = V_DST_N[1];
            DST_N_NEXT2(dstGrid) = V_DST_N[2];
            DST_N_NEXT3(dstGrid) = V_DST_N[3];

            DST_S(dstGrid) = V_DST_S[0];
            DST_S_NEXT(dstGrid) = V_DST_S[1];
            DST_S_NEXT2(dstGrid) = V_DST_S[2];
            DST_S_NEXT3(dstGrid) = V_DST_S[3];

            DST_E(dstGrid) = V_DST_E[0];
            DST_E_NEXT(dstGrid) = V_DST_E[1];
            DST_E_NEXT2(dstGrid) = V_DST_E[2];
            DST_E_NEXT3(dstGrid) = V_DST_E[3];

            DST_W(dstGrid) = V_DST_W[0];
            DST_W_NEXT(dstGrid) = V_DST_W[1];
            DST_W_NEXT2(dstGrid) = V_DST_W[2];
            DST_W_NEXT3(dstGrid) = V_DST_W[3];

            DST_T(dstGrid) = V_DST_T[0];
            DST_T_NEXT(dstGrid) = V_DST_T[1];
            DST_T_NEXT2(dstGrid) = V_DST_T[2];
            DST_T_NEXT3(dstGrid) = V_DST_T[3];

            DST_B(dstGrid) = V_DST_B[0];
            DST_B_NEXT(dstGrid) = V_DST_B[1];
            DST_B_NEXT2(dstGrid) = V_DST_B[2];
            DST_B_NEXT3(dstGrid) = V_DST_B[3];

            DST_NE(dstGrid) = V_DST_NE[0];
            DST_NE_NEXT(dstGrid) = V_DST_NE[1];
            DST_NE_NEXT2(dstGrid) = V_DST_NE[2];
            DST_NE_NEXT3(dstGrid) = V_DST_NE[3];

            DST_NW(dstGrid) = V_DST_NW[0];
            DST_NW_NEXT(dstGrid) = V_DST_NW[1];
            DST_NW_NEXT2(dstGrid) = V_DST_NW[2];
            DST_NW_NEXT3(dstGrid) = V_DST_NW[3];

            DST_SE(dstGrid) = V_DST_SE[0];
            DST_SE_NEXT(dstGrid) = V_DST_SE[1];
            DST_SE_NEXT2(dstGrid) = V_DST_SE[2];
            DST_SE_NEXT3(dstGrid) = V_DST_SE[3];

            DST_SW(dstGrid) = V_DST_SW[0];
            DST_SW_NEXT(dstGrid) = V_DST_SW[1];
            DST_SW_NEXT2(dstGrid) = V_DST_SW[2];
            DST_SW_NEXT3(dstGrid) = V_DST_SW[3];

            DST_NT(dstGrid) = V_DST_NT[0];
            DST_NT_NEXT(dstGrid) = V_DST_NT[1];
            DST_NT_NEXT2(dstGrid) = V_DST_NT[2];
            DST_NT_NEXT3(dstGrid) = V_DST_NT[3];

            DST_NB(dstGrid) = V_DST_NB[0];
            DST_NB_NEXT(dstGrid) = V_DST_NB[1];
            DST_NB_NEXT2(dstGrid) = V_DST_NB[2];
            DST_NB_NEXT3(dstGrid) = V_DST_NB[3];

            DST_ST(dstGrid) = V_DST_ST[0];
            DST_ST_NEXT(dstGrid) = V_DST_ST[1];
            DST_ST_NEXT2(dstGrid) = V_DST_ST[2];
            DST_ST_NEXT3(dstGrid) = V_DST_ST[3];

            DST_SB(dstGrid) = V_DST_SB[0];
            DST_SB_NEXT(dstGrid) = V_DST_SB[1];
            DST_SB_NEXT2(dstGrid) = V_DST_SB[2];
            DST_SB_NEXT3(dstGrid) = V_DST_SB[3];

            DST_ET(dstGrid) = V_DST_ET[0];
            DST_ET_NEXT(dstGrid) = V_DST_ET[1];
            DST_ET_NEXT2(dstGrid) = V_DST_ET[2];
            DST_ET_NEXT3(dstGrid) = V_DST_ET[3];

            DST_EB(dstGrid) = V_DST_EB[0];
            DST_EB_NEXT(dstGrid) = V_DST_EB[1];
            DST_EB_NEXT2(dstGrid) = V_DST_EB[2];
            DST_EB_NEXT3(dstGrid) = V_DST_EB[3];

            DST_WT(dstGrid) = V_DST_WT[0];
            DST_WT_NEXT(dstGrid) = V_DST_WT[1];
            DST_WT_NEXT2(dstGrid) = V_DST_WT[2];
            DST_WT_NEXT3(dstGrid) = V_DST_WT[3];

            DST_WB(dstGrid) = V_DST_WB[0];
            DST_WB_NEXT(dstGrid) = V_DST_WB[1];
            DST_WB_NEXT2(dstGrid) = V_DST_WB[2];
            DST_WB_NEXT3(dstGrid) = V_DST_WB[3];

            i += N_CELL_ENTRIES + N_CELL_ENTRIES + N_CELL_ENTRIES;
        }
        else{
            if(!TEST_FLAG_SWEEP(srcGrid, OBSTACLE)){
                rho = + SRC_C ( srcGrid ) + SRC_N ( srcGrid )
                      + SRC_S ( srcGrid ) + SRC_E ( srcGrid )
                      + SRC_W ( srcGrid ) + SRC_T ( srcGrid )
                      + SRC_B ( srcGrid ) + SRC_NE( srcGrid )
                      + SRC_NW( srcGrid ) + SRC_SE( srcGrid )
                      + SRC_SW( srcGrid ) + SRC_NT( srcGrid )
                      + SRC_NB( srcGrid ) + SRC_ST( srcGrid )
                      + SRC_SB( srcGrid ) + SRC_ET( srcGrid )
                      + SRC_EB( srcGrid ) + SRC_WT( srcGrid )
                      + SRC_WB( srcGrid );

                ux = + SRC_E ( srcGrid ) - SRC_W ( srcGrid )
                     + SRC_NE( srcGrid ) - SRC_NW( srcGrid )
                     + SRC_SE( srcGrid ) - SRC_SW( srcGrid )
                     + SRC_ET( srcGrid ) + SRC_EB( srcGrid )
                     - SRC_WT( srcGrid ) - SRC_WB( srcGrid );
                uy = + SRC_N ( srcGrid ) - SRC_S ( srcGrid )
                     + SRC_NE( srcGrid ) + SRC_NW( srcGrid )
                     - SRC_SE( srcGrid ) - SRC_SW( srcGrid )
                     + SRC_NT( srcGrid ) + SRC_NB( srcGrid )
                     - SRC_ST( srcGrid ) - SRC_SB( srcGrid );
                uz = + SRC_T ( srcGrid ) - SRC_B ( srcGrid )
                     + SRC_NT( srcGrid ) - SRC_NB( srcGrid )
                     + SRC_ST( srcGrid ) - SRC_SB( srcGrid )
                     + SRC_ET( srcGrid ) - SRC_EB( srcGrid )
                     + SRC_WT( srcGrid ) - SRC_WB( srcGrid );

                ux /= rho;
                uy /= rho;
                uz /= rho;

                if( TEST_FLAG_SWEEP( srcGrid, ACCEL )) {
                    ux = 0.005;
                    uy = 0.002;
                    uz = 0.000;
                }

                u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

                feqs[C ] =            DFL1*rho*(1.0                         - u2);
                feqs[N ] = feqs[S ] = DFL2*rho*(1.0 + 4.5*(+uy   )*(+uy   ) - u2);
                feqs[E ] = feqs[W ] = DFL2*rho*(1.0 + 4.5*(+ux   )*(+ux   ) - u2);
                feqs[T ] = feqs[B ] = DFL2*rho*(1.0 + 4.5*(+uz   )*(+uz   ) - u2);
                feqs[NE] = feqs[SW] = DFL3*rho*(1.0 + 4.5*(+ux+uy)*(+ux+uy) - u2);
                feqs[NW] = feqs[SE] = DFL3*rho*(1.0 + 4.5*(-ux+uy)*(-ux+uy) - u2);
                feqs[NT] = feqs[SB] = DFL3*rho*(1.0 + 4.5*(+uy+uz)*(+uy+uz) - u2);
                feqs[NB] = feqs[ST] = DFL3*rho*(1.0 + 4.5*(+uy-uz)*(+uy-uz) - u2);
                feqs[ET] = feqs[WB] = DFL3*rho*(1.0 + 4.5*(+ux+uz)*(+ux+uz) - u2);
                feqs[EB] = feqs[WT] = DFL3*rho*(1.0 + 4.5*(+ux-uz)*(+ux-uz) - u2);

                feqa[C ] = 0.0;
                feqa[S ] = - (feqa[N ] = DFL2*rho*3.0*(+uy   ));
                feqa[W ] = - (feqa[E ] = DFL2*rho*3.0*(+ux   ));
                feqa[B ] = - (feqa[T ] = DFL2*rho*3.0*(+uz   ));
                feqa[SW] = - (feqa[NE] = DFL3*rho*3.0*(+ux+uy));
                feqa[SE] = - (feqa[NW] = DFL3*rho*3.0*(-ux+uy));
                feqa[SB] = - (feqa[NT] = DFL3*rho*3.0*(+uy+uz));
                feqa[ST] = - (feqa[NB] = DFL3*rho*3.0*(+uy-uz));
                feqa[WB] = - (feqa[ET] = DFL3*rho*3.0*(+ux+uz));
                feqa[WT] = - (feqa[EB] = DFL3*rho*3.0*(+ux-uz));

                fs[C ] =                 SRC_C ( srcGrid );
                fs[N ] = fs[S ] = 0.5 * (SRC_N ( srcGrid ) + SRC_S ( srcGrid ));
                fs[E ] = fs[W ] = 0.5 * (SRC_E ( srcGrid ) + SRC_W ( srcGrid ));
                fs[T ] = fs[B ] = 0.5 * (SRC_T ( srcGrid ) + SRC_B ( srcGrid ));
                fs[NE] = fs[SW] = 0.5 * (SRC_NE( srcGrid ) + SRC_SW( srcGrid ));
                fs[NW] = fs[SE] = 0.5 * (SRC_NW( srcGrid ) + SRC_SE( srcGrid ));
                fs[NT] = fs[SB] = 0.5 * (SRC_NT( srcGrid ) + SRC_SB( srcGrid ));
                fs[NB] = fs[ST] = 0.5 * (SRC_NB( srcGrid ) + SRC_ST( srcGrid ));
                fs[ET] = fs[WB] = 0.5 * (SRC_ET( srcGrid ) + SRC_WB( srcGrid ));
                fs[EB] = fs[WT] = 0.5 * (SRC_EB( srcGrid ) + SRC_WT( srcGrid ));

                fa[C ] = 0.0;
                fa[S ] = - (fa[N ] = 0.5 * (SRC_N ( srcGrid ) - SRC_S ( srcGrid )));
                fa[W ] = - (fa[E ] = 0.5 * (SRC_E ( srcGrid ) - SRC_W ( srcGrid )));
                fa[B ] = - (fa[T ] = 0.5 * (SRC_T ( srcGrid ) - SRC_B ( srcGrid )));
                fa[SW] = - (fa[NE] = 0.5 * (SRC_NE( srcGrid ) - SRC_SW( srcGrid )));
                fa[SE] = - (fa[NW] = 0.5 * (SRC_NW( srcGrid ) - SRC_SE( srcGrid )));
                fa[SB] = - (fa[NT] = 0.5 * (SRC_NT( srcGrid ) - SRC_SB( srcGrid )));
                fa[ST] = - (fa[NB] = 0.5 * (SRC_NB( srcGrid ) - SRC_ST( srcGrid )));
                fa[WB] = - (fa[ET] = 0.5 * (SRC_ET( srcGrid ) - SRC_WB( srcGrid )));
                fa[WT] = - (fa[EB] = 0.5 * (SRC_EB( srcGrid ) - SRC_WT( srcGrid )));

                DST_C ( dstGrid ) = SRC_C ( srcGrid ) - OMEGA * (fs[C ] - feqs[C ])                                ;
                DST_N ( dstGrid ) = SRC_N ( srcGrid ) - OMEGA * (fs[N ] - feqs[N ]) - lambda0 * (fa[N ] - feqa[N ]);
                DST_S ( dstGrid ) = SRC_S ( srcGrid ) - OMEGA * (fs[S ] - feqs[S ]) - lambda0 * (fa[S ] - feqa[S ]);
                DST_E ( dstGrid ) = SRC_E ( srcGrid ) - OMEGA * (fs[E ] - feqs[E ]) - lambda0 * (fa[E ] - feqa[E ]);
                DST_W ( dstGrid ) = SRC_W ( srcGrid ) - OMEGA * (fs[W ] - feqs[W ]) - lambda0 * (fa[W ] - feqa[W ]);
                DST_T ( dstGrid ) = SRC_T ( srcGrid ) - OMEGA * (fs[T ] - feqs[T ]) - lambda0 * (fa[T ] - feqa[T ]);
                DST_B ( dstGrid ) = SRC_B ( srcGrid ) - OMEGA * (fs[B ] - feqs[B ]) - lambda0 * (fa[B ] - feqa[B ]);
                DST_NE( dstGrid ) = SRC_NE( srcGrid ) - OMEGA * (fs[NE] - feqs[NE]) - lambda0 * (fa[NE] - feqa[NE]);
                DST_NW( dstGrid ) = SRC_NW( srcGrid ) - OMEGA * (fs[NW] - feqs[NW]) - lambda0 * (fa[NW] - feqa[NW]);
                DST_SE( dstGrid ) = SRC_SE( srcGrid ) - OMEGA * (fs[SE] - feqs[SE]) - lambda0 * (fa[SE] - feqa[SE]);
                DST_SW( dstGrid ) = SRC_SW( srcGrid ) - OMEGA * (fs[SW] - feqs[SW]) - lambda0 * (fa[SW] - feqa[SW]);
                DST_NT( dstGrid ) = SRC_NT( srcGrid ) - OMEGA * (fs[NT] - feqs[NT]) - lambda0 * (fa[NT] - feqa[NT]);
                DST_NB( dstGrid ) = SRC_NB( srcGrid ) - OMEGA * (fs[NB] - feqs[NB]) - lambda0 * (fa[NB] - feqa[NB]);
                DST_ST( dstGrid ) = SRC_ST( srcGrid ) - OMEGA * (fs[ST] - feqs[ST]) - lambda0 * (fa[ST] - feqa[ST]);
                DST_SB( dstGrid ) = SRC_SB( srcGrid ) - OMEGA * (fs[SB] - feqs[SB]) - lambda0 * (fa[SB] - feqa[SB]);
                DST_ET( dstGrid ) = SRC_ET( srcGrid ) - OMEGA * (fs[ET] - feqs[ET]) - lambda0 * (fa[ET] - feqa[ET]);
                DST_EB( dstGrid ) = SRC_EB( srcGrid ) - OMEGA * (fs[EB] - feqs[EB]) - lambda0 * (fa[EB] - feqa[EB]);
                DST_WT( dstGrid ) = SRC_WT( srcGrid ) - OMEGA * (fs[WT] - feqs[WT]) - lambda0 * (fa[WT] - feqa[WT]);
                DST_WB( dstGrid ) = SRC_WB( srcGrid ) - OMEGA * (fs[WB] - feqs[WB]) - lambda0 * (fa[WB] - feqa[WB]);
            }
            else{
                DST_C ( dstGrid ) = SRC_C ( srcGrid );
                DST_S ( dstGrid ) = SRC_N ( srcGrid );
                DST_N ( dstGrid ) = SRC_S ( srcGrid );
                DST_W ( dstGrid ) = SRC_E ( srcGrid );
                DST_E ( dstGrid ) = SRC_W ( srcGrid );
                DST_B ( dstGrid ) = SRC_T ( srcGrid );
                DST_T ( dstGrid ) = SRC_B ( srcGrid );
                DST_SW( dstGrid ) = SRC_NE( srcGrid );
                DST_SE( dstGrid ) = SRC_NW( srcGrid );
                DST_NW( dstGrid ) = SRC_SE( srcGrid );
                DST_NE( dstGrid ) = SRC_SW( srcGrid );
                DST_SB( dstGrid ) = SRC_NT( srcGrid );
                DST_ST( dstGrid ) = SRC_NB( srcGrid );
                DST_NB( dstGrid ) = SRC_ST( srcGrid );
                DST_NT( dstGrid ) = SRC_SB( srcGrid );
                DST_WB( dstGrid ) = SRC_ET( srcGrid );
                DST_WT( dstGrid ) = SRC_EB( srcGrid );
                DST_EB( dstGrid ) = SRC_WT( srcGrid );
                DST_ET( dstGrid ) = SRC_WB( srcGrid );
            }
            //i += N_CELL_ENTRIES;
            if(!TEST_FLAG_SWEEP_NEXT(srcGrid, OBSTACLE)){
                rho = + SRC_C_NEXT( srcGrid ) + SRC_N_NEXT( srcGrid )
                      + SRC_S_NEXT( srcGrid ) + SRC_E_NEXT( srcGrid )
                      + SRC_W_NEXT( srcGrid ) + SRC_T_NEXT( srcGrid )
                      + SRC_B_NEXT( srcGrid ) + SRC_NE_NEXT( srcGrid )
                      + SRC_NW_NEXT( srcGrid ) + SRC_SE_NEXT( srcGrid )
                      + SRC_SW_NEXT( srcGrid ) + SRC_NT_NEXT( srcGrid )
                      + SRC_NB_NEXT( srcGrid ) + SRC_ST_NEXT( srcGrid )
                      + SRC_SB_NEXT( srcGrid ) + SRC_ET_NEXT( srcGrid )
                      + SRC_EB_NEXT( srcGrid ) + SRC_WT_NEXT( srcGrid )
                      + SRC_WB_NEXT( srcGrid );

                ux = + SRC_E_NEXT( srcGrid ) - SRC_W_NEXT( srcGrid )
                     + SRC_NE_NEXT( srcGrid ) - SRC_NW_NEXT( srcGrid )
                     + SRC_SE_NEXT( srcGrid ) - SRC_SW_NEXT( srcGrid )
                     + SRC_ET_NEXT( srcGrid ) + SRC_EB_NEXT( srcGrid )
                     - SRC_WT_NEXT( srcGrid ) - SRC_WB_NEXT( srcGrid );
                uy = + SRC_N_NEXT( srcGrid ) - SRC_S_NEXT( srcGrid )
                     + SRC_NE_NEXT( srcGrid ) + SRC_NW_NEXT( srcGrid )
                     - SRC_SE_NEXT( srcGrid ) - SRC_SW_NEXT( srcGrid )
                     + SRC_NT_NEXT( srcGrid ) + SRC_NB_NEXT( srcGrid )
                     - SRC_ST_NEXT( srcGrid ) - SRC_SB_NEXT( srcGrid );
                uz = + SRC_T_NEXT( srcGrid ) - SRC_B_NEXT( srcGrid )
                     + SRC_NT_NEXT( srcGrid ) - SRC_NB_NEXT( srcGrid )
                     + SRC_ST_NEXT( srcGrid ) - SRC_SB_NEXT( srcGrid )
                     + SRC_ET_NEXT( srcGrid ) - SRC_EB_NEXT( srcGrid )
                     + SRC_WT_NEXT( srcGrid ) - SRC_WB_NEXT( srcGrid );

                ux /= rho;
                uy /= rho;
                uz /= rho;

                if( TEST_FLAG_SWEEP_NEXT( srcGrid, ACCEL )) {
                    ux = 0.005;
                    uy = 0.002;
                    uz = 0.000;
                }

                u2 = 1.5 *(ux*ux + uy*uy + uz*uz);

                feqs[C ] =            DFL1*rho*(1.0                         - u2);
                feqs[N ] = feqs[S ] = DFL2*rho*(1.0 + 4.5*(+uy   )*(+uy   ) - u2);
                feqs[E ] = feqs[W ] = DFL2*rho*(1.0 + 4.5*(+ux   )*(+ux   ) - u2);
                feqs[T ] = feqs[B ] = DFL2*rho*(1.0 + 4.5*(+uz   )*(+uz   ) - u2);
                feqs[NE] = feqs[SW] = DFL3*rho*(1.0 + 4.5*(+ux+uy)*(+ux+uy) - u2);
                feqs[NW] = feqs[SE] = DFL3*rho*(1.0 + 4.5*(-ux+uy)*(-ux+uy) - u2);
                feqs[NT] = feqs[SB] = DFL3*rho*(1.0 + 4.5*(+uy+uz)*(+uy+uz) - u2);
                feqs[NB] = feqs[ST] = DFL3*rho*(1.0 + 4.5*(+uy-uz)*(+uy-uz) - u2);
                feqs[ET] = feqs[WB] = DFL3*rho*(1.0 + 4.5*(+ux+uz)*(+ux+uz) - u2);
                feqs[EB] = feqs[WT] = DFL3*rho*(1.0 + 4.5*(+ux-uz)*(+ux-uz) - u2);

                feqa[C ] = 0.0;
                feqa[S ] = -(feqa[N ] = DFL2*rho*3.0*(+uy   ));
                feqa[W ] = -(feqa[E ] = DFL2*rho*3.0*(+ux   ));
                feqa[B ] = -(feqa[T ] = DFL2*rho*3.0*(+uz   ));
                feqa[SW] = -(feqa[NE] = DFL3*rho*3.0*(+ux+uy));
                feqa[SE] = -(feqa[NW] = DFL3*rho*3.0*(-ux+uy));
                feqa[SB] = -(feqa[NT] = DFL3*rho*3.0*(+uy+uz));
                feqa[ST] = -(feqa[NB] = DFL3*rho*3.0*(+uy-uz));
                feqa[WB] = -(feqa[ET] = DFL3*rho*3.0*(+ux+uz));
                feqa[WT] = -(feqa[EB] = DFL3*rho*3.0*(+ux-uz));

                fs[C ] =                 SRC_C_NEXT( srcGrid );
                fs[N ] = fs[S ] = 0.5 *(SRC_N_NEXT( srcGrid ) + SRC_S_NEXT( srcGrid ));
                fs[E ] = fs[W ] = 0.5 *(SRC_E_NEXT( srcGrid ) + SRC_W_NEXT( srcGrid ));
                fs[T ] = fs[B ] = 0.5 *(SRC_T_NEXT( srcGrid ) + SRC_B_NEXT( srcGrid ));
                fs[NE] = fs[SW] = 0.5 *(SRC_NE_NEXT( srcGrid ) + SRC_SW_NEXT( srcGrid ));
                fs[NW] = fs[SE] = 0.5 *(SRC_NW_NEXT( srcGrid ) + SRC_SE_NEXT( srcGrid ));
                fs[NT] = fs[SB] = 0.5 *(SRC_NT_NEXT( srcGrid ) + SRC_SB_NEXT( srcGrid ));
                fs[NB] = fs[ST] = 0.5 *(SRC_NB_NEXT( srcGrid ) + SRC_ST_NEXT( srcGrid ));
                fs[ET] = fs[WB] = 0.5 *(SRC_ET_NEXT( srcGrid ) + SRC_WB_NEXT( srcGrid ));
                fs[EB] = fs[WT] = 0.5 *(SRC_EB_NEXT( srcGrid ) + SRC_WT_NEXT( srcGrid ));

                fa[C ] = 0.0;
                fa[S ] = -(fa[N ] = 0.5 *(SRC_N_NEXT( srcGrid ) - SRC_S_NEXT( srcGrid )));
                fa[W ] = -(fa[E ] = 0.5 *(SRC_E_NEXT( srcGrid ) - SRC_W_NEXT( srcGrid )));
                fa[B ] = -(fa[T ] = 0.5 *(SRC_T_NEXT( srcGrid ) - SRC_B_NEXT( srcGrid )));
                fa[SW] = -(fa[NE] = 0.5 *(SRC_NE_NEXT( srcGrid ) - SRC_SW_NEXT( srcGrid )));
                fa[SE] = -(fa[NW] = 0.5 *(SRC_NW_NEXT( srcGrid ) - SRC_SE_NEXT( srcGrid )));
                fa[SB] = -(fa[NT] = 0.5 *(SRC_NT_NEXT( srcGrid ) - SRC_SB_NEXT( srcGrid )));
                fa[ST] = -(fa[NB] = 0.5 *(SRC_NB_NEXT( srcGrid ) - SRC_ST_NEXT( srcGrid )));
                fa[WB] = -(fa[ET] = 0.5 *(SRC_ET_NEXT( srcGrid ) - SRC_WB_NEXT( srcGrid )));
                fa[WT] = -(fa[EB] = 0.5 *(SRC_EB_NEXT( srcGrid ) - SRC_WT_NEXT( srcGrid )));

                DST_C_NEXT( dstGrid ) = SRC_C_NEXT( srcGrid ) - OMEGA *(fs[C ] - feqs[C ])                                ;
                DST_N_NEXT( dstGrid ) = SRC_N_NEXT( srcGrid ) - OMEGA *(fs[N ] - feqs[N ]) - lambda0 *(fa[N ] - feqa[N ]);
                DST_S_NEXT( dstGrid ) = SRC_S_NEXT( srcGrid ) - OMEGA *(fs[S ] - feqs[S ]) - lambda0 *(fa[S ] - feqa[S ]);
                DST_E_NEXT( dstGrid ) = SRC_E_NEXT( srcGrid ) - OMEGA *(fs[E ] - feqs[E ]) - lambda0 *(fa[E ] - feqa[E ]);
                DST_W_NEXT( dstGrid ) = SRC_W_NEXT( srcGrid ) - OMEGA *(fs[W ] - feqs[W ]) - lambda0 *(fa[W ] - feqa[W ]);
                DST_T_NEXT( dstGrid ) = SRC_T_NEXT( srcGrid ) - OMEGA *(fs[T ] - feqs[T ]) - lambda0 *(fa[T ] - feqa[T ]);
                DST_B_NEXT( dstGrid ) = SRC_B_NEXT( srcGrid ) - OMEGA *(fs[B ] - feqs[B ]) - lambda0 *(fa[B ] - feqa[B ]);
                DST_NE_NEXT( dstGrid ) = SRC_NE_NEXT( srcGrid ) - OMEGA *(fs[NE] - feqs[NE]) - lambda0 *(fa[NE] - feqa[NE]);
                DST_NW_NEXT( dstGrid ) = SRC_NW_NEXT( srcGrid ) - OMEGA *(fs[NW] - feqs[NW]) - lambda0 *(fa[NW] - feqa[NW]);
                DST_SE_NEXT( dstGrid ) = SRC_SE_NEXT( srcGrid ) - OMEGA *(fs[SE] - feqs[SE]) - lambda0 *(fa[SE] - feqa[SE]);
                DST_SW_NEXT( dstGrid ) = SRC_SW_NEXT( srcGrid ) - OMEGA *(fs[SW] - feqs[SW]) - lambda0 *(fa[SW] - feqa[SW]);
                DST_NT_NEXT( dstGrid ) = SRC_NT_NEXT( srcGrid ) - OMEGA *(fs[NT] - feqs[NT]) - lambda0 *(fa[NT] - feqa[NT]);
                DST_NB_NEXT( dstGrid ) = SRC_NB_NEXT( srcGrid ) - OMEGA *(fs[NB] - feqs[NB]) - lambda0 *(fa[NB] - feqa[NB]);
                DST_ST_NEXT( dstGrid ) = SRC_ST_NEXT( srcGrid ) - OMEGA *(fs[ST] - feqs[ST]) - lambda0 *(fa[ST] - feqa[ST]);
                DST_SB_NEXT( dstGrid ) = SRC_SB_NEXT( srcGrid ) - OMEGA *(fs[SB] - feqs[SB]) - lambda0 *(fa[SB] - feqa[SB]);
                DST_ET_NEXT( dstGrid ) = SRC_ET_NEXT( srcGrid ) - OMEGA *(fs[ET] - feqs[ET]) - lambda0 *(fa[ET] - feqa[ET]);
                DST_EB_NEXT( dstGrid ) = SRC_EB_NEXT( srcGrid ) - OMEGA *(fs[EB] - feqs[EB]) - lambda0 *(fa[EB] - feqa[EB]);
                DST_WT_NEXT( dstGrid ) = SRC_WT_NEXT( srcGrid ) - OMEGA *(fs[WT] - feqs[WT]) - lambda0 *(fa[WT] - feqa[WT]);
                DST_WB_NEXT( dstGrid ) = SRC_WB_NEXT( srcGrid ) - OMEGA *(fs[WB] - feqs[WB]) - lambda0 *(fa[WB] - feqa[WB]);
            }
            else{  DST_C_NEXT( dstGrid ) = SRC_C_NEXT( srcGrid );
                DST_S_NEXT( dstGrid ) = SRC_N_NEXT( srcGrid );
                DST_N_NEXT( dstGrid ) = SRC_S_NEXT( srcGrid );
                DST_W_NEXT( dstGrid ) = SRC_E_NEXT( srcGrid );
                DST_E_NEXT( dstGrid ) = SRC_W_NEXT( srcGrid );
                DST_B_NEXT( dstGrid ) = SRC_T_NEXT( srcGrid );
                DST_T_NEXT( dstGrid ) = SRC_B_NEXT( srcGrid );
                DST_SW_NEXT( dstGrid ) = SRC_NE_NEXT( srcGrid );
                DST_SE_NEXT( dstGrid ) = SRC_NW_NEXT( srcGrid );
                DST_NW_NEXT( dstGrid ) = SRC_SE_NEXT( srcGrid );
                DST_NE_NEXT( dstGrid ) = SRC_SW_NEXT( srcGrid );
                DST_SB_NEXT( dstGrid ) = SRC_NT_NEXT( srcGrid );
                DST_ST_NEXT( dstGrid ) = SRC_NB_NEXT( srcGrid );
                DST_NB_NEXT( dstGrid ) = SRC_ST_NEXT( srcGrid );
                DST_NT_NEXT( dstGrid ) = SRC_SB_NEXT( srcGrid );
                DST_WB_NEXT( dstGrid ) = SRC_ET_NEXT( srcGrid );
                DST_WT_NEXT( dstGrid ) = SRC_EB_NEXT( srcGrid );
                DST_EB_NEXT( dstGrid ) = SRC_WT_NEXT( srcGrid );
                DST_ET_NEXT( dstGrid ) = SRC_WB_NEXT( srcGrid );
            }

            if(!TEST_FLAG_SWEEP_NEXT2(srcGrid, OBSTACLE)){
                rho = + SRC_C_NEXT2( srcGrid ) + SRC_N_NEXT2( srcGrid )
                      + SRC_S_NEXT2( srcGrid ) + SRC_E_NEXT2( srcGrid )
                      + SRC_W_NEXT2( srcGrid ) + SRC_T_NEXT2( srcGrid )
                      + SRC_B_NEXT2( srcGrid ) + SRC_NE_NEXT2( srcGrid )
                      + SRC_NW_NEXT2( srcGrid ) + SRC_SE_NEXT2( srcGrid )
                      + SRC_SW_NEXT2( srcGrid ) + SRC_NT_NEXT2( srcGrid )
                      + SRC_NB_NEXT2( srcGrid ) + SRC_ST_NEXT2( srcGrid )
                      + SRC_SB_NEXT2( srcGrid ) + SRC_ET_NEXT2( srcGrid )
                      + SRC_EB_NEXT2( srcGrid ) + SRC_WT_NEXT2( srcGrid )
                      + SRC_WB_NEXT2( srcGrid );

                ux = + SRC_E_NEXT2( srcGrid ) - SRC_W_NEXT2( srcGrid )
                     + SRC_NE_NEXT2( srcGrid ) - SRC_NW_NEXT2( srcGrid )
                     + SRC_SE_NEXT2( srcGrid ) - SRC_SW_NEXT2( srcGrid )
                     + SRC_ET_NEXT2( srcGrid ) + SRC_EB_NEXT2( srcGrid )
                     - SRC_WT_NEXT2( srcGrid ) - SRC_WB_NEXT2( srcGrid );
                uy = + SRC_N_NEXT2( srcGrid ) - SRC_S_NEXT2( srcGrid )
                     + SRC_NE_NEXT2( srcGrid ) + SRC_NW_NEXT2( srcGrid )
                     - SRC_SE_NEXT2( srcGrid ) - SRC_SW_NEXT2( srcGrid )
                     + SRC_NT_NEXT2( srcGrid ) + SRC_NB_NEXT2( srcGrid )
                     - SRC_ST_NEXT2( srcGrid ) - SRC_SB_NEXT2( srcGrid );
                uz = + SRC_T_NEXT2( srcGrid ) - SRC_B_NEXT2( srcGrid )
                     + SRC_NT_NEXT2( srcGrid ) - SRC_NB_NEXT2( srcGrid )
                     + SRC_ST_NEXT2( srcGrid ) - SRC_SB_NEXT2( srcGrid )
                     + SRC_ET_NEXT2( srcGrid ) - SRC_EB_NEXT2( srcGrid )
                     + SRC_WT_NEXT2( srcGrid ) - SRC_WB_NEXT2( srcGrid );

                ux /= rho;
                uy /= rho;
                uz /= rho;

                if( TEST_FLAG_SWEEP_NEXT2( srcGrid, ACCEL )) {
                    ux = 0.005;
                    uy = 0.002;
                    uz = 0.000;
                }

                u2 = 1.5 *(ux*ux + uy*uy + uz*uz);

                feqs[C ] =            DFL1*rho*(1.0                         - u2);
                feqs[N ] = feqs[S ] = DFL2*rho*(1.0 + 4.5*(+uy   )*(+uy   ) - u2);
                feqs[E ] = feqs[W ] = DFL2*rho*(1.0 + 4.5*(+ux   )*(+ux   ) - u2);
                feqs[T ] = feqs[B ] = DFL2*rho*(1.0 + 4.5*(+uz   )*(+uz   ) - u2);
                feqs[NE] = feqs[SW] = DFL3*rho*(1.0 + 4.5*(+ux+uy)*(+ux+uy) - u2);
                feqs[NW] = feqs[SE] = DFL3*rho*(1.0 + 4.5*(-ux+uy)*(-ux+uy) - u2);
                feqs[NT] = feqs[SB] = DFL3*rho*(1.0 + 4.5*(+uy+uz)*(+uy+uz) - u2);
                feqs[NB] = feqs[ST] = DFL3*rho*(1.0 + 4.5*(+uy-uz)*(+uy-uz) - u2);
                feqs[ET] = feqs[WB] = DFL3*rho*(1.0 + 4.5*(+ux+uz)*(+ux+uz) - u2);
                feqs[EB] = feqs[WT] = DFL3*rho*(1.0 + 4.5*(+ux-uz)*(+ux-uz) - u2);

                feqa[C ] = 0.0;
                feqa[S ] = -(feqa[N ] = DFL2*rho*3.0*(+uy   ));
                feqa[W ] = -(feqa[E ] = DFL2*rho*3.0*(+ux   ));
                feqa[B ] = -(feqa[T ] = DFL2*rho*3.0*(+uz   ));
                feqa[SW] = -(feqa[NE] = DFL3*rho*3.0*(+ux+uy));
                feqa[SE] = -(feqa[NW] = DFL3*rho*3.0*(-ux+uy));
                feqa[SB] = -(feqa[NT] = DFL3*rho*3.0*(+uy+uz));
                feqa[ST] = -(feqa[NB] = DFL3*rho*3.0*(+uy-uz));
                feqa[WB] = -(feqa[ET] = DFL3*rho*3.0*(+ux+uz));
                feqa[WT] = -(feqa[EB] = DFL3*rho*3.0*(+ux-uz));

                fs[C ] =                 SRC_C_NEXT2( srcGrid );
                fs[N ] = fs[S ] = 0.5 *(SRC_N_NEXT2( srcGrid ) + SRC_S_NEXT2( srcGrid ));
                fs[E ] = fs[W ] = 0.5 *(SRC_E_NEXT2( srcGrid ) + SRC_W_NEXT2( srcGrid ));
                fs[T ] = fs[B ] = 0.5 *(SRC_T_NEXT2( srcGrid ) + SRC_B_NEXT2( srcGrid ));
                fs[NE] = fs[SW] = 0.5 *(SRC_NE_NEXT2( srcGrid ) + SRC_SW_NEXT2( srcGrid ));
                fs[NW] = fs[SE] = 0.5 *(SRC_NW_NEXT2( srcGrid ) + SRC_SE_NEXT2( srcGrid ));
                fs[NT] = fs[SB] = 0.5 *(SRC_NT_NEXT2( srcGrid ) + SRC_SB_NEXT2( srcGrid ));
                fs[NB] = fs[ST] = 0.5 *(SRC_NB_NEXT2( srcGrid ) + SRC_ST_NEXT2( srcGrid ));
                fs[ET] = fs[WB] = 0.5 *(SRC_ET_NEXT2( srcGrid ) + SRC_WB_NEXT2( srcGrid ));
                fs[EB] = fs[WT] = 0.5 *(SRC_EB_NEXT2( srcGrid ) + SRC_WT_NEXT2( srcGrid ));

                fa[C ] = 0.0;
                fa[S ] = -(fa[N ] = 0.5 *(SRC_N_NEXT2( srcGrid ) - SRC_S_NEXT2( srcGrid )));
                fa[W ] = -(fa[E ] = 0.5 *(SRC_E_NEXT2( srcGrid ) - SRC_W_NEXT2( srcGrid )));
                fa[B ] = -(fa[T ] = 0.5 *(SRC_T_NEXT2( srcGrid ) - SRC_B_NEXT2( srcGrid )));
                fa[SW] = -(fa[NE] = 0.5 *(SRC_NE_NEXT2( srcGrid ) - SRC_SW_NEXT2( srcGrid )));
                fa[SE] = -(fa[NW] = 0.5 *(SRC_NW_NEXT2( srcGrid ) - SRC_SE_NEXT2( srcGrid )));
                fa[SB] = -(fa[NT] = 0.5 *(SRC_NT_NEXT2( srcGrid ) - SRC_SB_NEXT2( srcGrid )));
                fa[ST] = -(fa[NB] = 0.5 *(SRC_NB_NEXT2( srcGrid ) - SRC_ST_NEXT2( srcGrid )));
                fa[WB] = -(fa[ET] = 0.5 *(SRC_ET_NEXT2( srcGrid ) - SRC_WB_NEXT2( srcGrid )));
                fa[WT] = -(fa[EB] = 0.5 *(SRC_EB_NEXT2( srcGrid ) - SRC_WT_NEXT2( srcGrid )));

                DST_C_NEXT2( dstGrid ) = SRC_C_NEXT2( srcGrid ) - OMEGA *(fs[C ] - feqs[C ])                                ;
                DST_N_NEXT2( dstGrid ) = SRC_N_NEXT2( srcGrid ) - OMEGA *(fs[N ] - feqs[N ]) - lambda0 *(fa[N ] - feqa[N ]);
                DST_S_NEXT2( dstGrid ) = SRC_S_NEXT2( srcGrid ) - OMEGA *(fs[S ] - feqs[S ]) - lambda0 *(fa[S ] - feqa[S ]);
                DST_E_NEXT2( dstGrid ) = SRC_E_NEXT2( srcGrid ) - OMEGA *(fs[E ] - feqs[E ]) - lambda0 *(fa[E ] - feqa[E ]);
                DST_W_NEXT2( dstGrid ) = SRC_W_NEXT2( srcGrid ) - OMEGA *(fs[W ] - feqs[W ]) - lambda0 *(fa[W ] - feqa[W ]);
                DST_T_NEXT2( dstGrid ) = SRC_T_NEXT2( srcGrid ) - OMEGA *(fs[T ] - feqs[T ]) - lambda0 *(fa[T ] - feqa[T ]);
                DST_B_NEXT2( dstGrid ) = SRC_B_NEXT2( srcGrid ) - OMEGA *(fs[B ] - feqs[B ]) - lambda0 *(fa[B ] - feqa[B ]);
                DST_NE_NEXT2( dstGrid ) = SRC_NE_NEXT2( srcGrid ) - OMEGA *(fs[NE] - feqs[NE]) - lambda0 *(fa[NE] - feqa[NE]);
                DST_NW_NEXT2( dstGrid ) = SRC_NW_NEXT2( srcGrid ) - OMEGA *(fs[NW] - feqs[NW]) - lambda0 *(fa[NW] - feqa[NW]);
                DST_SE_NEXT2( dstGrid ) = SRC_SE_NEXT2( srcGrid ) - OMEGA *(fs[SE] - feqs[SE]) - lambda0 *(fa[SE] - feqa[SE]);
                DST_SW_NEXT2( dstGrid ) = SRC_SW_NEXT2( srcGrid ) - OMEGA *(fs[SW] - feqs[SW]) - lambda0 *(fa[SW] - feqa[SW]);
                DST_NT_NEXT2( dstGrid ) = SRC_NT_NEXT2( srcGrid ) - OMEGA *(fs[NT] - feqs[NT]) - lambda0 *(fa[NT] - feqa[NT]);
                DST_NB_NEXT2( dstGrid ) = SRC_NB_NEXT2( srcGrid ) - OMEGA *(fs[NB] - feqs[NB]) - lambda0 *(fa[NB] - feqa[NB]);
                DST_ST_NEXT2( dstGrid ) = SRC_ST_NEXT2( srcGrid ) - OMEGA *(fs[ST] - feqs[ST]) - lambda0 *(fa[ST] - feqa[ST]);
                DST_SB_NEXT2( dstGrid ) = SRC_SB_NEXT2( srcGrid ) - OMEGA *(fs[SB] - feqs[SB]) - lambda0 *(fa[SB] - feqa[SB]);
                DST_ET_NEXT2( dstGrid ) = SRC_ET_NEXT2( srcGrid ) - OMEGA *(fs[ET] - feqs[ET]) - lambda0 *(fa[ET] - feqa[ET]);
                DST_EB_NEXT2( dstGrid ) = SRC_EB_NEXT2( srcGrid ) - OMEGA *(fs[EB] - feqs[EB]) - lambda0 *(fa[EB] - feqa[EB]);
                DST_WT_NEXT2( dstGrid ) = SRC_WT_NEXT2( srcGrid ) - OMEGA *(fs[WT] - feqs[WT]) - lambda0 *(fa[WT] - feqa[WT]);
                DST_WB_NEXT2( dstGrid ) = SRC_WB_NEXT2( srcGrid ) - OMEGA *(fs[WB] - feqs[WB]) - lambda0 *(fa[WB] - feqa[WB]);
            }
            else{  DST_C_NEXT2( dstGrid ) = SRC_C_NEXT2( srcGrid );
                DST_S_NEXT2( dstGrid ) = SRC_N_NEXT2( srcGrid );
                DST_N_NEXT2( dstGrid ) = SRC_S_NEXT2( srcGrid );
                DST_W_NEXT2( dstGrid ) = SRC_E_NEXT2( srcGrid );
                DST_E_NEXT2( dstGrid ) = SRC_W_NEXT2( srcGrid );
                DST_B_NEXT2( dstGrid ) = SRC_T_NEXT2( srcGrid );
                DST_T_NEXT2( dstGrid ) = SRC_B_NEXT2( srcGrid );
                DST_SW_NEXT2( dstGrid ) = SRC_NE_NEXT2( srcGrid );
                DST_SE_NEXT2( dstGrid ) = SRC_NW_NEXT2( srcGrid );
                DST_NW_NEXT2( dstGrid ) = SRC_SE_NEXT2( srcGrid );
                DST_NE_NEXT2( dstGrid ) = SRC_SW_NEXT2( srcGrid );
                DST_SB_NEXT2( dstGrid ) = SRC_NT_NEXT2( srcGrid );
                DST_ST_NEXT2( dstGrid ) = SRC_NB_NEXT2( srcGrid );
                DST_NB_NEXT2( dstGrid ) = SRC_ST_NEXT2( srcGrid );
                DST_NT_NEXT2( dstGrid ) = SRC_SB_NEXT2( srcGrid );
                DST_WB_NEXT2( dstGrid ) = SRC_ET_NEXT2( srcGrid );
                DST_WT_NEXT2( dstGrid ) = SRC_EB_NEXT2( srcGrid );
                DST_EB_NEXT2( dstGrid ) = SRC_WT_NEXT2( srcGrid );
                DST_ET_NEXT2( dstGrid ) = SRC_WB_NEXT2( srcGrid );
            }

            if(!TEST_FLAG_SWEEP_NEXT3(srcGrid, OBSTACLE)){
                rho = + SRC_C_NEXT3( srcGrid ) + SRC_N_NEXT3( srcGrid )
                      + SRC_S_NEXT3( srcGrid ) + SRC_E_NEXT3( srcGrid )
                      + SRC_W_NEXT3( srcGrid ) + SRC_T_NEXT3( srcGrid )
                      + SRC_B_NEXT3( srcGrid ) + SRC_NE_NEXT3( srcGrid )
                      + SRC_NW_NEXT3( srcGrid ) + SRC_SE_NEXT3( srcGrid )
                      + SRC_SW_NEXT3( srcGrid ) + SRC_NT_NEXT3( srcGrid )
                      + SRC_NB_NEXT3( srcGrid ) + SRC_ST_NEXT3( srcGrid )
                      + SRC_SB_NEXT3( srcGrid ) + SRC_ET_NEXT3( srcGrid )
                      + SRC_EB_NEXT3( srcGrid ) + SRC_WT_NEXT3( srcGrid )
                      + SRC_WB_NEXT3( srcGrid );

                ux = + SRC_E_NEXT3( srcGrid ) - SRC_W_NEXT3( srcGrid )
                     + SRC_NE_NEXT3( srcGrid ) - SRC_NW_NEXT3( srcGrid )
                     + SRC_SE_NEXT3( srcGrid ) - SRC_SW_NEXT3( srcGrid )
                     + SRC_ET_NEXT3( srcGrid ) + SRC_EB_NEXT3( srcGrid )
                     - SRC_WT_NEXT3( srcGrid ) - SRC_WB_NEXT3( srcGrid );
                uy = + SRC_N_NEXT3( srcGrid ) - SRC_S_NEXT3( srcGrid )
                     + SRC_NE_NEXT3( srcGrid ) + SRC_NW_NEXT3( srcGrid )
                     - SRC_SE_NEXT3( srcGrid ) - SRC_SW_NEXT3( srcGrid )
                     + SRC_NT_NEXT3( srcGrid ) + SRC_NB_NEXT3( srcGrid )
                     - SRC_ST_NEXT3( srcGrid ) - SRC_SB_NEXT3( srcGrid );
                uz = + SRC_T_NEXT3( srcGrid ) - SRC_B_NEXT3( srcGrid )
                     + SRC_NT_NEXT3( srcGrid ) - SRC_NB_NEXT3( srcGrid )
                     + SRC_ST_NEXT3( srcGrid ) - SRC_SB_NEXT3( srcGrid )
                     + SRC_ET_NEXT3( srcGrid ) - SRC_EB_NEXT3( srcGrid )
                     + SRC_WT_NEXT3( srcGrid ) - SRC_WB_NEXT3( srcGrid );

                ux /= rho;
                uy /= rho;
                uz /= rho;

                if( TEST_FLAG_SWEEP_NEXT3( srcGrid, ACCEL )) {
                    ux = 0.005;
                    uy = 0.002;
                    uz = 0.000;
                }

                u2 = 1.5 *(ux*ux + uy*uy + uz*uz);

                feqs[C ] =            DFL1*rho*(1.0                         - u2);
                feqs[N ] = feqs[S ] = DFL2*rho*(1.0 + 4.5*(+uy   )*(+uy   ) - u2);
                feqs[E ] = feqs[W ] = DFL2*rho*(1.0 + 4.5*(+ux   )*(+ux   ) - u2);
                feqs[T ] = feqs[B ] = DFL2*rho*(1.0 + 4.5*(+uz   )*(+uz   ) - u2);
                feqs[NE] = feqs[SW] = DFL3*rho*(1.0 + 4.5*(+ux+uy)*(+ux+uy) - u2);
                feqs[NW] = feqs[SE] = DFL3*rho*(1.0 + 4.5*(-ux+uy)*(-ux+uy) - u2);
                feqs[NT] = feqs[SB] = DFL3*rho*(1.0 + 4.5*(+uy+uz)*(+uy+uz) - u2);
                feqs[NB] = feqs[ST] = DFL3*rho*(1.0 + 4.5*(+uy-uz)*(+uy-uz) - u2);
                feqs[ET] = feqs[WB] = DFL3*rho*(1.0 + 4.5*(+ux+uz)*(+ux+uz) - u2);
                feqs[EB] = feqs[WT] = DFL3*rho*(1.0 + 4.5*(+ux-uz)*(+ux-uz) - u2);

                feqa[C ] = 0.0;
                feqa[S ] = -(feqa[N ] = DFL2*rho*3.0*(+uy   ));
                feqa[W ] = -(feqa[E ] = DFL2*rho*3.0*(+ux   ));
                feqa[B ] = -(feqa[T ] = DFL2*rho*3.0*(+uz   ));
                feqa[SW] = -(feqa[NE] = DFL3*rho*3.0*(+ux+uy));
                feqa[SE] = -(feqa[NW] = DFL3*rho*3.0*(-ux+uy));
                feqa[SB] = -(feqa[NT] = DFL3*rho*3.0*(+uy+uz));
                feqa[ST] = -(feqa[NB] = DFL3*rho*3.0*(+uy-uz));
                feqa[WB] = -(feqa[ET] = DFL3*rho*3.0*(+ux+uz));
                feqa[WT] = -(feqa[EB] = DFL3*rho*3.0*(+ux-uz));

                fs[C ] =                 SRC_C_NEXT3( srcGrid );
                fs[N ] = fs[S ] = 0.5 *(SRC_N_NEXT3( srcGrid ) + SRC_S_NEXT3( srcGrid ));
                fs[E ] = fs[W ] = 0.5 *(SRC_E_NEXT3( srcGrid ) + SRC_W_NEXT3( srcGrid ));
                fs[T ] = fs[B ] = 0.5 *(SRC_T_NEXT3( srcGrid ) + SRC_B_NEXT3( srcGrid ));
                fs[NE] = fs[SW] = 0.5 *(SRC_NE_NEXT3( srcGrid ) + SRC_SW_NEXT3( srcGrid ));
                fs[NW] = fs[SE] = 0.5 *(SRC_NW_NEXT3( srcGrid ) + SRC_SE_NEXT3( srcGrid ));
                fs[NT] = fs[SB] = 0.5 *(SRC_NT_NEXT3( srcGrid ) + SRC_SB_NEXT3( srcGrid ));
                fs[NB] = fs[ST] = 0.5 *(SRC_NB_NEXT3( srcGrid ) + SRC_ST_NEXT3( srcGrid ));
                fs[ET] = fs[WB] = 0.5 *(SRC_ET_NEXT3( srcGrid ) + SRC_WB_NEXT3( srcGrid ));
                fs[EB] = fs[WT] = 0.5 *(SRC_EB_NEXT3( srcGrid ) + SRC_WT_NEXT3( srcGrid ));

                fa[C ] = 0.0;
                fa[S ] = -(fa[N ] = 0.5 *(SRC_N_NEXT3( srcGrid ) - SRC_S_NEXT3( srcGrid )));
                fa[W ] = -(fa[E ] = 0.5 *(SRC_E_NEXT3( srcGrid ) - SRC_W_NEXT3( srcGrid )));
                fa[B ] = -(fa[T ] = 0.5 *(SRC_T_NEXT3( srcGrid ) - SRC_B_NEXT3( srcGrid )));
                fa[SW] = -(fa[NE] = 0.5 *(SRC_NE_NEXT3( srcGrid ) - SRC_SW_NEXT3( srcGrid )));
                fa[SE] = -(fa[NW] = 0.5 *(SRC_NW_NEXT3( srcGrid ) - SRC_SE_NEXT3( srcGrid )));
                fa[SB] = -(fa[NT] = 0.5 *(SRC_NT_NEXT3( srcGrid ) - SRC_SB_NEXT3( srcGrid )));
                fa[ST] = -(fa[NB] = 0.5 *(SRC_NB_NEXT3( srcGrid ) - SRC_ST_NEXT3( srcGrid )));
                fa[WB] = -(fa[ET] = 0.5 *(SRC_ET_NEXT3( srcGrid ) - SRC_WB_NEXT3( srcGrid )));
                fa[WT] = -(fa[EB] = 0.5 *(SRC_EB_NEXT3( srcGrid ) - SRC_WT_NEXT3( srcGrid )));

                DST_C_NEXT3( dstGrid ) = SRC_C_NEXT3( srcGrid ) - OMEGA *(fs[C ] - feqs[C ])                                ;
                DST_N_NEXT3( dstGrid ) = SRC_N_NEXT3( srcGrid ) - OMEGA *(fs[N ] - feqs[N ]) - lambda0 *(fa[N ] - feqa[N ]);
                DST_S_NEXT3( dstGrid ) = SRC_S_NEXT3( srcGrid ) - OMEGA *(fs[S ] - feqs[S ]) - lambda0 *(fa[S ] - feqa[S ]);
                DST_E_NEXT3( dstGrid ) = SRC_E_NEXT3( srcGrid ) - OMEGA *(fs[E ] - feqs[E ]) - lambda0 *(fa[E ] - feqa[E ]);
                DST_W_NEXT3( dstGrid ) = SRC_W_NEXT3( srcGrid ) - OMEGA *(fs[W ] - feqs[W ]) - lambda0 *(fa[W ] - feqa[W ]);
                DST_T_NEXT3( dstGrid ) = SRC_T_NEXT3( srcGrid ) - OMEGA *(fs[T ] - feqs[T ]) - lambda0 *(fa[T ] - feqa[T ]);
                DST_B_NEXT3( dstGrid ) = SRC_B_NEXT3( srcGrid ) - OMEGA *(fs[B ] - feqs[B ]) - lambda0 *(fa[B ] - feqa[B ]);
                DST_NE_NEXT3( dstGrid ) = SRC_NE_NEXT3( srcGrid ) - OMEGA *(fs[NE] - feqs[NE]) - lambda0 *(fa[NE] - feqa[NE]);
                DST_NW_NEXT3( dstGrid ) = SRC_NW_NEXT3( srcGrid ) - OMEGA *(fs[NW] - feqs[NW]) - lambda0 *(fa[NW] - feqa[NW]);
                DST_SE_NEXT3( dstGrid ) = SRC_SE_NEXT3( srcGrid ) - OMEGA *(fs[SE] - feqs[SE]) - lambda0 *(fa[SE] - feqa[SE]);
                DST_SW_NEXT3( dstGrid ) = SRC_SW_NEXT3( srcGrid ) - OMEGA *(fs[SW] - feqs[SW]) - lambda0 *(fa[SW] - feqa[SW]);
                DST_NT_NEXT3( dstGrid ) = SRC_NT_NEXT3( srcGrid ) - OMEGA *(fs[NT] - feqs[NT]) - lambda0 *(fa[NT] - feqa[NT]);
                DST_NB_NEXT3( dstGrid ) = SRC_NB_NEXT3( srcGrid ) - OMEGA *(fs[NB] - feqs[NB]) - lambda0 *(fa[NB] - feqa[NB]);
                DST_ST_NEXT3( dstGrid ) = SRC_ST_NEXT3( srcGrid ) - OMEGA *(fs[ST] - feqs[ST]) - lambda0 *(fa[ST] - feqa[ST]);
                DST_SB_NEXT3( dstGrid ) = SRC_SB_NEXT3( srcGrid ) - OMEGA *(fs[SB] - feqs[SB]) - lambda0 *(fa[SB] - feqa[SB]);
                DST_ET_NEXT3( dstGrid ) = SRC_ET_NEXT3( srcGrid ) - OMEGA *(fs[ET] - feqs[ET]) - lambda0 *(fa[ET] - feqa[ET]);
                DST_EB_NEXT3( dstGrid ) = SRC_EB_NEXT3( srcGrid ) - OMEGA *(fs[EB] - feqs[EB]) - lambda0 *(fa[EB] - feqa[EB]);
                DST_WT_NEXT3( dstGrid ) = SRC_WT_NEXT3( srcGrid ) - OMEGA *(fs[WT] - feqs[WT]) - lambda0 *(fa[WT] - feqa[WT]);
                DST_WB_NEXT3( dstGrid ) = SRC_WB_NEXT3( srcGrid ) - OMEGA *(fs[WB] - feqs[WB]) - lambda0 *(fa[WB] - feqa[WB]);
            }
            else{  DST_C_NEXT3( dstGrid ) = SRC_C_NEXT3( srcGrid );
                DST_S_NEXT3( dstGrid ) = SRC_N_NEXT3( srcGrid );
                DST_N_NEXT3( dstGrid ) = SRC_S_NEXT3( srcGrid );
                DST_W_NEXT3( dstGrid ) = SRC_E_NEXT3( srcGrid );
                DST_E_NEXT3( dstGrid ) = SRC_W_NEXT3( srcGrid );
                DST_B_NEXT3( dstGrid ) = SRC_T_NEXT3( srcGrid );
                DST_T_NEXT3( dstGrid ) = SRC_B_NEXT3( srcGrid );
                DST_SW_NEXT3( dstGrid ) = SRC_NE_NEXT3( srcGrid );
                DST_SE_NEXT3( dstGrid ) = SRC_NW_NEXT3( srcGrid );
                DST_NW_NEXT3( dstGrid ) = SRC_SE_NEXT3( srcGrid );
                DST_NE_NEXT3( dstGrid ) = SRC_SW_NEXT3( srcGrid );
                DST_SB_NEXT3( dstGrid ) = SRC_NT_NEXT3( srcGrid );
                DST_ST_NEXT3( dstGrid ) = SRC_NB_NEXT3( srcGrid );
                DST_NB_NEXT3( dstGrid ) = SRC_ST_NEXT3( srcGrid );
                DST_NT_NEXT3( dstGrid ) = SRC_SB_NEXT3( srcGrid );
                DST_WB_NEXT3( dstGrid ) = SRC_ET_NEXT3( srcGrid );
                DST_WT_NEXT3( dstGrid ) = SRC_EB_NEXT3( srcGrid );
                DST_EB_NEXT3( dstGrid ) = SRC_WT_NEXT3( srcGrid );
                DST_ET_NEXT3( dstGrid ) = SRC_WB_NEXT3( srcGrid );
            }
            i += N_CELL_ENTRIES + N_CELL_ENTRIES + N_CELL_ENTRIES;
            //printf("Debug i2:%d\n", i);

        }
    SWEEP_END
}
/*############################################################################*/

void LBM_handleInOutFlow( LBM_Grid srcGrid ) {
	double ux , uy , uz , rho ,
	       ux1, uy1, uz1, rho1,
	       ux2, uy2, uz2, rho2,
	       u2, px, py;
	SWEEP_VAR

	/* inflow */
	/*voption indep*/
#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
#pragma omp parallel for private( ux, uy, uz, rho, ux1, uy1, uz1, rho1, \
                                  ux2, uy2, uz2, rho2, u2, px, py )
#endif
	SWEEP_START( 0, 0, 0, 0, 0, 1 )
		rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 1, WB );
		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, 2, WB );

		rho = 2.0*rho1 - rho2;

		px = (SWEEP_X / (0.5*(SIZE_X-1))) - 1.0;
		py = (SWEEP_Y / (0.5*(SIZE_Y-1))) - 1.0;
		ux = 0.00;
		uy = 0.00;
		uz = 0.01 * (1.0-px*px) * (1.0-py*py);

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0                                 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
	SWEEP_END

	/* outflow */
	/*voption indep*/
#if (defined(_OPENMP) || defined(SPEC_OPENMP)) && !defined(SPEC_SUPPRESS_OPENMP) && !defined(SPEC_AUTO_SUPPRESS_OPENMP)
#pragma omp parallel for private( ux, uy, uz, rho, ux1, uy1, uz1, rho1, \
                                  ux2, uy2, uz2, rho2, u2, px, py )
#endif
	SWEEP_START( 0, 0, SIZE_Z-1, 0, 0, SIZE_Z )
		rho1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		ux1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, W  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );
		uy1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, S  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NW )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB );
		uz1 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, B  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, NB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, SB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, EB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -1, WB );

		ux1 /= rho1;
		uy1 /= rho1;
		uz1 /= rho1;

		rho2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, C  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT )
		       + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		ux2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, E  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, W  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );
		uy2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, N  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, S  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NE ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NW )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SE ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SW )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		      - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB );
		uz2 = + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, T  ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, B  )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, NB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ST ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, SB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, ET ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, EB )
		      + GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WT ) - GRID_ENTRY_SWEEP( srcGrid, 0, 0, -2, WB );

		ux2 /= rho2;
		uy2 /= rho2;
		uz2 /= rho2;

		rho = 1.0;

		ux = 2*ux1 - ux2;
		uy = 2*uy1 - uy2;
		uz = 2*uz1 - uz2;

		u2 = 1.5 * (ux*ux + uy*uy + uz*uz);

		LOCAL( srcGrid, C ) = DFL1*rho*(1.0                                 - u2);

		LOCAL( srcGrid, N ) = DFL2*rho*(1.0 +       uy*(4.5*uy       + 3.0) - u2);
		LOCAL( srcGrid, S ) = DFL2*rho*(1.0 +       uy*(4.5*uy       - 3.0) - u2);
		LOCAL( srcGrid, E ) = DFL2*rho*(1.0 +       ux*(4.5*ux       + 3.0) - u2);
		LOCAL( srcGrid, W ) = DFL2*rho*(1.0 +       ux*(4.5*ux       - 3.0) - u2);
		LOCAL( srcGrid, T ) = DFL2*rho*(1.0 +       uz*(4.5*uz       + 3.0) - u2);
		LOCAL( srcGrid, B ) = DFL2*rho*(1.0 +       uz*(4.5*uz       - 3.0) - u2);

		LOCAL( srcGrid, NE) = DFL3*rho*(1.0 + (+ux+uy)*(4.5*(+ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, NW) = DFL3*rho*(1.0 + (-ux+uy)*(4.5*(-ux+uy) + 3.0) - u2);
		LOCAL( srcGrid, SE) = DFL3*rho*(1.0 + (+ux-uy)*(4.5*(+ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, SW) = DFL3*rho*(1.0 + (-ux-uy)*(4.5*(-ux-uy) + 3.0) - u2);
		LOCAL( srcGrid, NT) = DFL3*rho*(1.0 + (+uy+uz)*(4.5*(+uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, NB) = DFL3*rho*(1.0 + (+uy-uz)*(4.5*(+uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ST) = DFL3*rho*(1.0 + (-uy+uz)*(4.5*(-uy+uz) + 3.0) - u2);
		LOCAL( srcGrid, SB) = DFL3*rho*(1.0 + (-uy-uz)*(4.5*(-uy-uz) + 3.0) - u2);
		LOCAL( srcGrid, ET) = DFL3*rho*(1.0 + (+ux+uz)*(4.5*(+ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, EB) = DFL3*rho*(1.0 + (+ux-uz)*(4.5*(+ux-uz) + 3.0) - u2);
		LOCAL( srcGrid, WT) = DFL3*rho*(1.0 + (-ux+uz)*(4.5*(-ux+uz) + 3.0) - u2);
		LOCAL( srcGrid, WB) = DFL3*rho*(1.0 + (-ux-uz)*(4.5*(-ux-uz) + 3.0) - u2);
	SWEEP_END
}

/*############################################################################*/

void LBM_showGridStatistics( LBM_Grid grid ) {
	int nObstacleCells = 0,
	    nAccelCells    = 0,
	    nFluidCells    = 0;
	double ux, uy, uz;
	double minU2  = 1e+30, maxU2  = -1e+30, u2;
	double minRho = 1e+30, maxRho = -1e+30, rho;
	double mass = 0;

	SWEEP_VAR

	SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
		rho = + LOCAL( grid, C  ) + LOCAL( grid, N  )
		      + LOCAL( grid, S  ) + LOCAL( grid, E  )
		      + LOCAL( grid, W  ) + LOCAL( grid, T  )
		      + LOCAL( grid, B  ) + LOCAL( grid, NE )
		      + LOCAL( grid, NW ) + LOCAL( grid, SE )
		      + LOCAL( grid, SW ) + LOCAL( grid, NT )
		      + LOCAL( grid, NB ) + LOCAL( grid, ST )
		      + LOCAL( grid, SB ) + LOCAL( grid, ET )
		      + LOCAL( grid, EB ) + LOCAL( grid, WT )
		      + LOCAL( grid, WB );
        //printf("i:%d rho:%lf mass:%lf\n", i ,rho, mass);
		if( rho < minRho ) minRho = rho;
		if( rho > maxRho ) maxRho = rho;
		mass += rho;

		if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
			nObstacleCells++;
		}
		else {
			if( TEST_FLAG_SWEEP( grid, ACCEL ))
				nAccelCells++;
			else
				nFluidCells++;

			ux = + LOCAL( grid, E  ) - LOCAL( grid, W  )
			     + LOCAL( grid, NE ) - LOCAL( grid, NW )
			     + LOCAL( grid, SE ) - LOCAL( grid, SW )
			     + LOCAL( grid, ET ) + LOCAL( grid, EB )
			     - LOCAL( grid, WT ) - LOCAL( grid, WB );
			uy = + LOCAL( grid, N  ) - LOCAL( grid, S  )
			     + LOCAL( grid, NE ) + LOCAL( grid, NW )
			     - LOCAL( grid, SE ) - LOCAL( grid, SW )
			     + LOCAL( grid, NT ) + LOCAL( grid, NB )
			     - LOCAL( grid, ST ) - LOCAL( grid, SB );
			uz = + LOCAL( grid, T  ) - LOCAL( grid, B  )
			     + LOCAL( grid, NT ) - LOCAL( grid, NB )
			     + LOCAL( grid, ST ) - LOCAL( grid, SB )
			     + LOCAL( grid, ET ) - LOCAL( grid, EB )
			     + LOCAL( grid, WT ) - LOCAL( grid, WB );
			u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
			if( u2 < minU2 ) minU2 = u2;
			if( u2 > maxU2 ) maxU2 = u2;
		}
	SWEEP_END

	printf( "LBM_showGridStatistics:\n"
	        "\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
	        "\tminRho: %8.6f maxRho: %8.6f Mass: %e\n"
	        "\tminU  : %8.6f maxU  : %8.6f\n\n",
	        nObstacleCells, nAccelCells, nFluidCells,
	        minRho, maxRho, mass,
	        sqrt( minU2 ), sqrt( maxU2 ) );
}

void LBM_showGridStatistics_debug( LBM_Grid grid ) {
    int nObstacleCells = 0,
            nAccelCells    = 0,
            nFluidCells    = 0;
    double ux, uy, uz;
    double minU2  = 1e+30, maxU2  = -1e+30, u2;
    double minRho = 1e+30, maxRho = -1e+30, rho;
    double mass = 0;

    SWEEP_VAR

    SWEEP_START( 0, 0, 0, 0, 0, SIZE_Z )
        rho = + LOCAL( grid, C  ) + LOCAL( grid, N  )
              + LOCAL( grid, S  ) + LOCAL( grid, E  )
              + LOCAL( grid, W  ) + LOCAL( grid, T  )
              + LOCAL( grid, B  ) + LOCAL( grid, NE )
              + LOCAL( grid, NW ) + LOCAL( grid, SE )
              + LOCAL( grid, SW ) + LOCAL( grid, NT )
              + LOCAL( grid, NB ) + LOCAL( grid, ST )
              + LOCAL( grid, SB ) + LOCAL( grid, ET )
              + LOCAL( grid, EB ) + LOCAL( grid, WT )
              + LOCAL( grid, WB );
        printf("i:%d rho:%lf mass:%lf\n", i ,rho, mass);
        if( rho < minRho ) minRho = rho;
        if( rho > maxRho ) maxRho = rho;
        mass += rho;

        if( TEST_FLAG_SWEEP( grid, OBSTACLE )) {
            nObstacleCells++;
        }
        else {
            if( TEST_FLAG_SWEEP( grid, ACCEL ))
                nAccelCells++;
            else
                nFluidCells++;

            ux = + LOCAL( grid, E  ) - LOCAL( grid, W  )
                 + LOCAL( grid, NE ) - LOCAL( grid, NW )
                 + LOCAL( grid, SE ) - LOCAL( grid, SW )
                 + LOCAL( grid, ET ) + LOCAL( grid, EB )
                 - LOCAL( grid, WT ) - LOCAL( grid, WB );
            uy = + LOCAL( grid, N  ) - LOCAL( grid, S  )
                 + LOCAL( grid, NE ) + LOCAL( grid, NW )
                 - LOCAL( grid, SE ) - LOCAL( grid, SW )
                 + LOCAL( grid, NT ) + LOCAL( grid, NB )
                 - LOCAL( grid, ST ) - LOCAL( grid, SB );
            uz = + LOCAL( grid, T  ) - LOCAL( grid, B  )
                 + LOCAL( grid, NT ) - LOCAL( grid, NB )
                 + LOCAL( grid, ST ) - LOCAL( grid, SB )
                 + LOCAL( grid, ET ) - LOCAL( grid, EB )
                 + LOCAL( grid, WT ) - LOCAL( grid, WB );
            u2 = (ux*ux + uy*uy + uz*uz) / (rho*rho);
            if( u2 < minU2 ) minU2 = u2;
            if( u2 > maxU2 ) maxU2 = u2;
        }
    SWEEP_END

    printf( "LBM_showGridStatistics:\n"
            "\tnObstacleCells: %7i nAccelCells: %7i nFluidCells: %7i\n"
            "\tminRho: %8.6f maxRho: %8.6f Mass: %e\n"
            "\tminU  : %8.6f maxU  : %8.6f\n\n",
            nObstacleCells, nAccelCells, nFluidCells,
            minRho, maxRho, mass,
            sqrt( minU2 ), sqrt( maxU2 ) );
}
/*############################################################################*/

static void storeValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		const char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
#if !defined(SPEC)
		int i;
#else
               size_t i;
#endif

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			buffer[i] = vPtr[sizeof( OUTPUT_PRECISION ) - i - 1];

		fwrite( buffer, sizeof( OUTPUT_PRECISION ), 1, file );
	}
	else {                                                     /* little endian */
		fwrite( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

/*############################################################################*/

static void loadValue( FILE* file, OUTPUT_PRECISION* v ) {
	const int litteBigEndianTest = 1;
	if( (*((unsigned char*) &litteBigEndianTest)) == 0 ) {         /* big endian */
		char* vPtr = (char*) v;
		char buffer[sizeof( OUTPUT_PRECISION )];
#if !defined(SPEC)
		int i;
#else
               size_t i;
#endif

		fread( buffer, sizeof( OUTPUT_PRECISION ), 1, file );

		for (i = 0; i < sizeof( OUTPUT_PRECISION ); i++)
			vPtr[i] = buffer[sizeof( OUTPUT_PRECISION ) - i - 1];
	}
	else {                                                     /* little endian */
		fread( v, sizeof( OUTPUT_PRECISION ), 1, file );
	}
}

/*############################################################################*/

void LBM_storeVelocityField( LBM_Grid grid, const char* filename,
                             const int binary ) {
	int x, y, z;
	OUTPUT_PRECISION rho, ux, uy, uz;

	FILE* file = fopen( filename, (binary ? "wb" : "w") );

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				rho = + GRID_ENTRY( grid, x, y, z, C  ) + GRID_ENTRY( grid, x, y, z, N  )
				      + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
				      + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
				      + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
				      + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
				      + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
				      + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
				      + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
				      + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
				      + GRID_ENTRY( grid, x, y, z, WB );
				ux = + GRID_ENTRY( grid, x, y, z, E  ) - GRID_ENTRY( grid, x, y, z, W  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW ) 
				     + GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB ) 
				     - GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				uy = + GRID_ENTRY( grid, x, y, z, N  ) - GRID_ENTRY( grid, x, y, z, S  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW ) 
				     - GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB ) 
				     - GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
				uz = + GRID_ENTRY( grid, x, y, z, T  ) - GRID_ENTRY( grid, x, y, z, B  ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB ) 
				     + GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB ) 
				     + GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					/*
					fwrite( &ux, sizeof( ux ), 1, file );
					fwrite( &uy, sizeof( uy ), 1, file );
					fwrite( &uz, sizeof( uz ), 1, file );
					*/
					storeValue( file, &ux );
					storeValue( file, &uy );
					storeValue( file, &uz );
				} else
					fprintf( file, "%e %e %e\n", ux, uy, uz );

			}
		}
	}

	fclose( file );
}

/*############################################################################*/

void LBM_compareVelocityField( LBM_Grid grid, const char* filename,
                             const int binary ) {
	int x, y, z;
	double rho, ux, uy, uz;
	OUTPUT_PRECISION fileUx, fileUy, fileUz,
	                 dUx, dUy, dUz,
	                 diff2, maxDiff2 = -1e+30;

	FILE* file = fopen( filename, (binary ? "rb" : "r") );

	for( z = 0; z < SIZE_Z; z++ ) {
		for( y = 0; y < SIZE_Y; y++ ) {
			for( x = 0; x < SIZE_X; x++ ) {
				rho = + GRID_ENTRY( grid, x, y, z, C  ) + GRID_ENTRY( grid, x, y, z, N  )
				      + GRID_ENTRY( grid, x, y, z, S  ) + GRID_ENTRY( grid, x, y, z, E  )
				      + GRID_ENTRY( grid, x, y, z, W  ) + GRID_ENTRY( grid, x, y, z, T  )
				      + GRID_ENTRY( grid, x, y, z, B  ) + GRID_ENTRY( grid, x, y, z, NE )
				      + GRID_ENTRY( grid, x, y, z, NW ) + GRID_ENTRY( grid, x, y, z, SE )
				      + GRID_ENTRY( grid, x, y, z, SW ) + GRID_ENTRY( grid, x, y, z, NT )
				      + GRID_ENTRY( grid, x, y, z, NB ) + GRID_ENTRY( grid, x, y, z, ST )
				      + GRID_ENTRY( grid, x, y, z, SB ) + GRID_ENTRY( grid, x, y, z, ET )
				      + GRID_ENTRY( grid, x, y, z, EB ) + GRID_ENTRY( grid, x, y, z, WT )
				      + GRID_ENTRY( grid, x, y, z, WB );
				ux = + GRID_ENTRY( grid, x, y, z, E  ) - GRID_ENTRY( grid, x, y, z, W  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) - GRID_ENTRY( grid, x, y, z, NW ) 
				     + GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) + GRID_ENTRY( grid, x, y, z, EB ) 
				     - GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				uy = + GRID_ENTRY( grid, x, y, z, N  ) - GRID_ENTRY( grid, x, y, z, S  ) 
				     + GRID_ENTRY( grid, x, y, z, NE ) + GRID_ENTRY( grid, x, y, z, NW ) 
				     - GRID_ENTRY( grid, x, y, z, SE ) - GRID_ENTRY( grid, x, y, z, SW ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) + GRID_ENTRY( grid, x, y, z, NB ) 
				     - GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB );
				uz = + GRID_ENTRY( grid, x, y, z, T  ) - GRID_ENTRY( grid, x, y, z, B  ) 
				     + GRID_ENTRY( grid, x, y, z, NT ) - GRID_ENTRY( grid, x, y, z, NB ) 
				     + GRID_ENTRY( grid, x, y, z, ST ) - GRID_ENTRY( grid, x, y, z, SB ) 
				     + GRID_ENTRY( grid, x, y, z, ET ) - GRID_ENTRY( grid, x, y, z, EB ) 
				     + GRID_ENTRY( grid, x, y, z, WT ) - GRID_ENTRY( grid, x, y, z, WB );
				ux /= rho;
				uy /= rho;
				uz /= rho;

				if( binary ) {
					loadValue( file, &fileUx );
					loadValue( file, &fileUy );
					loadValue( file, &fileUz );
				}
				else {
#if !defined(SPEC)
					if( sizeof( OUTPUT_PRECISION ) == sizeof( double )) {
						fscanf( file, "%lf %lf %lf\n", &fileUx, &fileUy, &fileUz );
					}
					else {
#endif
						fscanf( file, "%f %f %f\n", &fileUx, &fileUy, &fileUz );
#if !defined(SPEC)
					}
#endif
				}

				dUx = ux - fileUx;
				dUy = uy - fileUy;
				dUz = uz - fileUz;
				diff2 = dUx*dUx + dUy*dUy + dUz*dUz;
				if( diff2 > maxDiff2 ) maxDiff2 = diff2;
			}
		}
	}

#ifdef SPEC
	printf( "LBM_compareVelocityField: maxDiff = %e  \n\n",
	        sqrt( maxDiff2 )  );
#else
	printf( "LBM_compareVelocityField: maxDiff = %e  ==>  %s\n\n",
	        sqrt( maxDiff2 ),
	        sqrt( maxDiff2 ) > 1e-5 ? "##### ERROR #####" : "OK" );
#endif
	fclose( file );
}

