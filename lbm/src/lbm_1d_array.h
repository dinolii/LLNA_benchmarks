/* $Id: lbm_1d_array.h,v 1.3 2004/05/03 19:45:58 larsonre Exp $ */

#ifndef _LBM_MACROS_H_
#define _LBM_MACROS_H_

/*############################################################################*/

typedef double LBM_Grid[SIZE_Z*SIZE_Y*SIZE_X*N_CELL_ENTRIES];
typedef LBM_Grid* LBM_GridPtr;

/*############################################################################*/

#define CALC_INDEX(x,y,z,e) ((e)+N_CELL_ENTRIES*((x)+ \
                             (y)*SIZE_X+(z)*SIZE_X*SIZE_Y))

#define SWEEP_VAR int i;

#define SWEEP_START(x1,y1,z1,x2,y2,z2) \
	for( i = CALC_INDEX(x1, y1, z1, 0); \
	     i < CALC_INDEX(x2, y2, z2, 0); \
			 i += N_CELL_ENTRIES ) {

#define SWEEP_END }


#define SWEEP_2START(x1,y1,z1,x2,y2,z2) \
	for( i = CALC_INDEX(x1, y1, z1, 0); \
	     i < CALC_INDEX(x2, y2, z2, 0); \
			 i += 2*N_CELL_ENTRIES ) {

#define SWEEP_4START(x1,y1,z1,x2,y2,z2) \
	for( i = CALC_INDEX(x1, y1, z1, 0); \
	     i < CALC_INDEX(x2, y2, z2, 0); \
			 i += 4*N_CELL_ENTRIES ) {

#define SWEEP_2STARTV1(x1,y1,z1,x2,y2,z2) \
	for( i = CALC_INDEX(x1, y1, z1, 0); \
	     i < CALC_INDEX(x2, y2, z2, 0); \
		  ) {

#define SWEEP_X  ((i / N_CELL_ENTRIES) % SIZE_X)
#define SWEEP_Y (((i / N_CELL_ENTRIES) / SIZE_X) % SIZE_Y)
#define SWEEP_Z  ((i / N_CELL_ENTRIES) / (SIZE_X*SIZE_Y))

#define GRID_ENTRY(g,x,y,z,e)          ((g)[CALC_INDEX( x,  y,  z, e)])
#define GRID_ENTRY_SWEEP(g,dx,dy,dz,e) ((g)[CALC_INDEX(dx, dy, dz, e)+(i)])

#define GRID_ENTRY_SWEEP_NEXT(g,dx,dy,dz,e) ((g)[(CALC_INDEX(dx, dy, dz, e)+(i+N_CELL_ENTRIES))])

#define GRID_ENTRY_SWEEP_NEXT2(g,dx,dy,dz,e) ((g)[(CALC_INDEX(dx, dy, dz, e)+(i+2*N_CELL_ENTRIES))])
#define GRID_ENTRY_SWEEP_NEXT3(g,dx,dy,dz,e) ((g)[(CALC_INDEX(dx, dy, dz, e)+(i+3*N_CELL_ENTRIES))])


#define LOCAL(g,e)       (GRID_ENTRY_SWEEP( g,  0,  0,  0, e ))
#define NEIGHBOR_C(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0,  0, e ))
#define NEIGHBOR_N(g,e)  (GRID_ENTRY_SWEEP( g,  0, +1,  0, e ))
#define NEIGHBOR_S(g,e)  (GRID_ENTRY_SWEEP( g,  0, -1,  0, e ))
#define NEIGHBOR_E(g,e)  (GRID_ENTRY_SWEEP( g, +1,  0,  0, e ))
#define NEIGHBOR_W(g,e)  (GRID_ENTRY_SWEEP( g, -1,  0,  0, e ))
#define NEIGHBOR_T(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0, +1, e ))
#define NEIGHBOR_B(g,e)  (GRID_ENTRY_SWEEP( g,  0,  0, -1, e ))
#define NEIGHBOR_NE(g,e) (GRID_ENTRY_SWEEP( g, +1, +1,  0, e ))
#define NEIGHBOR_NW(g,e) (GRID_ENTRY_SWEEP( g, -1, +1,  0, e ))
#define NEIGHBOR_SE(g,e) (GRID_ENTRY_SWEEP( g, +1, -1,  0, e ))
#define NEIGHBOR_SW(g,e) (GRID_ENTRY_SWEEP( g, -1, -1,  0, e ))
#define NEIGHBOR_NT(g,e) (GRID_ENTRY_SWEEP( g,  0, +1, +1, e ))
#define NEIGHBOR_NB(g,e) (GRID_ENTRY_SWEEP( g,  0, +1, -1, e ))
#define NEIGHBOR_ST(g,e) (GRID_ENTRY_SWEEP( g,  0, -1, +1, e ))
#define NEIGHBOR_SB(g,e) (GRID_ENTRY_SWEEP( g,  0, -1, -1, e ))
#define NEIGHBOR_ET(g,e) (GRID_ENTRY_SWEEP( g, +1,  0, +1, e ))
#define NEIGHBOR_EB(g,e) (GRID_ENTRY_SWEEP( g, +1,  0, -1, e ))
#define NEIGHBOR_WT(g,e) (GRID_ENTRY_SWEEP( g, -1,  0, +1, e ))
#define NEIGHBOR_WB(g,e) (GRID_ENTRY_SWEEP( g, -1,  0, -1, e ))

#define LOCAL_NEXT(g,e)       (GRID_ENTRY_SWEEP_NEXT(g, 0, 0, 0, e))
#define LOCAL_NEXT2(g,e)       (GRID_ENTRY_SWEEP_NEXT2(g, 0, 0, 0, e))
#define LOCAL_NEXT3(g,e)       (GRID_ENTRY_SWEEP_NEXT3(g, 0, 0, 0, e))

#define NEIGHBOR_C_NEXT(g,e)  (GRID_ENTRY_SWEEP_NEXT( g,  0,  0,  0, e ))
#define NEIGHBOR_N_NEXT(g,e)  (GRID_ENTRY_SWEEP_NEXT( g,  0, +1,  0, e ))
#define NEIGHBOR_S_NEXT(g,e)  (GRID_ENTRY_SWEEP_NEXT( g,  0, -1,  0, e ))
#define NEIGHBOR_E_NEXT(g,e)  (GRID_ENTRY_SWEEP_NEXT( g, +1,  0,  0, e ))
#define NEIGHBOR_W_NEXT(g,e)  (GRID_ENTRY_SWEEP_NEXT( g, -1,  0,  0, e ))
#define NEIGHBOR_T_NEXT(g,e)  (GRID_ENTRY_SWEEP_NEXT( g,  0,  0, +1, e ))
#define NEIGHBOR_B_NEXT(g,e)  (GRID_ENTRY_SWEEP_NEXT( g,  0,  0, -1, e ))
#define NEIGHBOR_NE_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g, +1, +1,  0, e ))
#define NEIGHBOR_NW_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g, -1, +1,  0, e ))
#define NEIGHBOR_SE_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g, +1, -1,  0, e ))
#define NEIGHBOR_SW_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g, -1, -1,  0, e ))
#define NEIGHBOR_NT_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g,  0, +1, +1, e ))
#define NEIGHBOR_NB_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g,  0, +1, -1, e ))
#define NEIGHBOR_ST_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g,  0, -1, +1, e ))
#define NEIGHBOR_SB_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g,  0, -1, -1, e ))
#define NEIGHBOR_ET_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g, +1,  0, +1, e ))
#define NEIGHBOR_EB_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g, +1,  0, -1, e ))
#define NEIGHBOR_WT_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g, -1,  0, +1, e ))
#define NEIGHBOR_WB_NEXT(g,e) (GRID_ENTRY_SWEEP_NEXT( g, -1,  0, -1, e ))

#define NEIGHBOR_C_NEXT2(g,e)  (GRID_ENTRY_SWEEP_NEXT2( g,  0,  0,  0, e ))
#define NEIGHBOR_N_NEXT2(g,e)  (GRID_ENTRY_SWEEP_NEXT2( g,  0, +1,  0, e ))
#define NEIGHBOR_S_NEXT2(g,e)  (GRID_ENTRY_SWEEP_NEXT2( g,  0, -1,  0, e ))
#define NEIGHBOR_E_NEXT2(g,e)  (GRID_ENTRY_SWEEP_NEXT2( g, +1,  0,  0, e ))
#define NEIGHBOR_W_NEXT2(g,e)  (GRID_ENTRY_SWEEP_NEXT2( g, -1,  0,  0, e ))
#define NEIGHBOR_T_NEXT2(g,e)  (GRID_ENTRY_SWEEP_NEXT2( g,  0,  0, +1, e ))
#define NEIGHBOR_B_NEXT2(g,e)  (GRID_ENTRY_SWEEP_NEXT2( g,  0,  0, -1, e ))
#define NEIGHBOR_NE_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g, +1, +1,  0, e ))
#define NEIGHBOR_NW_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g, -1, +1,  0, e ))
#define NEIGHBOR_SE_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g, +1, -1,  0, e ))
#define NEIGHBOR_SW_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g, -1, -1,  0, e ))
#define NEIGHBOR_NT_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g,  0, +1, +1, e ))
#define NEIGHBOR_NB_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g,  0, +1, -1, e ))
#define NEIGHBOR_ST_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g,  0, -1, +1, e ))
#define NEIGHBOR_SB_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g,  0, -1, -1, e ))
#define NEIGHBOR_ET_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g, +1,  0, +1, e ))
#define NEIGHBOR_EB_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g, +1,  0, -1, e ))
#define NEIGHBOR_WT_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g, -1,  0, +1, e ))
#define NEIGHBOR_WB_NEXT2(g,e) (GRID_ENTRY_SWEEP_NEXT2( g, -1,  0, -1, e ))

#define NEIGHBOR_C_NEXT3(g,e)  (GRID_ENTRY_SWEEP_NEXT3( g,  0,  0,  0, e ))
#define NEIGHBOR_N_NEXT3(g,e)  (GRID_ENTRY_SWEEP_NEXT3( g,  0, +1,  0, e ))
#define NEIGHBOR_S_NEXT3(g,e)  (GRID_ENTRY_SWEEP_NEXT3( g,  0, -1,  0, e ))
#define NEIGHBOR_E_NEXT3(g,e)  (GRID_ENTRY_SWEEP_NEXT3( g, +1,  0,  0, e ))
#define NEIGHBOR_W_NEXT3(g,e)  (GRID_ENTRY_SWEEP_NEXT3( g, -1,  0,  0, e ))
#define NEIGHBOR_T_NEXT3(g,e)  (GRID_ENTRY_SWEEP_NEXT3( g,  0,  0, +1, e ))
#define NEIGHBOR_B_NEXT3(g,e)  (GRID_ENTRY_SWEEP_NEXT3( g,  0,  0, -1, e ))
#define NEIGHBOR_NE_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g, +1, +1,  0, e ))
#define NEIGHBOR_NW_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g, -1, +1,  0, e ))
#define NEIGHBOR_SE_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g, +1, -1,  0, e ))
#define NEIGHBOR_SW_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g, -1, -1,  0, e ))
#define NEIGHBOR_NT_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g,  0, +1, +1, e ))
#define NEIGHBOR_NB_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g,  0, +1, -1, e ))
#define NEIGHBOR_ST_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g,  0, -1, +1, e ))
#define NEIGHBOR_SB_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g,  0, -1, -1, e ))
#define NEIGHBOR_ET_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g, +1,  0, +1, e ))
#define NEIGHBOR_EB_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g, +1,  0, -1, e ))
#define NEIGHBOR_WT_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g, -1,  0, +1, e ))
#define NEIGHBOR_WB_NEXT3(g,e) (GRID_ENTRY_SWEEP_NEXT3( g, -1,  0, -1, e ))

#define COLLIDE_STREAM
#ifdef COLLIDE_STREAM

#define SRC_C(g)  (LOCAL( g, C  ))
#define SRC_N(g)  (LOCAL( g, N  ))
#define SRC_S(g)  (LOCAL( g, S  ))
#define SRC_E(g)  (LOCAL( g, E  ))
#define SRC_W(g)  (LOCAL( g, W  ))
#define SRC_T(g)  (LOCAL( g, T  ))
#define SRC_B(g)  (LOCAL( g, B  ))
#define SRC_NE(g) (LOCAL( g, NE ))
#define SRC_NW(g) (LOCAL( g, NW ))
#define SRC_SE(g) (LOCAL( g, SE ))
#define SRC_SW(g) (LOCAL( g, SW ))
#define SRC_NT(g) (LOCAL( g, NT ))
#define SRC_NB(g) (LOCAL( g, NB ))
#define SRC_ST(g) (LOCAL( g, ST ))
#define SRC_SB(g) (LOCAL( g, SB ))
#define SRC_ET(g) (LOCAL( g, ET ))
#define SRC_EB(g) (LOCAL( g, EB ))
#define SRC_WT(g) (LOCAL( g, WT ))
#define SRC_WB(g) (LOCAL( g, WB ))


#define SRC_C_NEXT(g)  (LOCAL_NEXT( g, C  ))
#define SRC_N_NEXT(g)  (LOCAL_NEXT( g, N  ))
#define SRC_S_NEXT(g)  (LOCAL_NEXT( g, S  ))
#define SRC_E_NEXT(g)  (LOCAL_NEXT( g, E  ))
#define SRC_W_NEXT(g)  (LOCAL_NEXT( g, W  ))
#define SRC_T_NEXT(g)  (LOCAL_NEXT( g, T  ))
#define SRC_B_NEXT(g)  (LOCAL_NEXT( g, B  ))
#define SRC_NE_NEXT(g) (LOCAL_NEXT( g, NE ))
#define SRC_NW_NEXT(g) (LOCAL_NEXT( g, NW ))
#define SRC_SE_NEXT(g) (LOCAL_NEXT( g, SE ))
#define SRC_SW_NEXT(g) (LOCAL_NEXT( g, SW ))
#define SRC_NT_NEXT(g) (LOCAL_NEXT( g, NT ))
#define SRC_NB_NEXT(g) (LOCAL_NEXT( g, NB ))
#define SRC_ST_NEXT(g) (LOCAL_NEXT( g, ST ))
#define SRC_SB_NEXT(g) (LOCAL_NEXT( g, SB ))
#define SRC_ET_NEXT(g) (LOCAL_NEXT( g, ET ))
#define SRC_EB_NEXT(g) (LOCAL_NEXT( g, EB ))
#define SRC_WT_NEXT(g) (LOCAL_NEXT( g, WT ))
#define SRC_WB_NEXT(g) (LOCAL_NEXT( g, WB ))

#define SRC_C_NEXT2(g)  (LOCAL_NEXT2( g, C  ))
#define SRC_N_NEXT2(g)  (LOCAL_NEXT2( g, N  ))
#define SRC_S_NEXT2(g)  (LOCAL_NEXT2( g, S  ))
#define SRC_E_NEXT2(g)  (LOCAL_NEXT2( g, E  ))
#define SRC_W_NEXT2(g)  (LOCAL_NEXT2( g, W  ))
#define SRC_T_NEXT2(g)  (LOCAL_NEXT2( g, T  ))
#define SRC_B_NEXT2(g)  (LOCAL_NEXT2( g, B  ))
#define SRC_NE_NEXT2(g) (LOCAL_NEXT2( g, NE ))
#define SRC_NW_NEXT2(g) (LOCAL_NEXT2( g, NW ))
#define SRC_SE_NEXT2(g) (LOCAL_NEXT2( g, SE ))
#define SRC_SW_NEXT2(g) (LOCAL_NEXT2( g, SW ))
#define SRC_NT_NEXT2(g) (LOCAL_NEXT2( g, NT ))
#define SRC_NB_NEXT2(g) (LOCAL_NEXT2( g, NB ))
#define SRC_ST_NEXT2(g) (LOCAL_NEXT2( g, ST ))
#define SRC_SB_NEXT2(g) (LOCAL_NEXT2( g, SB ))
#define SRC_ET_NEXT2(g) (LOCAL_NEXT2( g, ET ))
#define SRC_EB_NEXT2(g) (LOCAL_NEXT2( g, EB ))
#define SRC_WT_NEXT2(g) (LOCAL_NEXT2( g, WT ))
#define SRC_WB_NEXT2(g) (LOCAL_NEXT2( g, WB ))

#define SRC_C_NEXT3(g)  (LOCAL_NEXT3( g, C  ))
#define SRC_N_NEXT3(g)  (LOCAL_NEXT3( g, N  ))
#define SRC_S_NEXT3(g)  (LOCAL_NEXT3( g, S  ))
#define SRC_E_NEXT3(g)  (LOCAL_NEXT3( g, E  ))
#define SRC_W_NEXT3(g)  (LOCAL_NEXT3( g, W  ))
#define SRC_T_NEXT3(g)  (LOCAL_NEXT3( g, T  ))
#define SRC_B_NEXT3(g)  (LOCAL_NEXT3( g, B  ))
#define SRC_NE_NEXT3(g) (LOCAL_NEXT3( g, NE ))
#define SRC_NW_NEXT3(g) (LOCAL_NEXT3( g, NW ))
#define SRC_SE_NEXT3(g) (LOCAL_NEXT3( g, SE ))
#define SRC_SW_NEXT3(g) (LOCAL_NEXT3( g, SW ))
#define SRC_NT_NEXT3(g) (LOCAL_NEXT3( g, NT ))
#define SRC_NB_NEXT3(g) (LOCAL_NEXT3( g, NB ))
#define SRC_ST_NEXT3(g) (LOCAL_NEXT3( g, ST ))
#define SRC_SB_NEXT3(g) (LOCAL_NEXT3( g, SB ))
#define SRC_ET_NEXT3(g) (LOCAL_NEXT3( g, ET ))
#define SRC_EB_NEXT3(g) (LOCAL_NEXT3( g, EB ))
#define SRC_WT_NEXT3(g) (LOCAL_NEXT3( g, WT ))
#define SRC_WB_NEXT3(g) (LOCAL_NEXT3( g, WB ))

#define DST_C(g)  (NEIGHBOR_C ( g, C  ))
#define DST_N(g)  (NEIGHBOR_N ( g, N  ))
#define DST_S(g)  (NEIGHBOR_S ( g, S  ))
#define DST_E(g)  (NEIGHBOR_E ( g, E  ))
#define DST_W(g)  (NEIGHBOR_W ( g, W  ))
#define DST_T(g)  (NEIGHBOR_T ( g, T  ))
#define DST_B(g)  (NEIGHBOR_B ( g, B  ))
#define DST_NE(g) (NEIGHBOR_NE( g, NE ))
#define DST_NW(g) (NEIGHBOR_NW( g, NW ))
#define DST_SE(g) (NEIGHBOR_SE( g, SE ))
#define DST_SW(g) (NEIGHBOR_SW( g, SW ))
#define DST_NT(g) (NEIGHBOR_NT( g, NT ))
#define DST_NB(g) (NEIGHBOR_NB( g, NB ))
#define DST_ST(g) (NEIGHBOR_ST( g, ST ))
#define DST_SB(g) (NEIGHBOR_SB( g, SB ))
#define DST_ET(g) (NEIGHBOR_ET( g, ET ))
#define DST_EB(g) (NEIGHBOR_EB( g, EB ))
#define DST_WT(g) (NEIGHBOR_WT( g, WT ))
#define DST_WB(g) (NEIGHBOR_WB( g, WB ))


#define DST_C_NEXT(g)  (NEIGHBOR_C_NEXT ( g, C  ))
#define DST_N_NEXT(g)  (NEIGHBOR_N_NEXT ( g, N  ))
#define DST_S_NEXT(g)  (NEIGHBOR_S_NEXT ( g, S  ))
#define DST_E_NEXT(g)  (NEIGHBOR_E_NEXT ( g, E  ))
#define DST_W_NEXT(g)  (NEIGHBOR_W_NEXT ( g, W  ))
#define DST_T_NEXT(g)  (NEIGHBOR_T_NEXT ( g, T  ))
#define DST_B_NEXT(g)  (NEIGHBOR_B_NEXT ( g, B  ))
#define DST_NE_NEXT(g) (NEIGHBOR_NE_NEXT( g, NE ))
#define DST_NW_NEXT(g) (NEIGHBOR_NW_NEXT( g, NW ))
#define DST_SE_NEXT(g) (NEIGHBOR_SE_NEXT( g, SE ))
#define DST_SW_NEXT(g) (NEIGHBOR_SW_NEXT( g, SW ))
#define DST_NT_NEXT(g) (NEIGHBOR_NT_NEXT( g, NT ))
#define DST_NB_NEXT(g) (NEIGHBOR_NB_NEXT( g, NB ))
#define DST_ST_NEXT(g) (NEIGHBOR_ST_NEXT( g, ST ))
#define DST_SB_NEXT(g) (NEIGHBOR_SB_NEXT( g, SB ))
#define DST_ET_NEXT(g) (NEIGHBOR_ET_NEXT( g, ET ))
#define DST_EB_NEXT(g) (NEIGHBOR_EB_NEXT( g, EB ))
#define DST_WT_NEXT(g) (NEIGHBOR_WT_NEXT( g, WT ))
#define DST_WB_NEXT(g) (NEIGHBOR_WB_NEXT( g, WB ))

#define DST_C_NEXT2(g)  (NEIGHBOR_C_NEXT2 ( g, C  ))
#define DST_N_NEXT2(g)  (NEIGHBOR_N_NEXT2 ( g, N  ))
#define DST_S_NEXT2(g)  (NEIGHBOR_S_NEXT2 ( g, S  ))
#define DST_E_NEXT2(g)  (NEIGHBOR_E_NEXT2 ( g, E  ))
#define DST_W_NEXT2(g)  (NEIGHBOR_W_NEXT2 ( g, W  ))
#define DST_T_NEXT2(g)  (NEIGHBOR_T_NEXT2 ( g, T  ))
#define DST_B_NEXT2(g)  (NEIGHBOR_B_NEXT2 ( g, B  ))
#define DST_NE_NEXT2(g) (NEIGHBOR_NE_NEXT2( g, NE ))
#define DST_NW_NEXT2(g) (NEIGHBOR_NW_NEXT2( g, NW ))
#define DST_SE_NEXT2(g) (NEIGHBOR_SE_NEXT2( g, SE ))
#define DST_SW_NEXT2(g) (NEIGHBOR_SW_NEXT2( g, SW ))
#define DST_NT_NEXT2(g) (NEIGHBOR_NT_NEXT2( g, NT ))
#define DST_NB_NEXT2(g) (NEIGHBOR_NB_NEXT2( g, NB ))
#define DST_ST_NEXT2(g) (NEIGHBOR_ST_NEXT2( g, ST ))
#define DST_SB_NEXT2(g) (NEIGHBOR_SB_NEXT2( g, SB ))
#define DST_ET_NEXT2(g) (NEIGHBOR_ET_NEXT2( g, ET ))
#define DST_EB_NEXT2(g) (NEIGHBOR_EB_NEXT2( g, EB ))
#define DST_WT_NEXT2(g) (NEIGHBOR_WT_NEXT2( g, WT ))
#define DST_WB_NEXT2(g) (NEIGHBOR_WB_NEXT2( g, WB ))

#define DST_C_NEXT3(g)  (NEIGHBOR_C_NEXT3 ( g, C  ))
#define DST_N_NEXT3(g)  (NEIGHBOR_N_NEXT3 ( g, N  ))
#define DST_S_NEXT3(g)  (NEIGHBOR_S_NEXT3 ( g, S  ))
#define DST_E_NEXT3(g)  (NEIGHBOR_E_NEXT3 ( g, E  ))
#define DST_W_NEXT3(g)  (NEIGHBOR_W_NEXT3 ( g, W  ))
#define DST_T_NEXT3(g)  (NEIGHBOR_T_NEXT3 ( g, T  ))
#define DST_B_NEXT3(g)  (NEIGHBOR_B_NEXT3 ( g, B  ))
#define DST_NE_NEXT3(g) (NEIGHBOR_NE_NEXT3( g, NE ))
#define DST_NW_NEXT3(g) (NEIGHBOR_NW_NEXT3( g, NW ))
#define DST_SE_NEXT3(g) (NEIGHBOR_SE_NEXT3( g, SE ))
#define DST_SW_NEXT3(g) (NEIGHBOR_SW_NEXT3( g, SW ))
#define DST_NT_NEXT3(g) (NEIGHBOR_NT_NEXT3( g, NT ))
#define DST_NB_NEXT3(g) (NEIGHBOR_NB_NEXT3( g, NB ))
#define DST_ST_NEXT3(g) (NEIGHBOR_ST_NEXT3( g, ST ))
#define DST_SB_NEXT3(g) (NEIGHBOR_SB_NEXT3( g, SB ))
#define DST_ET_NEXT3(g) (NEIGHBOR_ET_NEXT3( g, ET ))
#define DST_EB_NEXT3(g) (NEIGHBOR_EB_NEXT3( g, EB ))
#define DST_WT_NEXT3(g) (NEIGHBOR_WT_NEXT3( g, WT ))
#define DST_WB_NEXT3(g) (NEIGHBOR_WB_NEXT3( g, WB ))


#else /* COLLIDE_STREAM */

#define SRC_C(g)  (NEIGHBOR_C ( g, C  ))
#define SRC_N(g)  (NEIGHBOR_S ( g, N  ))
#define SRC_S(g)  (NEIGHBOR_N ( g, S  ))
#define SRC_E(g)  (NEIGHBOR_W ( g, E  ))
#define SRC_W(g)  (NEIGHBOR_E ( g, W  ))
#define SRC_T(g)  (NEIGHBOR_B ( g, T  ))
#define SRC_B(g)  (NEIGHBOR_T ( g, B  ))
#define SRC_NE(g) (NEIGHBOR_SW( g, NE ))
#define SRC_NW(g) (NEIGHBOR_SE( g, NW ))
#define SRC_SE(g) (NEIGHBOR_NW( g, SE ))
#define SRC_SW(g) (NEIGHBOR_NE( g, SW ))
#define SRC_NT(g) (NEIGHBOR_SB( g, NT ))
#define SRC_NB(g) (NEIGHBOR_ST( g, NB ))
#define SRC_ST(g) (NEIGHBOR_NB( g, ST ))
#define SRC_SB(g) (NEIGHBOR_NT( g, SB ))
#define SRC_ET(g) (NEIGHBOR_WB( g, ET ))
#define SRC_EB(g) (NEIGHBOR_WT( g, EB ))
#define SRC_WT(g) (NEIGHBOR_EB( g, WT ))
#define SRC_WB(g) (NEIGHBOR_ET( g, WB ))

#define DST_C(g)  (LOCAL( g, C  ))
#define DST_N(g)  (LOCAL( g, N  ))
#define DST_S(g)  (LOCAL( g, S  ))
#define DST_E(g)  (LOCAL( g, E  ))
#define DST_W(g)  (LOCAL( g, W  ))
#define DST_T(g)  (LOCAL( g, T  ))
#define DST_B(g)  (LOCAL( g, B  ))
#define DST_NE(g) (LOCAL( g, NE ))
#define DST_NW(g) (LOCAL( g, NW ))
#define DST_SE(g) (LOCAL( g, SE ))
#define DST_SW(g) (LOCAL( g, SW ))
#define DST_NT(g) (LOCAL( g, NT ))
#define DST_NB(g) (LOCAL( g, NB ))
#define DST_ST(g) (LOCAL( g, ST ))
#define DST_SB(g) (LOCAL( g, SB ))
#define DST_ET(g) (LOCAL( g, ET ))
#define DST_EB(g) (LOCAL( g, EB ))
#define DST_WT(g) (LOCAL( g, WT ))
#define DST_WB(g) (LOCAL( g, WB ))

#endif /* COLLIDE_STREAM */

#define MAGIC_CAST(v) ((unsigned int*) ((void*) (&(v))))
#define FLAG_VAR(v) unsigned int* const _aux_ = MAGIC_CAST(v)

#define TEST_FLAG_SWEEP_NEXT(g,f)     ((*MAGIC_CAST(LOCAL_NEXT(g, FLAGS))) & (f))
#define TEST_FLAG_SWEEP_NEXT2(g,f)     ((*MAGIC_CAST(LOCAL_NEXT2(g, FLAGS))) & (f))
#define TEST_FLAG_SWEEP_NEXT3(g,f)     ((*MAGIC_CAST(LOCAL_NEXT3(g, FLAGS))) & (f))


#define TEST_FLAG_SWEEP(g,f)     ((*MAGIC_CAST(LOCAL(g, FLAGS))) & (f))
#define SET_FLAG_SWEEP(g,f)      {FLAG_VAR(LOCAL(g, FLAGS)); (*_aux_) |=  (f);}
#define CLEAR_FLAG_SWEEP(g,f)    {FLAG_VAR(LOCAL(g, FLAGS)); (*_aux_) &= ~(f);}
#define CLEAR_ALL_FLAGS_SWEEP(g) {FLAG_VAR(LOCAL(g, FLAGS)); (*_aux_)  =    0;}

#define TEST_FLAG(g,x,y,z,f)     ((*MAGIC_CAST(GRID_ENTRY(g, x, y, z, FLAGS))) & (f))
#define SET_FLAG(g,x,y,z,f)      {FLAG_VAR(GRID_ENTRY(g, x, y, z, FLAGS)); (*_aux_) |=  (f);}
#define CLEAR_FLAG(g,x,y,z,f)    {FLAG_VAR(GRID_ENTRY(g, x, y, z, FLAGS)); (*_aux_) &= ~(f);}
#define CLEAR_ALL_FLAGS(g,x,y,z) {FLAG_VAR(GRID_ENTRY(g, x, y, z, FLAGS)); (*_aux_)  =    0;}

/*############################################################################*/

#endif /* _LBM_MACROS_H_ */

