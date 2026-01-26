/*
	This file is part of the Snoopy code.

    Snoopy code is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Snoopy code is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Snoopy code.  If not, see <http://www.gnu.org/licenses/>.
*/



#include "common.h"
#include "gfft.h"
#include "output/output_dump.h"
#include "symmetries.h"

#include "debug.h"



/** Allow one to init a structure in real space using ordinary defined x,y,z coordinates */

void init_SpatialStructure(struct Field fldi) {
	double *x,*y,*z;
	int i,j,k;

	/*******************************************************************
	** This part does not need to be modified **************************
	********************************************************************/
	// Allocate coordinate arrays
	x = (double *) fftw_malloc( sizeof(double complex) * NTOTAL_COMPLEX);
	if (x == NULL) ERROR_HANDLER( ERROR_CRITICAL, "No memory for x allocation");

	y = (double *) fftw_malloc( sizeof(double complex) * NTOTAL_COMPLEX);
	if (y == NULL) ERROR_HANDLER( ERROR_CRITICAL, "No memory for y allocation");

	z = (double *) fftw_malloc( sizeof(double complex) * NTOTAL_COMPLEX);
	if (z == NULL) ERROR_HANDLER( ERROR_CRITICAL, "No memory for z allocation");

	// Initialize the arrays
	// MPI_Printf("NZ = %d \n", NZ);
  #ifndef WITH_2D
	for(i = 0 ; i < NX/NPROC ; i++) {
		for(j = 0 ; j < NY ; j++) {
			for(k = 0 ; k < NZ ; k++) {
				x[k + (NZ + 2) * j + (NZ + 2) * NY * i] = - param.lx / 2 + (param.lx * (i + rank * NX / NPROC)) / NX;
				y[k + (NZ + 2) * j + (NZ + 2) * NY * i] = - param.ly / 2 + (param.ly * j ) / NY;
				z[k + (NZ + 2) * j + (NZ + 2) * NY * i] = - param.lz / 2 + (param.lz * k ) / NZ;
			}
		}
	}
	#else
	for(i = 0 ; i < NX/NPROC ; i++) {
		for(j = 0 ; j < NY ; j++) {
			for(k = 0 ; k < NZ ; k++) {
				x[k + (NZ) * j + (NZ) * (NY+2) * i] = - param.lx / 2 + (param.lx * (i + rank * NX / NPROC)) / NX;
				y[k + (NZ) * j + (NZ) * (NY+2) * i] = - param.ly / 2 + (param.ly * j ) / NY;
				// MPI_Printf("x = %.4f \t y = %.4f \n", x[k + (NZ) * j + (NZ) * (NY+2) * i],y[k + (NZ) * j + (NZ) * (NY+2) * i]);
				// MPI_Printf("idx = %d , i= %d, j=%d, k=%d \n", k + (NZ) * j + (NZ) * (NY+2) * i,i,j,k);
				z[k + (NZ) * j + (NZ) * (NY+2) * i] = - param.lz / 2 + (param.lz * k ) / NZ;
			}
		}
	}
  #endif
	// Initialize the extra points (k=NZ and k=NZ+1) to zero to prevent stupid things from happening...
	#ifndef WITH_2D
	for(i = 0 ; i < NX/NPROC ; i++) {
		for(j = 0 ; j < NY ; j++) {
			for(k = NZ ; k < NZ + 2 ; k++) {
				x[k + (NZ + 2) * j + (NZ + 2) * NY * i] = 0.0;
				y[k + (NZ + 2) * j + (NZ + 2) * NY * i] = 0.0;
				z[k + (NZ + 2) * j + (NZ + 2) * NY * i] = 0.0;
			}
		}
	}
	#else
	for(i = 0 ; i < NX/NPROC ; i++) {
		for(j = NY ; j < NY + 2; j++) {
			for(k = 0 ; k < NZ ; k++) {
				x[k + (NZ ) * j + (NZ ) * (NY + 2) * i] = 0.0;
				y[k + (NZ ) * j + (NZ ) * (NY + 2) * i] = 0.0;
				z[k + (NZ ) * j + (NZ ) * (NY + 2) * i] = 0.0;
			}
		}
	}
	#endif

	// Init work array to zero
	#ifndef WITH_2D
	for(i = 0 ; i < NX/NPROC ; i++) {
		for(j = 0 ; j < NY ; j++) {
			for(k = 0 ; k < NZ + 2 ; k++) {
				wr1[k + (NZ + 2) * j + (NZ + 2) * NY * i] = 0.0;
				wr2[k + (NZ + 2) * j + (NZ + 2) * NY * i] = 0.0;
				wr3[k + (NZ + 2) * j + (NZ + 2) * NY * i] = 0.0;
				wr4[k + (NZ + 2) * j + (NZ + 2) * NY * i] = 0.0;
				wr5[k + (NZ + 2) * j + (NZ + 2) * NY * i] = 0.0;
				wr6[k + (NZ + 2) * j + (NZ + 2) * NY * i] = 0.0;
			}
		}
	}
	#else
	for(i = 0 ; i < NX/NPROC ; i++) {
		for(j = 0 ; j < NY ; j++) {
			for(k = 0 ; k < NZ ; k++) {
				wr1[k + (NZ) * j + (NZ) * (NY+2) * i] = 0.0;
				wr2[k + (NZ) * j + (NZ) * (NY+2) * i] = 0.0;
				wr3[k + (NZ) * j + (NZ) * (NY+2) * i] = 0.0;
				wr4[k + (NZ) * j + (NZ) * (NY+2) * i] = 0.0;
				wr5[k + (NZ) * j + (NZ) * (NY+2) * i] = 0.0;
				wr6[k + (NZ) * j + (NZ) * (NY+2) * i] = 0.0;
			}
		}
	}
	#endif
	/*******************************************************************
	** This part can be modified              **************************
	********************************************************************/

	// The velocity field vx,vy,vz is stored in wr1,wr2,wr3
	// The magnetic field bx,by,bz is stored in wr4,wr5,wr6 (ignored if MHD is not set)
	double sigma  = 0.376063;
	double vA2    = param.bz0*param.bz0;
	double kZ     = 4.0*2.0*M_PI;
	double kX     = 2.0*2.0*M_PI;
	double sigma_eta  = sigma + (kX*kX + kZ*kZ)/param.reynolds_m;
	double sigma_nu   = sigma + (kX*kX + kZ*kZ)/param.reynolds;

	for(i = 0 ; i < 2*NTOTAL_COMPLEX ; i++) {
		// Example: init a flux tube in the x direction+a vertical displacement

		// wr1[i] =  0.000;
		// wr1[i] =  0.0001*cos(2.0*M_PI*y[i]);
		// wr2[i] =  0.0001*cos(2.0*M_PI*x[i]);
		wr5[i] = 1.414e-5*sin(2.0*M_PI*z[i]);
		//////////////////////////////////////////////////////////////////////////
		// MRI MODE
		// wr1[i] = 0.0001*cos(kX*x[i] + kZ*z[i]);
		// wr2[i] = wr1[i]*(param.shear*kZ*kZ*vA2/(sigma_eta*sigma_eta) - (2*param.omega - param.shear))/(sigma_nu + kZ*kZ*vA2/sigma_eta);
		// wr3[i] = -(kX/kZ)*0.0001*cos(kX*x[i] + kZ*z[i]);

		// wr4[i] = -0.0001*sin(kX*x[i] + kZ*z[i])*param.bz0*kZ/sigma_eta;
		// wr5[i] = 0.0001*sin(kX*x[i] + kZ*z[i])*( param.bz0*kZ*param.shear/(sigma_eta*sigma_eta) - (param.bz0*kZ/sigma_eta)*(param.shear*kZ*kZ*vA2/(sigma_eta*sigma_eta) - (2*param.omega - param.shear))/(sigma_nu + kZ*kZ*vA2/sigma_eta) );
		// wr6[i] = (kX/kZ)*0.0001*sin(kX*x[i] + kZ*z[i])*param.bz0*kZ/sigma_eta;
		/////////////////////////////////////////////////////////////////////////
	}
	/*******************************************************************
	** This part does not need to be modified **************************
	********************************************************************/
	// Fourier transform everything
	gfft_r2c(wr1);
	gfft_r2c(wr2);
	gfft_r2c(wr3);
	gfft_r2c(wr4);
	gfft_r2c(wr5);
	gfft_r2c(wr6);
	gfft_r2c(wr7);

	// Transfer data in the relevant array (including dealiasing mask)
	for(i = 0 ; i < NTOTAL_COMPLEX ; i++) {
		fldi.vx[i] += w1[i] * mask[i];
		fldi.vy[i] += w2[i] * mask[i];
		fldi.vz[i] += w3[i] * mask[i];
#ifdef MHD
		fldi.bx[i] += w4[i] * mask[i];
		fldi.by[i] += w5[i] * mask[i];
		fldi.bz[i] += w6[i] * mask[i];
#endif
#ifdef BOUSSINESQ
	fldi.th[i] += w7[i] * mask[i];
#endif	// BOUSSINESQ
	}


	// free memory
	fftw_free(x);
	fftw_free(y);
	fftw_free(z);

	//done
	return;
}


void init_KidaVortex(struct Field fldi) {
	double a = param.vortex_a;
	double b = param.vortex_b;

	int i,j,k;

	double w0, x, y;
	double chi;

	chi = b / a;
	w0 = 1.0/chi*(chi + 1.0)/(chi-1.0);			// According to Kida!

	for(i = 0 ; i < NX/NPROC ; i++) {
		x = - param.lx / 2 + (param.lx * (i + rank * NX / NPROC)) / NX;
		for(j = 0 ; j < NY ; j++) {
			y = - param.ly / 2 + (param.ly * j) / NY;
#ifdef WITH_2D
			if(x * x / (a * a) + y * y / (b * b) < 1) {
					// we are in the vortex
					wr1[j + (NY+2) * i] = -w0;
			}
			else {
				wr1[j + (NY+2) * i] = 0.0;
			}
#else
			for(k = 0 ; k < NZ ; k++) {
				if(x * x / (a * a) + y * y / (b * b) < 1) {
					// we are in the vortex
					wr1[k + j*(NZ+2) + (NZ+2) * NY * i] = -w0;
				}
				else {
					wr1[k + j*(NZ+2) + (NZ+2) * NY * i] = 0.0;
				}
			}
#endif
		}
	}

	// transform
	gfft_r2c(wr1);

	for(i = 0 ; i < NTOTAL_COMPLEX ; i++) {
		fldi.vx[ i ] +=  I * ky[i] * w1[i] * ik2t[i];
		fldi.vy[ i ] += -I * kxt[i] * w1[i] * ik2t[i];
	}

	// done
	return;
}

/************************************/
/** Init some crazy structure involving
/** A kida vortex and a vertical structure
/** for the field */
/***********************************/
void init_Bench(struct Field fldi) {
	const double a = 0.3;
	const double b = 0.4;

	int i,j,k;

	double w0, x, y;
	double chi;

	chi = b / a;
	w0 = 1.0/chi*(chi + 1)/(chi-1.0);			// According to Kida!

	for(i = 0 ; i < NX/NPROC ; i++) {
		x = - param.lx / 2. + (param.lx * (i + rank * NX / NPROC)) / NX;
		for(j = 0 ; j < NY ; j++) {
			y = - param.ly / 2. + (param.ly * j) / NY;
			for(k = 0 ; k < NZ ; k++) {
				if(x * x / (a * a) + y * y / (b * b) < 1) {
					// we are in the vortex
					wr1[k + j*(NZ+2) + (NZ+2) * NY * i] = -w0;
				}
				else {
					wr1[k + j*(NZ+2) + (NZ+2) * NY * i] = 0.0;
				}
			}
		}
	}

	// transform
	gfft_r2c(wr1);

	for(i = 0 ; i < NTOTAL_COMPLEX ; i++) {
		fldi.vx[ i ] +=  I * ky[i] * w1[i] * ik2t[i];
		fldi.vy[ i ] += -I * kxt[i] * w1[i] * ik2t[i];
	}

	// Brake vertical symmetry
	if(rank==0) {
		fldi.vx[1] = 1000.0 / NTOTAL;
		fldi.vy[1] = 1000.0 / NTOTAL;
#ifdef MHD
		fldi.bx[1] = 1000.0 / NTOTAL;
		fldi.by[1] = 1000.0 / NTOTAL;
#endif
	}
	// done
	return;
}


void init_LargeScaleNoise(struct Field fldi) {
	int i,j,k;
	int num_force=0;
	int total_num_force;
	double fact;

	for( i = 0; i < NX_COMPLEX/NPROC; i++) {
		for( j = 0; j < NY_COMPLEX; j++) {
			for( k = 0; k < NZ_COMPLEX; k++) {
				if( (pow(k2t[ IDX3D ], 0.5) / ( 2.0*M_PI ) < 1.0 / param.noise_cut_length) && ( k2t[ IDX3D] > 0.0 ) ){ //to rule out mean perturbations
					fldi.vx[ IDX3D ] += param.per_amplitude_large * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
					fldi.vy[ IDX3D ] += param.per_amplitude_large * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
					fldi.vz[ IDX3D ] += param.per_amplitude_large * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
#ifdef MHD
					// fldi.bx[ IDX3D ] += param.per_amplitude_large * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
					// fldi.by[ IDX3D ] += param.per_amplitude_large * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
					// fldi.bz[ IDX3D ] += param.per_amplitude_large * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
#endif
					if(mask[IDX3D] > 0) num_force++;
				}
			}
		}
	}

	// Get the total number of forced scales.
#ifdef MPI_SUPPORT
	MPI_Allreduce( &num_force, &total_num_force, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
	total_num_force=num_force;
#endif

	fact=pow(total_num_force,0.5);

	// Divide by the total number of modes
	for( i = 0; i < NX_COMPLEX/NPROC; i++) {
		for( j = 0; j < NY_COMPLEX; j++) {
			for( k = 0; k < NZ_COMPLEX; k++) {
				fldi.vx[ IDX3D ] = fldi.vx[ IDX3D ] / fact;
				fldi.vy[ IDX3D ] = fldi.vy[ IDX3D ] / fact;
				fldi.vz[ IDX3D ] = fldi.vz[ IDX3D ] / fact;
#ifdef MHD
				fldi.bx[ IDX3D ] = fldi.bx[ IDX3D ] / fact;
				fldi.by[ IDX3D ] = fldi.by[ IDX3D ] / fact;
				fldi.bz[ IDX3D ] = fldi.bz[ IDX3D ] / fact;
#endif
			}
		}
	}

  enforce_complex_symm(fldi);
}

/******************************************
** Large scale 2D (x,y) noise *************
*******************************************/

void init_LargeScale2DNoise(struct Field fldi) {
	int i,j,k;
	int num_force=0;
	int total_num_force;
	double fact;

	for( i = 0; i < NX_COMPLEX/NPROC; i++) {
		for( j = 0; j < NY_COMPLEX; j++) {
			k=0;
			if(kz[ IDX3D ] == 0.0) {
				// MPI_Printf("k/2pi = %.2e \n", (pow(k2t[ IDX3D ], 0.5) / ( 2.0*M_PI )));
				// MPI_Printf("k_lim = %.2e \n", 0.0000000001 / param.noise_cut_length_2D);
				// if ((pow(k2t[ IDX3D ], 0.5) / ( 2.0*M_PI ) > 10.0 / param.noise_cut_length_2D) && (pow(k2t[ IDX3D ], 0.5) / ( 2.0*M_PI ) < 20.0 / param.noise_cut_length_2D) ) {

				if (pow(k2t[ IDX3D ], 0.5) / ( 2.0*M_PI ) < 1.0 / param.noise_cut_length_2D) {
				// if(pow(k2t[ IDX3D ], 0.5) / ( 2.0*M_PI ) <= 2.0) {
					fldi.vx[ IDX3D ] += param.per_amplitude_large_2D * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
					fldi.vy[ IDX3D ] += param.per_amplitude_large_2D * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
					// fldi.vx[ IDX3D ] += 0.00001 * mask[IDX3D] * cexp( I * 2.0*M_PI) * NTOTAL;
					// fldi.vy[ IDX3D ] += 0.00001 * mask[IDX3D] * cexp( I * 2.0*M_PI) * NTOTAL;
#ifdef MHD
#ifdef BNOISE
					if (IDX3D > 0.0) {
						fldi.bx[ IDX3D ] += 100*param.per_amplitude_large_2D * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
						fldi.by[ IDX3D ] += 100*param.per_amplitude_large_2D * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * NTOTAL;
					}
#endif
#endif
					if(mask[IDX3D] > 0) num_force++;
				}
			}
		}
	}

	// Get the total number of forced scales.
#ifdef MPI_SUPPORT
	MPI_Allreduce( &num_force, &total_num_force, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#else
	total_num_force=num_force;
#endif

	fact=pow(total_num_force,0.5);

	// Divide by the total number of modes
	for( i = 0; i < NX_COMPLEX/NPROC; i++) {
		for( j = 0; j < NY_COMPLEX; j++) {
			k=0;
			if(kz[ IDX3D ] == 0.0) {
				fldi.vx[ IDX3D ] = fldi.vx[ IDX3D ] / fact;
				fldi.vy[ IDX3D ] = fldi.vy[ IDX3D ] / fact;
#ifdef MHD
#ifdef BNOISE
				fldi.bx[ IDX3D ] = fldi.bx[ IDX3D ] / fact;
				fldi.by[ IDX3D ] = fldi.by[ IDX3D ] / fact;
#endif
#endif
			}
		}
	}

  enforce_complex_symm(fldi);
}


void init_WhiteNoise(struct Field fldi) {
	int i,j,k;
	double fact;

	// Excite (2/3)^3*NTOTAL modes
	fact = pow(27.0/8.0*NTOTAL, 0.5);

	for( i = 0; i < NX_COMPLEX/NPROC; i++) {
		for( j = 0; j < NY_COMPLEX; j++) {
			for( k = 0; k < NZ_COMPLEX; k++) {
				fldi.vx[ IDX3D ] += param.per_amplitude_noise * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * fact;
				fldi.vy[ IDX3D ] += param.per_amplitude_noise * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * fact;
				fldi.vz[ IDX3D ] += param.per_amplitude_noise * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * fact;
#ifdef BNOISE
				fldi.bx[ IDX3D ] += 0.01*param.per_amplitude_noise * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * fact;
				fldi.by[ IDX3D ] += 0.01*param.per_amplitude_noise * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * fact;
				fldi.bz[ IDX3D ] += 0.01*param.per_amplitude_noise * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * fact;
#endif
			}
		}
	}

	enforce_complex_symm(fldi);

}


void init_WhiteNoiseVelocities(struct Field fldi) {
	int i,j,k;
	double fact;

	// Excite (2/3)^3*NTOTAL modes
	fact = pow(27.0/8.0*NTOTAL, 0.5);

	for( i = 0; i < NX_COMPLEX/NPROC; i++) {
		for( j = 0; j < NY_COMPLEX; j++) {
			for( k = 0; k < NZ_COMPLEX; k++) {
				fldi.vx[ IDX3D ] += param.per_amplitude_noise * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * fact;
				fldi.vy[ IDX3D ] += param.per_amplitude_noise * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * fact;
				fldi.vz[ IDX3D ] += param.per_amplitude_noise * mask[IDX3D] * randm() * cexp( I * 2.0*M_PI*randm() ) * fact;
			}
		}
	}

	enforce_complex_symm(fldi);

}

void init_MeanField(struct Field fldi) {
#ifdef MHD
	if(rank==0) {
		fldi.bx[0] = param.bx0 * ((double) NTOTAL);
		fldi.by[0] = param.by0 * ((double) NTOTAL);
		fldi.bz[0] = param.bz0 * ((double) NTOTAL);

		// fldi.vx[1] = 1e2* param.bx0 * ((double) NTOTAL);
	}
#endif
}


/** Init the flow arrays... */
void init_flow(struct Field fldi) {
	int i,n;
	int j,k;

	double dummy_var;

	DEBUG_START_FUNC;
	// Initialise vectors to 0

	for( n = 0 ; n < fldi.nfield ; n++) {
		for( i = 0 ; i < NTOTAL_COMPLEX ; i++) {
			fldi.farray[n][i] = 0.0;
		}
	}

#ifdef COMPRESSIBLE
	// Initialise the density to 1...
	if(rank==0) {
		fldi.d[0] = (double) NTOTAL;
	}
#endif
	if(param.init_large_scale_noise) init_LargeScaleNoise(fldi);

	if(param.init_large_scale_2D_noise) init_LargeScale2DNoise(fldi);

	if(param.init_vortex) init_KidaVortex(fldi);

	if(param.init_spatial_structure) init_SpatialStructure(fldi);

	if(param.init_white_noise) init_WhiteNoise(fldi);

	// if(param.init_white_noise_vel) init_WhiteNoiseVelocities(fldi);

	if(param.init_bench) init_Bench(fldi);

	if(param.init_mean_field) init_MeanField(fldi);

	if(param.init_dump) {
		read_dump(fldi, &dummy_var,"init.dmp");
		MPI_Printf("Initial conditions read successfully from the restart dump\n");
	}

#ifdef BOUNDARY_C
	boundary_c(fldi);
#endif

	projector(fldi.vx,fldi.vy,fldi.vz);

#ifdef MHD
	projector(fldi.bx,fldi.by,fldi.bz);
#endif

#ifdef WITH_PARTICLES
#ifdef WITH_ROTATION
		if(rank==0) {
			kappa_tau2 = 2.0*param.omega*(2.0*param.omega-param.shear) * param.particles_stime * param.particles_stime + (param.particles_dg_ratio + 1.0) * (param.particles_dg_ratio + 1.0);

	// This is a non trivial equilibrium for the particles+gas system
			fldi.vx[0] = param.particles_epsilon*param.particles_stime*param.particles_dg_ratio / kappa_tau2 * ( (double) NTOTAL);
			fldi.vy[0] = param.particles_epsilon*param.particles_dg_ratio*(1.0+param.particles_dg_ratio)/(2.0*param.omega*kappa_tau2) * ( (double) NTOTAL);
		}
#endif
#endif

#ifdef DEBUG
	MPI_Printf("Initflow:\n");
	D_show_all(fldi);
	MPI_Printf("**************************************************************************************\n");
#endif

	DEBUG_END_FUNC;

	return;
}

