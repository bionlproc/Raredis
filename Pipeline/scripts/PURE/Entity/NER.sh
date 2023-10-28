#!/bin/bash
#SBATCH --time=3-00:00:00   				#Time for the job to run 
#SBATCH --job-name=EntBiolar   	#Name of the job
#SBATCH -N 1 					            #Number of nodes required
#SBATCH -n 1					            #Number of cores needed for the job
#SBATCH --partition=V4V32_CAS40M192_L	    #Name of the GPU queue
#SBATCH --account=gcl_rvkavu2_uksr 		    #Name of account to run under
#SBATCH --gres=gpu:1				        #Number of GPU cards needed
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sgu260@uky.edu
#SBATCH -e slurm-%j.err                      # Error file for this job.
#SBATCH -o slurm-%j.out   

module load ccs/singularity
container=/share/singularity/images/ccs/pytorch/pytorch.sinf

singularity exec --nv $container python run_entity.py
