#!/bin/sh
#SBATCH --job-name creating_images    # this is a parameter to help you sort your job when listing it
#SBATCH --error jobname-error.e%j     # optional. By default a file slurm-{jobid}.out will be created
#SBATCH --output jobname-out.o%j      # optional. By default the error and output files are merged
#SBATCH --ntasks 1                    # number of tasks in your job. One by default
#SBATCH --cpus-per-task 1             # number of cpus for each task. One by default
#SBATCH --partition public-cpu        # the partition to use. By default debug-cpu
#SBATCH --time 05:00:00               # maximum run time.
#SBATCH --mem-per-cpu=9000 # in MB

module load GCC/10.3.0
module load OpenMPI/4.1.1
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load NCCL/2.10.3-CUDA-11.3.1
module load Python/3.9.5

DIR=/srv/beegfs/scratch/users/n/nogueir3/keras-cv-yolov8-quantized-master/venv
if [ ! -d "$DIR" ];
then
    echo "Creating virtual env"
    python3 -m venv venv
fi

echo "Activating virtual env"
source ~/scratch/keras-cv-yolov8-quantized-master/venv/bin/activate
 
echo "Updating requirements"
pip3 install -r requirements.txt

echo "Running python script to train model"
srun python3 main.py

echo "Running python script to evaluate model"
srun python3 evaluate.py