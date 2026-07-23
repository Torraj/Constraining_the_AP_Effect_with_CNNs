#!/bin/sh

# THE NAME OF THE SBATCH JOB
#SBATCH -J hod  

# REDIRECT OUTPUT TO THE FOLLOWING STREAM

#SBATCH -e RUNinfo/hod-error%A_%a                                                      
#SBATCH -o RUNinfo/hod-out%A_%a

#SBATCH --nodes 1
#SBATCH --array=0-3
#SBATCH --ntasks=125

#SBATCH --partition=thin

# SET THE TIME LIMIT [HRS:MINS:SECS]

#SBATCH --time=12:00:00

#SBATCH --mail-user=m.biagetti@uva.nl
#SBATCH --mail-type=ALL

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/Codes/Libraries

module load 2021
module load Python/3.9.5-GCCcore-10.3.0

start=$(date)
echo "start time : $start"
RED=0.5
zE=0
SEED=1805
IDIR=/projects/0/einf733/PH-files/jacky-files/Halos/20210218/
CATLIST=("fiducial")
#CATLIST=("fiducial_ZA" "Mnu_p" "ns_m" "Ob2_p" "s8_m" "h_m" "Mnu_pp" "ns_p" "Om_m" "s8_p" "h_p" "Mnu_ppp" "Ob2_m" "Om_p")
REDDIR=/groups_003/group_tab_003.0
ODIR=/projects/0/einf733/PH-files/jacky-files/Galaxies/sancho/

SAMPLES=4
iSAMPLES=0
iRSD=1

for RUN in $(seq 0 124)
do
    (
    for CAT in ${CATLIST[@]}
    do
        for RSD in $(seq ${iRSD} 3)
        do
            for SS in $(seq ${iSAMPLES} $SAMPLES)
            do
	        iSEED=$((SEED + SS))
                ofile=${CAT}_HOD0_NFW_sample${SS}_1Gpc_z${RED}0_RSD${RSD}_run$((${RUN}+${SLURM_ARRAY_TASK_ID}*125)).npz
                python -u main.py --ifile ${IDIR}${CAT}/$((${RUN}+${SLURM_ARRAY_TASK_ID}*125)) --outfile ${ODIR}${CAT}/${ofile} --iRSD ${RSD} --zerr ${zE} --seed ${iSEED} --saveNgal 0
            done
        done
    done
    ) &
done
wait
endd=$(date) 
echo "current time : $endd"
###########################
