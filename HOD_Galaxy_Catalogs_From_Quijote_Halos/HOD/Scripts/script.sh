#!/bin/sh

IDIR=/Users/matteobiagetti/Halo_catalogs/
ODIR=/Users/matteobiagetti/GitProjects/py-power/output/


RUNS=15
start=$(date)
echo "start time : $start"
#CAT=Gfiduc
CAT=G85L
RED=1
#RSD=3
zE=0
SEED=1805


SAMPLES=0
for RUN in $(seq 1 15)
do
    for RSD in $(seq 3 3)
    do
        for SS in $(seq 0 $SAMPLES)
        do
	    iSEED=$((SEED + SS))
            ifile=halos_G85L_run${RUN}_z${RED}.00.npz
            ofile=${CAT}_HOD0_NFW_sample${SS}_2Gpc_z${RED}0_RSD${RSD}_run${RUN}.npz
            python -u main.py --ifile ${IDIR}${CAT}/${ifile} --outfile ${ODIR}${ofile} --iRSD ${RSD} --zerr ${zE} --seed ${iSEED} --saveNgal 0
           
        done
    done
done
endd=$(date) 
echo "current time : $endd"
###########################

###########################

# CAT=NG10L

# for RUN in $(seq 1 $RUNS)
# do
#     for SS in $(seq 0 $SAMPLES)
#     do
#         #
#         hfile=${CAT}/halos_${CAT}_run${RUN}_z${RED}.00.npz
#         #hfile=${RUN}
#         ofile=${CAT}_HOD0_sample${SS}_2Gpc_z${RED}.00_run${RUN}.npz ##DO NOT ERASE _HOD0_
#         python main.py --ifile ${IDIR}${hfile}  --outfile ${ODIR}${ofile} --iRSD ${RSD} --zerr ${zE} --seed ${SEED} --saveNgal 0
           
#     done
# done
# endd=$(date) 
# echo "current time : $endd"

