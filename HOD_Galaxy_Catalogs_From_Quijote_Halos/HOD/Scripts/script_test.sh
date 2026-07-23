#!/bin/sh

cd ../Examples/

IDIR=Quijote_Halos
ODIR=


RUNS=0
start=$(date)
echo "start time : $start"
CAT=Gfiduc
#CAT=G85L
RED=0.5
#RSD=3
zE=0
SEED=1805


SAMPLES=0
for RUN in $(seq 0 $RUNS)
do
    for RSD in 3
    do
        for SS in $(seq 0 $SAMPLES)
        do
	    iSEED=$((SEED + SS))
            ifile=${RUN}
            ofile=${CAT}_HOD0_NFW_sample${SS}_1Gpc_z${RED}0_RSD${RSD}_run${RUN}.npz
            python3 -u hod_test.py \
            --ifile ${IDIR}/${ifile} \
            --outfile ${ODIR}${ofile}\
            --iRSD ${RSD} \
            --zerr ${zE}  \
            --seed ${iSEED}
           
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

