#!/bin/bash
#PBS -N universal_tiny
#PBS -q gpu@pbs-m1.metacentrum.cz
#PBS -l select=1:ncpus=8:ngpus=1:mem=10gb:gpu_mem=40gb:scratch_local=20gb
#PBS -l walltime=48:00:00 
#PBS -j oe
#PBS -o job_output.log

export TMPDIR=$SCRATCHDIR
export TEMP=$SCRATCHDIR
export TMP=$SCRATCHDIR

WORKDIR=/storage/brno12-cerit/home/luhi/adversarial_attack
cd $WORKDIR
source Pyhon310/bin/activate || exit 1

module add ffmpeg

bash universal.sh || exit 2

clean_scratch