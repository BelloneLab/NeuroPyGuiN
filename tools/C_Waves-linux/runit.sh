#!/bin/sh

# You can't call C_Waves directly, rather, call it via runit.sh.
# You can call runit.sh two ways:
#
# 1) > runit.sh 'cmd-line-parameters'
# 2a) Edit parameters in runit.sh, then call it ...
# 2b) > runit.sh
#
# This script effectively says:
# "If there are no parameters sent to runit.sh, call C_Waves
# with the parameters hard coded here, else, pass all of the
# parameters through to C_Waves."
#
# Shell notes:
# - This script is provided for "sh" shell. If you need to use
# a different shell feel free to edit the script as required.
# Remember to change the first line to invoke that shell, for
# example, replace /bin/sh with /bin/bash
#
# - In most environments $0 returns the path and name of this
# script, but that is not guaranteed to be true. If using the
# bash shell, it is more reliable to define RUN_DIR like this:
# RUN_DIR=$(dirname $(readlink -f BASH_SOURCE[0]))
#
# - Enclosing whole parameter list in quotes is recommended, like this:
#
#    > runit.sh 'cmd-line-parameters'
#

if [ -z "$1" ]
then
    SRC=/groups/apig/apig/Austin_Graves/CatGT_C_Waves_test/SC_artifact_test_data/SC011/catgt_SC011_022319_g0/SC011_022319_g0_imec3
    DST=$SRC/OUT
    ARGS="-spikeglx_bin=$SRC/SC011_022319_g0_tcat.imec3.ap.bin"
    ARGS="$ARGS -clus_table_npy=$SRC/imec3_ks2/clus_Table.npy"
    ARGS="$ARGS -clus_time_npy=$SRC/imec3_ks2/spike_times.npy"
    ARGS="$ARGS -clus_lbl_npy=$SRC/imec3_ks2/spike_clusters.npy"
    ARGS="$ARGS -dest=$DST"
    ARGS="$ARGS -samples_per_spike=82 -pre_samples=20 -num_spikes=1000 -snr_radius_um=140"
else
    ARGS=$@
fi

RUN_DIR=$(dirname $(readlink -f $0))
export LD_LIBRARY_PATH=$RUN_DIR/links
$LD_LIBRARY_PATH/ld-linux-x86-64.so.2 --library-path $LD_LIBRARY_PATH $RUN_DIR/C_Waves $ARGS

