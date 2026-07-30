#!/bin/sh

# You can't call TPrime directly, rather, call it via runit.sh.
# You can call runit.sh two ways:
#
# 1) > runit.sh 'cmd-line-parameters'
# 2a) Edit parameters in runit.sh, then call it ...
# 2b) > runit.sh
#
# This script effectively says:
# "If there are no parameters sent to runit.sh, call TPrime
# with the parameters hard coded here, else, pass all of the
# parameters through to TPrime."
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
    SRC=Y:/tptest/time_trans_01_g0_tcat
    DST=Y:/tptest
    ARGS="-syncperiod=1.0"
    ARGS="$ARGS -tostream=$SRC.imec0.ap.SY_301_6_500.txt"
    ARGS="$ARGS -fromstream=7,$SRC.nidq.XA_0_500.txt"
    ARGS="$ARGS -events=7,$SRC.nidq.XA_1_7700.txt,$DST/out.txt"
else
    ARGS=$@
fi

RUN_DIR=$(dirname $(readlink -f $0))
export LD_LIBRARY_PATH=$RUN_DIR/links
$LD_LIBRARY_PATH/ld-linux-x86-64.so.2 --library-path $LD_LIBRARY_PATH $RUN_DIR/TPrime $ARGS

