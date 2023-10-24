#!/bin/bash


job_name=$1
command=$2

mkdir -p log

now=$(date +"%Y%m%d_%H%M%S")


# nohup
GLOG_vmodule=MemcachedClient=-1 \
srun --partition=VA \
--job-name=$job_name \
--kill-on-bad-exit=1 \
$command  2>&1|tee log/$job_name-$now.log &



