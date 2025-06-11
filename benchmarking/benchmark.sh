export BENCHMARK_PKG=${WORKSPACE}/benchmarking
export PYTHONPATH=${BENCHMARK_PKG}:$PYTHONPATH

export LOG_DIR=${OUTPUT_DIR}/logs/${JOB_NAME_POSTFIX}
mkdir -p $LOG_DIR
echo "LOG_DIR: $LOG_DIR"

export POD_NAME=${HOSTNAME:?}
export TRACE_FILE_PREFIX="${LOG_DIR}/trace"

torchrun --tee=3 --nproc-per-node=8 --nnodes=32 --master_port 23456 \
  ${WORKSPACE}/benchmarking/benchmark.py \
  --collective all_to_all \
  --tp 8 --pp 2 --vpp 6 --cp 2 --ep 32 --etp 2 \
  --profile-last \
  --nccl-comms nccl_configs.yaml \
  2>&1 | tee "${LOG_DIR}/benchmark-${POD_NAME}.log"

#  --tp 8 --pp 2 --vpp 6 --cp 2 --ep 32 --etp 2 \
#--tp 2 --pp 2 --vpp 1 --cp 1 --ep 1 --etp 1 \