export BENCHMARK_PKG=${WORKSPACE}/benchmarking
export PYTHONPATH=${BENCHMARK_PKG}:$PYTHONPATH

export LOG_DIR=${OUTPUT_DIR}/logs/${JOB_NAME_POSTFIX}
mkdir -p $LOG_DIR
echo "LOG_DIR: $LOG_DIR"

export POD_NAME=${HOSTNAME:?}
export TRACE_FILE_PREFIX="${LOG_DIR}/trace_model"

export NCCL_DEBUG=OFF
export NCCL_P2P_NET_CHUNKSIZE=134217728

torchrun --tee=3 --nproc-per-node=8 --nnodes=1 --master_port 23457 \
  ${WORKSPACE}/benchmarking/scripts/ops/ops_tuner.py \
  --model-configs ${WORKSPACE}/benchmarking/model_configs.yaml \
  --tuning-configs ${WORKSPACE}/benchmarking/tuning_configs.yaml \
  --profile-best \
  2>&1 | tee "${LOG_DIR}/configurator-${POD_NAME}.log"