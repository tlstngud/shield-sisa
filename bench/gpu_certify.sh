#!/bin/bash
# Certification of the CUDA engine's approximate modes (and the metrics of the exact fp32 mode), GPU 0.
#   bash bench/gpu_certify.sh            -> bench/artifacts/gpu_logits_<mode>.jsonl, gpu_cert_<mode>.{json,md},
#                                           gpu_compare_<mode>_vs_{f64,published}.txt, build/shield_v.weights.cuda.cert
# A mode gets a certificate line only if bench/metrics.py (the oracle's metric code) returns PASS against the exact
# baseline logits.cpu_f32 on all 840 val+test samples.  The line binds the weight-file SHA-256, the mode, the
# engine's compute revision (hash of the numeric sources; any source change invalidates it) and the decision threshold
# the metrics were computed at (SHIELDV_CUDA_CERT_V2: "threshold=<repr>", read back from the metrics JSON; an edited
# threshold in <weights>.json invalidates the line).
#   CPUSET (default 24-31) pins the run; CUDA_VISIBLE_DEVICES defaults to 0.
cd "$(dirname "$0")/.." || exit 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CPUSET=${CPUSET:-24-31}
PY="taskset -c $CPUSET ${PYTHON:-python3} -I"
A=bench/artifacts
W=build/shield_v.weights
SHA=$(sha256sum $W | cut -d' ' -f1)
CERT=$W.cuda.cert
TMP=$CERT.tmp.$$
echo "SHIELDV_CUDA_CERT_V2" > $TMP
REV=""
for M in fp32 bf16 fp16; do
  taskset -c $CPUSET build/cuda/gpu_cli --weights $W --mode $M --batch 4 --order map --api sync \
    --out $A/gpu_logits_$M.jsonl --threads 8 2> $A/gpu_logits_$M.log || { cat $A/gpu_logits_$M.log; exit 1; }
  R=$(grep -o '"revision": "[^"]*"' $A/gpu_logits_$M.log | cut -d'"' -f4)
  [ -z "$REV" ] && REV=$R
  [ "$R" != "$REV" ] && { echo "revision changed during certification"; exit 1; }
  $PY bench/metrics.py --logits $A/gpu_logits_$M.jsonl --field logit --baseline bench/artifacts/oracle_logits.jsonl \
    --baseline-field logits.cpu_f32 --name "cuda_$M ($REV)" --out-json $A/gpu_cert_$M.json --out-md $A/gpu_cert_$M.md \
    > $A/gpu_cert_$M.log 2>&1
  RC=$?
  $PY bench/compare_logits.py $A/gpu_logits_$M.jsonl bench/artifacts/oracle_logits.jsonl:logits.cpu_f64 > $A/gpu_compare_${M}_vs_f64.txt 2>&1
  $PY bench/compare_logits.py $A/gpu_logits_$M.jsonl bench/artifacts/oracle_logits.jsonl:logits.cpu_f32 > $A/gpu_compare_${M}_vs_f32.txt 2>&1
  $PY bench/compare_logits.py $A/gpu_logits_$M.jsonl bench/artifacts/oracle_logits.jsonl:logits.gpu_bf16_b4 > $A/gpu_compare_${M}_vs_published.txt 2>&1
  if [ $RC -eq 0 ] && [ "$M" != "fp32" ]; then
    JS=$(sha256sum $A/gpu_cert_$M.json | cut -d' ' -f1)
    THR=$($PY -c 'import json,sys; print(repr(float(json.load(open(sys.argv[1]))["threshold"])))' $A/gpu_cert_$M.json)
    [ -z "$THR" ] && { echo "no threshold in $A/gpu_cert_$M.json"; exit 1; }
    echo "$SHA $M $REV PASS $A/gpu_cert_$M.json $JS threshold=$THR" >> $TMP
  fi
  echo "$M: metrics exit $RC ($(grep -o 'Verdict: [^(]*' $A/gpu_cert_$M.log))"
done
mv $TMP $CERT
cat $CERT
taskset -c $CPUSET build/cuda/gpu_certcheck $W fp32 bf16 fp16 int8
