#!/usr/bin/env bash
# Download + extract LeWM datasets (HF: quentinll/lewm-*) into $STABLEWM_HOME. Resumable.
set -uo pipefail
H=/sfs/weka/scratch/qxe3fj/JEPA/stablewm_home; cd "$H"
get() { # repo file
  local r=$1 f=$2
  curl -sfL -C - -o "$f" "https://huggingface.co/datasets/$r/resolve/main/$f" || { echo "FAIL download $f"; return 1; }
  case $f in
    *.tar.zst) zstd -dc "$f" | tar -x && rm -f "$f" ;;
    *.zst) zstd -d --rm -q "$f" ;;
  esac && echo "OK $f" || echo "FAIL extract $f"
}
get quentinll/lewm-tworooms tworoom.tar.zst &
get quentinll/lewm-reacher reacher.tar.zst &
get quentinll/lewm-cube cube_single_expert.tar.zst &
get quentinll/lewm-pusht pusht_expert_train.h5.zst &
wait; ls -la "$H"; echo ALL_DONE
