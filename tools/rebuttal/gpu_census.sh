#!/bin/bash
# GPU census across CS93371, cs94085, cs94092 (run on CS93371). Read-only.
# For each GPU: utilisation, memory, and every process on it (user, pid, MiB, elapsed, command).
census() {
  nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits |
  while IFS=, read idx uuid util used total; do
    uuid=$(echo $uuid)
    echo "GPU$idx util=${util# }% mem=${used# }/${total# } MiB"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits |
    while IFS=, read u p m; do
      [ "$(echo $u)" = "$uuid" ] || continue
      p=$(echo $p)
      echo "    pid=$p mem=${m# }MiB user=$(ps -o user= -p $p 2>/dev/null) etime=$(ps -o etime= -p $p 2>/dev/null | tr -d ' ') cmd=$(ps -o args= -p $p 2>/dev/null | cut -c1-110)"
    done
  done
}
echo "===== cs93371"; census
echo "===== cs94085"
ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o HostName=10.176.140.100 -o HostKeyAlias=cs94085.utdallas.edu cs94085 "$(declare -f census); census" 2>&1
echo "===== cs94092"
ssh -n -o BatchMode=yes -o ConnectTimeout=10 cs94092 "$(declare -f census); census" 2>&1
