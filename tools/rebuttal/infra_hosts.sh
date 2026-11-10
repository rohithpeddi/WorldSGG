# INFRA (rebuttal): per-host settings for the remote GPU hosts, sourced by the other infra_*.sh scripts on CS93371.
#   . infra_hosts.sh <host>      host = cs94085 | cs94092
# Sets: RHOST, SSHOPT, SSH (ssh command, usable as rsync -e "$SSH"), RROOT (remote work root), RW (= RROOT/wsgg),
#       RM (sshfs mount root on the remote host), RQ (queue root), RRUNS (small host-local scratch),
#       RPY (CS93371 env python through the env mount), INFRA (this directory on CS93371).
# Nothing is copied to the remote hosts: env, code, data, grids and outputs are CS93371 directories mounted with
# infra_mount.sh (sshfs slave mode) under $RM/{data,grids,env,runs,code,reb}.
case "$1" in
  cs94085) SSHOPT="-o BatchMode=yes -o ServerAliveInterval=30 -o HostName=10.176.140.100 -o HostKeyAlias=cs94085.utdallas.edu"
           RROOT=/data1/rohith ;;
  cs94092) SSHOPT="-o BatchMode=yes -o ServerAliveInterval=30"
           RROOT=/data3/rohith ;;
  *) echo "infra_hosts.sh: unknown host '$1' (cs94085 | cs94092)" >&2; return 2 2>/dev/null || exit 2 ;;
esac
RHOST=$1
SSH="ssh $SSHOPT"
RW=$RROOT/wsgg
RM=$RW/mnt
RQ=$RW/queue
RRUNS=$RW/runs
RPY=$RM/env/bin/python
INFRA=/data3/rohith/ag/runs/rebuttal/infra     # CS93371 side: these scripts, logs, gate dumps
