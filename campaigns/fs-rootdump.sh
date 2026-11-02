#!/usr/bin/env bash
# fs-rootdump.sh <dir> <tag> - READ-ONLY snapshot of the memory controller / EMC / DRAM state, for
# comparing board 1's two light-load states. It lists and reads small debugfs files under clock and
# memory-controller directories and changes nothing; files whose names suggest an action are skipped.
# Run as root: sudo bash fs-rootdump.sh <dir> <tag>
[ "$(id -u)" = 0 ] || { echo "run as root (sudo bash $0 <dir> <tag>)"; exit 1; }
out=$1/rootdump-$2.txt.tmp
{
    echo "== $(date '+%F %T') $2"
    echo "== tegrastats (as root it shows EMC)"; timeout 2 tegrastats --interval 500 | head -2
    echo "== candidate directories"
    find /sys/kernel/debug -maxdepth 3 -type d \( -iname "*emc*" -o -iname "mc" -o -iname "*dram*" \) 2>/dev/null | sort
    for d in $(find /sys/kernel/debug -maxdepth 3 -type d \( -iname "*emc*" -o -iname "mc" -o -iname "*dram*" \) 2>/dev/null); do
        for f in $(find "$d" -maxdepth 2 -type f -perm -u=r 2>/dev/null | grep -viE "reset|trigger|clear|flush|start|stop|test|write|inject|dump_|regs$"); do
            v=$(timeout 1 head -c 2048 "$f" 2>/dev/null | tr '\n' ' ' | tr -s ' ')
            [ -n "$v" ] && echo "$f = $v"
        done
    done
} > "$out" 2>&1
chmod 644 "$out"; mv "$out" "$1/rootdump-$2.txt"
echo "wrote $1/rootdump-$2.txt ($(wc -l < "$1/rootdump-$2.txt") lines)"
