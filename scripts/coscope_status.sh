#!/usr/bin/env bash
# Print one-page status of all running coscope_* screens / logs.
cd /home/ninghanwen/lixin/CoScope
echo "=========================================="
echo "  CoScope status @ $(date)"
echo "=========================================="
echo
echo "--- screens ---"
screen -ls 2>&1 | grep -E "coscope|Sockets" || echo "(none)"
echo
echo "--- stage_b (A8 query_intent) ---"
LB=$(ls -t logs/stage_b_qwen_*.log 2>/dev/null | head -1)
[ -n "$LB" ] && tail -5 "$LB" || echo "(no log)"
echo
echo "--- 2wiki build ---"
LW=$(ls -t logs/build_2wiki_*.log 2>/dev/null | head -1)
[ -n "$LW" ] && tail -5 "$LW" || echo "(no log)"
echo
echo "--- k=20 eval ---"
LK=$(ls -t logs/qwen_k20_*.log 2>/dev/null | head -1)
[ -n "$LK" ] && tail -5 "$LK" || echo "(no log)"
echo
echo "--- output files ---"
ls -la data/processed/got/musique/test/eval_v9_qwen_full_k*.json 2>/dev/null
echo
ls data/processed/got/musique/test_qi/*.jsonl 2>/dev/null | wc -l | xargs -I {} echo "stage_b shards written: {}"
ls data/processed/got/2wikimhqa/test/*.jsonl 2>/dev/null | wc -l | xargs -I {} echo "2wiki shards written: {}"
echo
echo "--- python procs ---"
ps -ef | grep -E "eval_jsonl|generate_query|build_all" | grep -v grep | awk '{print $2, "->", substr($0, index($0,"python"))}' | head -10
