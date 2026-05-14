#!/bin/bash
# PDFRAG 评估测试脚本 — OCR + 12题QA + 评估报告
# 运行: bash /mnt/d/project/llm/hermes/pdfrag/run_eval.sh
set -e
cd /mnt/d/project/llm/hermes/pdfrag
source .venv/bin/activate

echo "=== 1/3 清空旧数据 ==="
rm -rf data/chroma/*

echo "=== 2/3 运行完整评估测试 (OCR + QA + 报告) ==="
# 预计 5~8 分钟，取决于网络和 CPU
python -u -m pytest tests/test_evaluation.py -v -s --tb=short 2>&1 | tee /tmp/pdfrag_eval.log

echo ""
echo "=== 3/3 查看报告 ==="
if [ -f data/eval_report.json ]; then
    python -c "
import json
r = json.load(open('data/eval_report.json'))
print(f'文档: {r.get(\"document\",\"?\")}')
print(f'通过率: {r[\"passed\"]}/{r[\"total_questions\"]} ({r[\"passed\"]/r[\"total_questions\"]*100:.0f}%)')
print(f'平均关键词召回: {r[\"avg_keyword_recall\"]:.4f}')
print(f'平均语义相似度: {r[\"avg_semantic_similarity\"]:.4f}')
print()
for m in r['results']:
    s = '✅' if m['passed'] else '❌'
    print(f'{s} Q{m[\"qa_id\"]} kw={m[\"keyword_recall\"]:.2f} sim={m[\"semantic_similarity\"]:.2f} | {m[\"question\"][:50]}')
"
fi
echo ""
echo "日志: /tmp/pdfrag_eval.log"
echo "报告: data/eval_report.json"
