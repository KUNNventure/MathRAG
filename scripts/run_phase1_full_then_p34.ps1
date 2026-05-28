# Phase 1 全量 48 题确认 → 有进步则跑 Phase 3.4
# 用法（项目根目录）: powershell -ExecutionPolicy Bypass -File scripts\run_phase1_full_then_p34.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = Join-Path $Root ".venv311\Scripts\python.exe"
$TestSet = "C:\Users\xsk\Desktop\RAG项目优化\golden_test_set.json"
$P1Dir = "results/p1.1_full48"
$P34Dir = "results/p3.4"
$Prompt34 = "config/prompts/answer_generation_v3.4.txt"

Write-Host "=== Phase 1: 48q full (1.1-b: k=10, hybrid, rerank v2) ===" -ForegroundColor Cyan
& $Python scripts/evaluate.py `
    --test-set $TestSet `
    --results-dir $P1Dir `
    --top-k 10 `
    --workers 4

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Compare vs baseline_v2 ===" -ForegroundColor Cyan
& $Python scripts/_compare_eval_reports.py `
    (Join-Path $Root "results/baseline_v2/report.json") `
    (Join-Path $Root "$P1Dir/report.json")
$cmpExit = $LASTEXITCODE

if ($cmpExit -ne 0) {
    Write-Host "`nPhase 1 未优于 baseline（score_full），停止，不跑 3.4。请分析 report.json / answers.jsonl。" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== Phase 3.4: 边界拒答 prompt（48q） ===" -ForegroundColor Cyan
& $Python scripts/evaluate.py `
    --test-set $TestSet `
    --results-dir $P34Dir `
    --top-k 10 `
    --workers 4 `
    --prompt-path $Prompt34

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "`n=== Phase 3.4 vs Phase 1 full（Faith / 边界类） ===" -ForegroundColor Cyan
& $Python scripts/_compare_eval_reports.py `
    (Join-Path $Root "$P1Dir/report.json") `
    (Join-Path $Root "$P34Dir/report.json")

Write-Host "`nDone. 结果: $P1Dir , $P34Dir" -ForegroundColor Green
