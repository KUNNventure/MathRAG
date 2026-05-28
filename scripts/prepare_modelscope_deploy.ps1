# Assemble ModelScope Space deploy folder (run from repo root).
# Usage: .\scripts\prepare_modelscope_deploy.ps1 [-OutputDir deploy\modelscope-bundle]

param(
    [string]$OutputDir = "deploy\modelscope-bundle"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TemplateDir = Join-Path $RepoRoot "deploy\modelscope"
$Out = Join-Path $RepoRoot $OutputDir

Write-Host "Repo:     $RepoRoot"
Write-Host "Template: $TemplateDir"
Write-Host "Output:   $Out"

if (-not (Test-Path $TemplateDir)) {
    throw "Missing template: $TemplateDir"
}

if (Test-Path $Out) {
    Write-Host "Clearing existing output (in-place)..."
    Get-ChildItem $Out -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
} else {
    New-Item -ItemType Directory -Path $Out | Out-Null
}

# Static deploy files
$staticItems = @(
    "app.py",
    "rag_query.py",
    "requirements.txt",
    ".gitattributes",
    "config"
)
foreach ($item in $staticItems) {
    $src = Join-Path $TemplateDir $item
    if (-not (Test-Path $src)) { throw "Missing: $src" }
    Copy-Item -Path $src -Destination (Join-Path $Out $item) -Recurse -Force
}

# Prompts
$promptDir = Join-Path $Out "config\prompts"
New-Item -ItemType Directory -Path $promptDir -Force | Out-Null
foreach ($name in @("answer_generation_v3.4.txt", "rerank.txt")) {
    $src = Join-Path $RepoRoot "config\prompts\$name"
    if (-not (Test-Path $src)) { throw "Missing prompt: $src" }
    Copy-Item $src (Join-Path $promptDir $name) -Force
}

# Source tree
Write-Host "Copying src/ ..."
Copy-Item -Path (Join-Path $RepoRoot "src") -Destination (Join-Path $Out "src") -Recurse -Force

# Chroma (Space root name chroma_db per deploy guide)
$chromaSrc = Join-Path $RepoRoot "data\db\chroma"
$chromaDst = Join-Path $Out "chroma_db"
if (-not (Test-Path $chromaSrc)) {
    throw "Chroma DB not found: $chromaSrc — run ingest first."
}
Write-Host "Copying chroma (~80MB) ..."
Copy-Item -Path $chromaSrc -Destination $chromaDst -Recurse -Force

# BM25 sparse index
$bm25Src = Join-Path $RepoRoot "data\db\bm25"
$bm25Dst = Join-Path $Out "data\db\bm25"
if (-not (Test-Path $bm25Src)) {
    throw "BM25 index not found: $bm25Src — run ingest first."
}
Write-Host "Copying BM25 index ..."
New-Item -ItemType Directory -Path (Split-Path $bm25Dst) -Force | Out-Null
Copy-Item -Path $bm25Src -Destination $bm25Dst -Recurse -Force

# Size summary
$size = (Get-ChildItem $Out -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ""
Write-Host "Done. Bundle: $Out"
Write-Host ("Total size: {0:N1} MB" -f ($size / 1MB))
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Create public Gradio Space on modelscope.cn"
Write-Host "  2. git clone <your-space-repo-url>"
Write-Host "  3. Copy all files from $Out into the clone root"
Write-Host "  4. git add . && git commit && git push"
Write-Host "  5. Space Settings -> Secrets: DASHSCOPE_API_KEY=<your-key>"
Write-Host "  (If push fails on large files: git lfs install && git lfs track 'chroma_db/**' 'data/db/bm25/**')"
