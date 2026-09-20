# AI Video Enhancer — one-shot build script (M7 B)
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#
# Steps: 1) PyInstaller onedir bundle -> dist\ai-video-enhancer\
#        2) copy ffmpeg.exe/ffprobe.exe into packaging\ffmpeg-bin\ (for the installer)
#        3) Inno Setup installer -> packaging\output\*.exe   (skipped if ISCC missing)

param(
    [string]$Python = "C:\Python314\python.exe",
    # Gyan full build via WinGet; override with -FfmpegDir <path-to-bin> for other builds.
    [string]$FfmpegDir = "C:\Users\junmi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin",
    [switch]$SkipInstaller,
    [switch]$SkipPyInstaller   # reuse an existing dist\ai-video-enhancer bundle
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $SkipPyInstaller) {
    Write-Host "== 1/3 PyInstaller (onedir, multi-GB — this takes a while) ==" -ForegroundColor Cyan
    & $Python -m PyInstaller packaging/aive.spec --noconfirm
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "== 1/3 PyInstaller — skipped (-SkipPyInstaller) ==" -ForegroundColor Cyan
}

Write-Host "== 2/3 Bundle FFmpeg from $FfmpegDir ==" -ForegroundColor Cyan
$bin = Join-Path $PSScriptRoot "ffmpeg-bin"
New-Item -ItemType Directory -Force $bin | Out-Null
foreach ($t in @("ffmpeg.exe", "ffprobe.exe")) {
    $src = Join-Path $FfmpegDir $t
    if (-not (Test-Path $src)) { throw "FFmpeg tool not found: $src" }
    Copy-Item $src (Join-Path $bin $t) -Force
}

if ($SkipInstaller) { return }

Write-Host "== 3/3 Inno Setup installer ==" -ForegroundColor Cyan
$iscc = $null
$cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($cmd) { $iscc = $cmd.Source }
elseif (Test-Path "C:\Program Files (x86)\Inno Setup 6\ISCC.exe") {
    $iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
}
if (-not $iscc) {
    Write-Warning "ISCC not found — skipping installer. Install Inno Setup 6 (winget install JRSoftware.InnoSetup), then re-run this script or compile packaging\ai-video-enhancer.iss manually."
    return
}
& $iscc (Join-Path $PSScriptRoot "ai-video-enhancer.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Done:" -ForegroundColor Green
Write-Host "  Portable bundle : dist\ai-video-enhancer\"
Write-Host "  Installer       : packaging\output\ai-video-enhancer-setup-0.1.0.exe"
