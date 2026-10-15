# AI Video Enhancer - one-shot build script (M7 B)
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
    Write-Host "== 1/3 PyInstaller (onedir, multi-GB - this takes a while) ==" -ForegroundColor Cyan
    & $Python -m PyInstaller packaging/aive.spec --noconfirm
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "== 1/3 PyInstaller - skipped (-SkipPyInstaller) ==" -ForegroundColor Cyan
}

Write-Host "== 2/3 Bundle FFmpeg ==" -ForegroundColor Cyan
# Resolve an FFmpeg bin dir that actually contains both tools. The default is a
# versioned WinGet path, but WinGet auto-updates rename it (e.g. ffmpeg-9.0.1 ->
# 9.0.2), so if the configured dir is missing we glob for any Gyan full_build bin
# under the WinGet package root and pick the newest. -FfmpegDir still overrides.
$tools = @("ffmpeg.exe", "ffprobe.exe")
function Test-FfmpegBin([string]$dir) {
    if (-not $dir) { return $false }
    foreach ($t in $tools) { if (-not (Test-Path (Join-Path $dir $t))) { return $false } }
    return $true
}
$ffbin = $null
if (Test-FfmpegBin $FfmpegDir) {
    $ffbin = $FfmpegDir
} else {
    $wingetRoot = "C:\Users\junmi\AppData\Local\Microsoft\WinGet\Packages"
    if (Test-Path $wingetRoot) {
        $bins = @(Get-ChildItem -Path $wingetRoot -Directory -Filter "Gyan.FFmpeg*" -ErrorAction SilentlyContinue |
            ForEach-Object { Get-ChildItem -Path $_.FullName -Directory -Filter "*-full_build" -ErrorAction SilentlyContinue } |
            Where-Object { Test-FfmpegBin (Join-Path $_.FullName "bin") } |
            Sort-Object LastWriteTime)
        if ($bins.Count -gt 0) { $ffbin = Join-Path $bins[-1].FullName "bin" }
    }
}
if (-not $ffbin) {
    throw "FFmpeg (ffmpeg.exe + ffprobe.exe) not found. Pass -FfmpegDir <path-to-bin>."
}
Write-Host "Using FFmpeg: $ffbin" -ForegroundColor DarkGray
$bin = Join-Path $PSScriptRoot "ffmpeg-bin"
New-Item -ItemType Directory -Force $bin | Out-Null
foreach ($t in $tools) { Copy-Item (Join-Path $ffbin $t) (Join-Path $bin $t) -Force }

if ($SkipInstaller) { return }

Write-Host "== 3/3 Inno Setup installer ==" -ForegroundColor Cyan
$iscc = $null
$cmd = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if ($cmd) { $iscc = $cmd.Source }
elseif (Test-Path "C:\Program Files (x86)\Inno Setup 6\ISCC.exe") {
    $iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
}
if (-not $iscc) {
    Write-Warning "ISCC not found - skipping installer. Install Inno Setup 6 (winget install JRSoftware.InnoSetup), then re-run this script or compile packaging\ai-video-enhancer.iss manually."
    return
}
& $iscc (Join-Path $PSScriptRoot "ai-video-enhancer.iss")
if ($LASTEXITCODE -ne 0) { throw "ISCC failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Done:" -ForegroundColor Green
Write-Host "  Portable bundle : dist\ai-video-enhancer\"
Write-Host "  Installer       : packaging\output\ai-video-enhancer-setup-0.1.0.exe"
