# AI Video Enhancer - release ZIP builder (GitHub Releases)
#
#   powershell -ExecutionPolicy Bypass -File packaging\make_release_zip.ps1 [-Version 0.1.0]
#
# Produces dist\ai-video-enhancer-<ver>-win-x64.zip containing the ai-video-enhancer\
# folder (exe + _internal + ffmpeg). If the zip exceeds ~1.9 GB it is split into
# <name>.zip.partNN files, because GitHub release assets are capped at 2 GB each;
# reassemble with:  copy /b part01+part02+... = full.zip   (see BUILD.md)

param(
    [string]$Version = "0.1.0",
    [string]$BundleDir = "",      # default: dist\ai-video-enhancer
    [switch]$SkipFfmpeg           # bundle already has ffmpeg\ staged
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

if (-not $BundleDir) { $BundleDir = Join-Path $repo "dist\ai-video-enhancer" }
if (-not (Test-Path (Join-Path $BundleDir "ai-video-enhancer.exe"))) {
    throw "Bundle not found at $BundleDir - run packaging\build.ps1 first."
}

# --- 1) Stage FFmpeg into the bundle (self-contained release) -------------------
# Frozen builds find ffmpeg under <exe>\ffmpeg\ (aive/stages/ingest.py).
if (-not $SkipFfmpeg) {
    $ffbin = Join-Path $PSScriptRoot "ffmpeg-bin"
    if (-not (Test-Path (Join-Path $ffbin "ffmpeg.exe"))) {
        throw "packaging\ffmpeg-bin\ffmpeg.exe missing - run packaging\build.ps1 first (it stages FFmpeg)."
    }
    $dst = Join-Path $BundleDir "ffmpeg"
    New-Item -ItemType Directory -Force $dst | Out-Null
    Copy-Item (Join-Path $ffbin "ffmpeg.exe"), (Join-Path $ffbin "ffprobe.exe") $dst -Force
    Write-Host "Staged FFmpeg into bundle\ffmpeg\" -ForegroundColor DarkGray
}

# --- 2) Zip with 7-Zip (fast deflate; ZIP64 handled automatically for >4 GB) ----
$sevenZip = $null
foreach ($c in @("C:\Program Files\7-Zip\7z.exe", "C:\Program Files (x86)\7-Zip\7z.exe")) {
    if (Test-Path $c) { $sevenZip = $c; break }
}
if (-not $sevenZip) { throw "7-Zip not found - install it first: winget install 7zip.7zip" }

$zipName = "ai-video-enhancer-$Version-win-x64.zip"
$zipPath = Join-Path $repo ("dist\" + $zipName)
Remove-Item $zipPath -ErrorAction SilentlyContinue
Push-Location (Join-Path $repo "dist")
try {
    & $sevenZip a -tzip -mx=1 $zipName "ai-video-enhancer" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7z zip failed (exit $LASTEXITCODE)" }
} finally { Pop-Location }

# --- 3) Split into <2 GB parts if needed (GitHub asset cap = 2 GB each) ---------
$size = [int64](Get-Item $zipPath).Length
Write-Host ("Zip size: {0:N2} GB" -f ($size / 1GB)) -ForegroundColor Cyan

# NOTE: parentheses are REQUIRED - PowerShell's cast precedence makes
# `[int64]1.85 * 1024 * 1024 * 1024` evaluate to exactly 2 GiB, which GitHub rejects.
if ($size -gt [int64](1.9 * 1GB)) {
    $partSize = [int64](1.85 * 1GB)
    # NOTE: the inner loop must fully consume each buffer before the outer
    # Read() refills it - otherwise a part boundary mid-buffer drops bytes.
    $fs = [System.IO.File]::OpenRead($zipPath)
    $n = 0; $partStream = $null; $inPart = [int64]0
    try {
        $buf = New-Object byte[] (64 * 1024 * 1024)
        while (($read = $fs.Read($buf, 0, $buf.Length)) -gt 0) {
            $off = 0
            while ($off -lt $read) {
                if ($null -eq $partStream) {
                    $n++
                    $partPath = "{0}.part{1:D2}" -f $zipPath, $n
                    $partStream = [System.IO.File]::Create($partPath)
                    $inPart = 0
                }
                $w = [Math]::Min([int64]($read - $off), $partSize - $inPart)
                $partStream.Write($buf, $off, $w)
                $off += $w; $inPart += $w
                if ($inPart -ge $partSize) { $partStream.Close(); $partStream = $null }
            }
        }
    } finally {
        $fs.Dispose()
        if ($partStream) { $partStream.Dispose() }
    }
    Remove-Item $zipPath
    $parts = (1..$n) | ForEach-Object { "{0}.part{1:D2}" -f $zipName, $_ }
    Write-Host "Split into $n parts (~1.85 GB each; GitHub caps assets at 2 GB)." -ForegroundColor Yellow
    Write-Host "Reassemble (Windows cmd):" -ForegroundColor Yellow
    Write-Host ("  copy /b " + ($parts -join "+") + " = $zipName")
} else {
    Write-Host "Single zip under the cap - upload as-is." -ForegroundColor Green
}

Write-Host ""
Write-Host "Done:" -ForegroundColor Green
Write-Host "  dist\$zipName (or .partNN files)"
