; AI Video Enhancer — Inno Setup 6 script (M7 B)
;
; Build the PyInstaller bundle first (packaging\build.ps1 does this, or run
; manually), then compile this script with ISCC:
;
;     "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\ai-video-enhancer.iss
;
; The installer bundles the PyInstaller onedir output plus FFmpeg under
; {app}\ffmpeg\ — aive.stages.ingest finds it automatically in frozen builds.

#define MyAppName "AI Video Enhancer"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Personal build"
#define MyAppExeName "ai-video-enhancer.exe"

[Setup]
; NOTE: keep AppId stable across versions so upgrades replace the old install.
AppId={{B7E4C9A2-5D31-4F6E-8A0C-D2E9F1A6B3C4}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\AI Video Enhancer
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=output
OutputBaseFilename=ai-video-enhancer-setup-{#MyAppVersion}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
; The bundle is several GB (torch + TensorRT); give the wizard room to breathe.
UninstallDisplayIcon={app}\{#MyAppExeName}

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; PyInstaller onedir output (dist\ai-video-enhancer\* = exe + _internal\)
Source: "..\dist\ai-video-enhancer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Bundled FFmpeg (Gyan full build) — copied to packaging\ffmpeg-bin by build.ps1
Source: "ffmpeg-bin\ffmpeg.exe"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion
Source: "ffmpeg-bin\ffprobe.exe"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
