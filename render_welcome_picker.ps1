Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$ErrorActionPreference = 'Stop'

function Pick-File([string]$title, [string]$filter) {
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = $title
    $dlg.Filter = $filter
    $dlg.Multiselect = $false
    $dlg.RestoreDirectory = $true
    $dlg.InitialDirectory = [Environment]::GetFolderPath('MyVideos')
    if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
        throw "Selection cancelled: $title"
    }
    return $dlg.FileName
}

function Find-Exe([string]$exeName, [string[]]$candidates) {
    foreach ($c in $candidates) {
        if ($c -and (Test-Path $c)) { return $c }
    }
    $cmd = Get-Command $exeName -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source) { return $cmd.Source }
    return $null
}

$ffmpeg = Find-Exe 'ffmpeg.exe' @(
    'C:\Users\qkrrn\AppData\Local\ChatGPT_RIFE4090\tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe',
    (Get-ChildItem "$env:LOCALAPPDATA\ChatGPT_RIFE4090\tools\ffmpeg" -Recurse -Filter 'ffmpeg.exe' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName)
)

$ffprobe = Find-Exe 'ffprobe.exe' @(
    'C:\Users\qkrrn\AppData\Local\ChatGPT_RIFE4090\tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin\ffprobe.exe',
    (Get-ChildItem "$env:LOCALAPPDATA\ChatGPT_RIFE4090\tools\ffmpeg" -Recurse -Filter 'ffprobe.exe' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName)
)

if (-not $ffmpeg -or -not $ffprobe) {
    [System.Windows.Forms.MessageBox]::Show('ffmpeg / ffprobe not found.', 'Error') | Out-Null
    exit 1
}

try {
    $video = Pick-File 'Select source video' 'Video Files|*.mp4;*.mov;*.mkv;*.avi|All Files|*.*'
    $image = Pick-File 'Select title image for 39-49 sec' 'Image Files|*.png;*.jpg;*.jpeg;*.webp;*.bmp|All Files|*.*'

    $durationText = (& $ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 $video).Trim()
    $duration = [double]::Parse($durationText, [System.Globalization.CultureInfo]::InvariantCulture)

    $dir = Split-Path $video -Parent
    $base = [System.IO.Path]::GetFileNameWithoutExtension($video)
    $out = Join-Path $dir ($base + '_WELCOME_39_49.mp4')
    if (Test-Path $out) { Remove-Item $out -Force }

    $fc = @"
[1:v]scale=-2:520,format=rgba,colorkey=0x000000:0.18:0.06,fade=t=in:st=39:d=0.85:alpha=1,fade=t=out:st=48.15:d=0.85:alpha=1[title];
[0:v][title]overlay=x=(W-w)/2:y='if(lt(t,39.85),1040-12*(t-39)/0.85,if(gt(t,48.15),1028+10*(t-48.15)/0.85,1028))':enable='between(t,39,49)':format=auto[v]
"@

    Write-Host "Video  : $video"
    Write-Host "Image  : $image"
    Write-Host "Output : $out"

    & $ffmpeg -y `
        -i $video `
        -loop 1 -framerate 60 -i $image `
        -filter_complex $fc `
        -map '[v]' -map '0:a?' `
        -t $duration `
        -c:v libx264 -preset fast -crf 14 `
        -profile:v high -level:v 5.2 -pix_fmt yuv420p `
        -c:a copy `
        -movflags +faststart `
        $out

    if ($LASTEXITCODE -ne 0) { throw "FFmpeg failed with code $LASTEXITCODE" }

    & $ffprobe -v error -show_entries stream=codec_name,width,height,avg_frame_rate -show_entries format=duration,size -of default=nw=1 $out
    Start-Process explorer.exe -ArgumentList "/select,`"$out`""
    [System.Windows.Forms.MessageBox]::Show("Completed.`n`n$out", 'Done') | Out-Null
}
catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Error') | Out-Null
    exit 1
}
