Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$ErrorActionPreference = 'Stop'

function Pick-File([string]$title, [string]$filter) {
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = $title
    $dlg.Filter = $filter
    $dlg.Multiselect = $false
    $dlg.RestoreDirectory = $true
    if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
        throw "선택이 취소되었습니다: $title"
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
    [System.Windows.Forms.MessageBox]::Show("ffmpeg / ffprobe를 찾지 못했습니다.", "오류") | Out-Null
    exit 1
}

try {
    $video = Pick-File '원본 영상 선택 (39~49초에 문구 적용)' 'Video Files|*.mp4;*.mov;*.mkv;*.avi|All Files|*.*'
    $image = Pick-File '문구 이미지 선택 (PNG/JPG/WebP)' 'Image Files|*.png;*.jpg;*.jpeg;*.webp;*.bmp|All Files|*.*'

    $durationText = (& $ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 $video).Trim()
    $duration = [double]::Parse($durationText, [System.Globalization.CultureInfo]::InvariantCulture)

    $dir = Split-Path $video -Parent
    $base = [System.IO.Path]::GetFileNameWithoutExtension($video)
    $out = Join-Path $dir ($base + '_WELCOME_39_49.mp4')

    if (Test-Path $out) { Remove-Item $out -Force }

    # 39~49초 / 자연스럽게 등장-유지-퇴장
    # 투명 PNG 권장. 검은 배경 이미지는 colorkey로 자동 제거.
    $fc = @"
[1:v]scale=-2:520,format=rgba,colorkey=0x000000:0.18:0.06,fade=t=in:st=39:d=0.85:alpha=1,fade=t=out:st=48.15:d=0.85:alpha=1[title];
[0:v][title]overlay=x=(W-w)/2:y='if(lt(t,39.85),1040-12*(t-39)/0.85,if(gt(t,48.15),1028+10*(t-48.15)/0.85,1028))':enable='between(t,39,49)':format=auto[v]
"@

    Write-Host "[INFO] Video   : $video"
    Write-Host "[INFO] Image   : $image"
    Write-Host "[INFO] Output  : $out"
    Write-Host "[INFO] Duration: $duration"

    & $ffmpeg -y `
        -i $video `
        -loop 1 -framerate 60 -i $image `
        -filter_complex $fc `
        -map '[v]' -map 0:a? `
        -t $duration `
        -c:v libx264 -preset fast -crf 14 `
        -profile:v high -level:v 5.2 -pix_fmt yuv420p `
        -c:a copy `
        -movflags +faststart `
        $out

    if ($LASTEXITCODE -ne 0) { throw 'FFmpeg 렌더링 실패' }

    & $ffprobe -v error `
        -show_entries stream=codec_name,width,height,avg_frame_rate `
        -show_entries format=duration,size `
        -of default=nw=1 `
        $out

    Start-Process explorer.exe -ArgumentList "/select,`"$out`""
    [System.Windows.Forms.MessageBox]::Show("완료되었습니다.`n`n$out", "완료") | Out-Null
}
catch {
    Write-Error $_
    [System.Windows.Forms.MessageBox]::Show($_.Exception.Message, "실행 중 오류") | Out-Null
    exit 1
}
