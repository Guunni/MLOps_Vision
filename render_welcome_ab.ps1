#requires -Version 5.1
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Add-Type -AssemblyName System.Windows.Forms

$VideoDir = Join-Path $env:USERPROFILE 'Downloads\디즈니 영상'
$Downloads = Join-Path $env:USERPROFILE 'Downloads'
$Desktop = [Environment]::GetFolderPath('Desktop')
function Fail([string]$msg) { throw $msg }
if (-not (Test-Path -LiteralPath $VideoDir -PathType Container)) { Fail "영상 폴더 없음: $VideoDir" }

$SourceNames = @(
  'Disney_intro_book_AI_4K60_WEDDING_TITLE_FINAL_v3.mp4',
  'Disney_intro_book_AI_4K60_WEDDING_TITLE_FINAL_v2.mp4',
  'Disney_intro_book_AI_4K60_WEDDING_TITLE_FINAL.mp4',
  'Disney_intro_book_AI_4K60_FINAL_EXACT.mp4'
)
$InputVideo = $null
foreach ($name in $SourceNames) {
  $candidate = Join-Path $VideoDir $name
  if (Test-Path -LiteralPath $candidate -PathType Leaf) { $InputVideo = $candidate; break }
}
if (-not $InputVideo) { Fail "원본 영상이 없습니다. $VideoDir 확인 필요" }

$Ffmpeg = Join-Path $env:LOCALAPPDATA 'ChatGPT_RIFE4090\tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin\ffmpeg.exe'
$Ffprobe = Join-Path $env:LOCALAPPDATA 'ChatGPT_RIFE4090\tools\ffmpeg\ffmpeg-9.0.2-essentials_build\bin\ffprobe.exe'
if (-not (Test-Path -LiteralPath $Ffmpeg)) {
  $found = Get-Command ffmpeg.exe -ErrorAction SilentlyContinue
  if ($found) { $Ffmpeg = $found.Source }
}
if (-not (Test-Path -LiteralPath $Ffprobe)) {
  $found = Get-Command ffprobe.exe -ErrorAction SilentlyContinue
  if ($found) { $Ffprobe = $found.Source }
}
if (-not (Test-Path -LiteralPath $Ffmpeg) -or -not (Test-Path -LiteralPath $Ffprobe)) {
  Fail "FFmpeg 또는 FFprobe를 찾을 수 없습니다. 기존 RIFE4090 도구 경로 확인 필요"
}

function Find-Image([string]$Label, [string[]]$Names) {
  $roots = @($VideoDir, $Downloads, $Desktop)
  foreach ($root in $roots) {
    foreach ($name in $Names) {
      $candidate = Join-Path $root $name
      if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        Write-Host "[$Label] 이미지: $candidate" -ForegroundColor Cyan
        return $candidate
      }
    }
  }
  foreach ($name in $Names) {
    $found = Get-ChildItem -LiteralPath $Downloads -Recurse -Filter $name -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
      Write-Host "[$Label] 이미지: $($found.FullName)" -ForegroundColor Cyan
      return $found.FullName
    }
  }
  Write-Host "[$Label] PNG를 선택하세요." -ForegroundColor Yellow
  $dlg = New-Object System.Windows.Forms.OpenFileDialog
  $dlg.Title = "$Label 웰컴 타이틀 PNG"
  $dlg.InitialDirectory = $Downloads
  $dlg.Filter = 'PNG files (*.png)|*.png|All files (*.*)|*.*'
  $dlg.Multiselect = $false
  if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
    Fail "$Label 파일 선택이 취소되었습니다."
  }
  return $dlg.FileName
}
$ImageA = Find-Image 'A' @(
  '청록빛_판타지_웨딩_타이틀_장식.png','WELCOME_A.png','WELCOME_VER_A.png','Welcome_To_Our_Wedding_A.png'
)
$ImageB = Find-Image 'B' @(
  '우아한_판타지_웨딩_환영_타이포그래피.png','WELCOME_B.png','WELCOME_VER_B.png','Welcome_To_Our_Wedding_B.png'
)
if ([String]::Equals($ImageA,$ImageB,[StringComparison]::OrdinalIgnoreCase)) {
  Fail "A/B 같은 PNG가 선택되었습니다. 서로 다른 시안을 선택하세요."
}

$culture = [Globalization.CultureInfo]::InvariantCulture
$durationRaw = (& $Ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$InputVideo" | Out-String).Trim()
if ($LASTEXITCODE -ne 0 -or -not $durationRaw) { Fail "원본 ffprobe 분석 실패" }
$duration = [double]::Parse($durationRaw,$culture)
if ($duration -lt 49) { Fail "원본 영상 길이가 49초보다 짧습니다." }
$durationText = $duration.ToString('0.000000',$culture)
$totalFrames = [int][Math]::Round($duration*60)

# Image has black backdrop? Screen blend overlays light and particles without a dark rectangle.
$graphTemplate = @'
color=c=black:s=3840x2160:r=60:d={DURATION},format=rgba[canvas];
[1:v]scale=2600:-2:flags=lanczos,format=rgba[title];
[canvas][title]overlay=x=(W-w)/2:y=(H-h)/2:shortest=1:format=auto,
format=gbrp,fade=t=in:st=39:d=0.85,
fade=t=out:st=48.15:d=0.85[graphic];
[0:v]format=gbrp[base];
[base][graphic]blend=all_mode=screen,format=yuv420p[v]
'@
$graph = $graphTemplate.Replace('{DURATION}',$durationText) -replace '[\r\n]',''

Write-Host ''
Write-Host "=== WELCOME A/B 4K60 비교 영상 렌더링 ===" -ForegroundColor Green
Write-Host "입력: $InputVideo"
Write-Host "A: $ImageA"
Write-Host "B: $ImageB"
Write-Host '39.0~49.0초 타이틀 / 등장 0.85초, 퇴장 0.85초'
Write-Host "$totalFrames 프레임, 길이 $durationText 초"
Write-Host ''

$jobs = @(
  @{ Label='A'; Image=$ImageA; Output=(Join-Path $VideoDir 'Disney_intro_book_AI_4K60_WELCOME_VER_A.mp4') },
  @{ Label='B'; Image=$ImageB; Output=(Join-Path $VideoDir 'Disney_intro_book_AI_4K60_WELCOME_VER_B.mp4') }
)
foreach ($job in $jobs) {
  Write-Host "[$($job.Label)] 렌더링 중: $($job.Output)" -ForegroundColor Cyan
  & $Ffmpeg -hide_banner -y `
    -i "$InputVideo" `
    -loop 1 -framerate 60 -i "$($job.Image)" `
    -filter_complex "$graph" `
    -map '[v]' -map '0:a?' `
    -t "$durationText" -frames:v $totalFrames -shortest `
    -c:v libx264 -preset fast -crf 15 -profile:v high -level:v 5.2 `
    -pix_fmt yuv420p -c:a copy -movflags +faststart `
    "$($job.Output)"
  if ($LASTEXITCODE -ne 0) { Fail "$($job.Label) 렌더링 실패. 위 FFmpeg 오류를 확인하세요." }

  $spec = (& $Ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$($job.Output)" | Out-String).Trim()
  if ($LASTEXITCODE -ne 0 -or $spec -notmatch '3840,2160,60/1') {
    Fail "$($job.Label) 출력 검증 실패: $spec"
  }
  $bytes = (Get-Item -LiteralPath $job.Output).Length
  Write-Host "[$($job.Label)] 완료: $([Math]::Round($bytes / 1MB,1)) MB / $spec" -ForegroundColor Green
}
Write-Host ''
Write-Host "완료: A/B 두 영상이 $VideoDir 에 저장됨" -ForegroundColor Green
Start-Process explorer.exe -ArgumentList ('/select,"' + $jobs[0].Output + '"')
