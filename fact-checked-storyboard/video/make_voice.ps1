# The Windows voice reads each line of script.json into voice/<scene>-<line>.wav (24 kHz, 16-bit, mono).
# Only the characters change for the speech engine (non-breaking hyphen -> "-", curly apostrophe -> "'"); the words stay.
param([string]$Voice = 'Microsoft Zira Desktop', [int]$Rate = -1)
Add-Type -AssemblyName System.Speech
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$scenes = Get-Content (Join-Path $here 'script.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$out = Join-Path $here 'voice'
New-Item -ItemType Directory -Force $out | Out-Null
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(24000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.SelectVoice($Voice)
$speaker.Rate = $Rate
foreach ($scene in $scenes) {
    $i = 0
    foreach ($line in $scene.lines) {
        $i++
        $file = Join-Path $out ("{0}-{1}.wav" -f $scene.scene, $i)
        $speaker.SetOutputToWaveFile($file, $format)
        $speaker.Speak(($line -replace [char]0x2011, '-' -replace [char]0x2019, "'"))
        $speaker.SetOutputToNull()
        "{0}  {1}" -f (Split-Path -Leaf $file), $line
    }
}
$speaker.Dispose()
