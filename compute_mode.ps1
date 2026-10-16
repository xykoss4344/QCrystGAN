# Switch every QGAN job (training, relax, LeMat, MatterGen, bench queue + children)
# between "low" (Idle priority, cores 8-15 only: the PC stays usable) and
# "full" (Normal priority, all 16 cores).
#   powershell -File compute_mode.ps1 low|full
param([ValidateSet('low','full')][string]$mode)
Set-Content runs\compute_mode.txt $mode   # job_scheduler.ps1 launches new jobs in this mode
$pri = @{low='Idle'; full='Normal'}[$mode]
$aff = @{low=0xFF00; full=0xFFFF}[$mode]
$match = 'train_crystal|relax_cifs|run_benchmarks|mattergen|bench_queue|bench_full|sun_track|spawn_main'
$ids = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $match -and $_.CommandLine -notmatch 'shell-snapshots' } |
       Select-Object -ExpandProperty ProcessId
# In full mode trainings sit one notch below the benchmarks: LeMat starved of CPU times out and drops structures.
$train = Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'train_crystal' } |
         Select-Object -ExpandProperty ProcessId
foreach ($i in $ids) { try { $p = Get-Process -Id $i -ErrorAction Stop
    $p.PriorityClass = if ($mode -eq 'full' -and $train -contains $i) { 'BelowNormal' } else { $pri }
    $p.ProcessorAffinity = [IntPtr]$aff } catch {} }
"$mode`: $($ids.Count) processes -> $pri, affinity 0x{0:X}" -f $aff
