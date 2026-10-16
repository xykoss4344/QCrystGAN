# Launch each line of runs/job_queue.txt (launch_like.py arguments) one at a time,
# only when free RAM > $minGB, then wait 10 min for it to reach steady memory.
# Done lines move to runs/job_queue_done.txt. Runs until the queue is empty.
# New launches follow runs\compute_mode.txt (written by compute_mode.ps1), capped at $maxJobs.
$minGB = 9
$maxJobs = @{low=5; full=6}  # 8 runs: 1.3 GB VRAM free; 7 runs crept back to 1.45 GB within 3 h (10-15). 6 keeps headroom
$minVramMB = 3500  # 6 runs sit at ~4.1 GB free; 6500 blocked even one-for-one replacement (10-13 22:00). User accepted ~2 GB free with 8 runs.
$py = 'C:\Users\Adminb\AppData\Local\Python\pythoncore-3.12-64\python.exe'
while ($true) {
    $q = @(Get-Content runs\job_queue.txt | Where-Object { $_.Trim() })
    if ($q.Count -eq 0) { break }
    $mode = "$(Get-Content runs\compute_mode.txt -ErrorAction SilentlyContinue)".Trim()
    if ($mode -ne 'full') { $mode = 'low' }
    $running = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -match 'train_crystal' }).Count
    $free = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    # VRAM headroom: a full GPU crashed the NVIDIA driver on 10-09 01:09 (black screen on wake, reboot needed).
    $vram = [int](nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    if ($free -gt $minGB -and $running -lt $maxJobs[$mode] -and $vram -gt $minVramMB) {
        $out = & $py launch_like.py ($q[0] -split ' ')
        Add-Content runs\job_queue_done.txt "$(Get-Date -Format 'MM-dd HH:mm') free=$([math]::Round($free,1))GB $($q[0]) -> $out"
        Set-Content runs\job_queue.txt ($q | Select-Object -Skip 1)
        & powershell -ExecutionPolicy Bypass -File compute_mode.ps1 $mode | Out-Null
        Start-Sleep 900  # let the new run ramp its VRAM before the next check (180 s let 4 launch in 10 min, 10-13)
    } else { Start-Sleep 120 }
}
