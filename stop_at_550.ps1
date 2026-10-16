# Stop 560-epoch trainings once checkpoint_550.pt exists: the paper samples at epoch 550,
# so epochs 551-560 are unused. Exits when no training runs are left.
while ($true) {
    $procs = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object CommandLine -match 'train_crystal')
    if ($procs.Count -eq 0 -and -not (Get-Content runs\job_queue.txt | Where-Object { $_.Trim() })) { break }
    foreach ($p in $procs) {
        if ($p.CommandLine -notmatch '--n_epochs 560\b' -or $p.CommandLine -notmatch '--out_folder (\S+)') { continue }
        $ck = Get-Item "$($matches[1])\checkpoint_550.pt" -ErrorAction SilentlyContinue
        # 60 s settle so a checkpoint still being written is never cut off
        if ($ck -and ((Get-Date) - $ck.LastWriteTime).TotalSeconds -gt 60) {
            Stop-Process -Id $p.ProcessId -Force
            Add-Content runs\stop_at_550.log "$(Get-Date -Format 'MM-dd HH:mm') stopped $($matches[1]) pid $($p.ProcessId)"
        }
    }
    Start-Sleep 120
}
