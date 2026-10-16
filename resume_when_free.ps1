# Resume the PIDs in runs/suspended_pids.txt once bench_queue.log reports "queue done".
Add-Type -Name Nt -Namespace W -MemberDefinition '[DllImport("ntdll.dll")] public static extern int NtResumeProcess(IntPtr h);'
while (-not (Select-String -Path bench_queue.log -Pattern 'queue done' -Quiet)) { Start-Sleep 60 }
foreach ($i in Get-Content runs\suspended_pids.txt) { try { [void][W.Nt]::NtResumeProcess((Get-Process -Id ([int]$i) -ErrorAction Stop).Handle) } catch {} }
Clear-Content runs\suspended_pids.txt
Add-Content runs\PLAN.md "- $(Get-Date -Format 'MM-dd HH:mm') bench_queue done -> resumed MatterGen n=300 LeMat (resume_when_free.ps1)."
