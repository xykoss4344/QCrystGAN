# When the oxide pretraining reaches epoch 50, queue the Mg-Mn-O stable fine-tune
# (full w23 recipe: force + relaxation distillation back on, Mg-Mn-O volume prior).
while (-not (Test-Path runs\p1_oxide_pretrain_s1\checkpoint_50.pt)) { Start-Sleep 300 }
Add-Content runs\job_queue.txt "runs/s1_stable10_w23_s1 runs/p2_pretrain_ft_stable_s1 resume_checkpoint=runs/p1_oxide_pretrain_s1/checkpoint_50.pt n_epochs=200"
Add-Content runs\PLAN.md "- $(Get-Date -Format 'MM-dd HH:mm') pretrain ep50 reached -> queued p2_pretrain_ft_stable_s1 (chain_pretrain_ft.ps1)."
