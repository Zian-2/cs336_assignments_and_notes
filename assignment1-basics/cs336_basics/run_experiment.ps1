# run_experiments.ps1

# 在脚本开头添加：等待 3 小时 (3 * 3600 秒)
Write-Host "starting" -ForegroundColor Yellow
Start-Sleep -Seconds (3 * 3600)

# 原有的实验配置
$experiments = @(
    @{ name="bs16_lr1e-3";  bs=16; steps=10000; lr=1e-3 }
)

# ... 后续循环代码保持不变 ...
$experiments = @(
    
    @{ name="bs16_lr1e-3";  bs=16; steps=10000; lr=1e-3 }

)

foreach ($exp in $experiments) {
    Write-Host "start$($exp.name)" -ForegroundColor Cyan
    
    # 计算 warmup_iters，确保是整数
    $warmup = [Math]::Floor($exp.steps / 10)
    
    uv run python transformer_training_loop.py `
        --data_path "../../data/owt_train.npy" `
        --val_data_path "../../data/owt_valid.npy" `
        --use_wandb `
        --batch_size $($exp.bs) `
        --context_length 256 `
        --d_model 512 `
        --num_layers 4 `
        --num_heads 16 `
        --d_ff 1344 `
        --max_iters $($exp.steps) `
        --warmup_iters $warmup `
        --lr $($exp.lr) `
        --device "cuda"
    
    Write-Host "$($exp.name) completed" -ForegroundColor Green


    
    # 关键：显式等待，给 Windows 驱动回收显存的时间
    Start-Sleep -Seconds 10
}