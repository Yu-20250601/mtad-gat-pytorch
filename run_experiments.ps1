param(
    [ValidateSet("compare", "ablation", "all", "rca")]
    [string]$Stage = "all",

    [string[]]$CompareGroups = @("1-2", "1-3", "2-1", "3-6"),

    [string[]]$AblationGroups = @("1-2", "3-6"),

    [int]$Epochs = 30,

    [int]$Lookback = 100,

    [int]$BatchSize = 256,

    [double]$InitLr = 1e-3,

    [double]$Dropout = 0.3,

    [double]$ValSplit = 0.1,

    [switch]$SkipRcaAfterTraining
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Get-CommonArgs {
    return @(
        "--dataset", "SMD",
        "--epochs", $Epochs.ToString(),
        "--lookback", $Lookback.ToString(),
        "--bs", $BatchSize.ToString(),
        "--init_lr", $InitLr.ToString(),
        "--dropout", $Dropout.ToString(),
        "--val_split", $ValSplit.ToString()
    )
}

function Get-ExperimentDefinitions {
    return @{
        "E1" = @{
            Id = "E1"
            Description = "Dense + GRU + GATv2"
            TrainArgs = @(
                "--use_adaptive_sparse_feat_gat", "False",
                "--recon_model", "gru",
                "--use_gatv2", "True",
                "--comment", "E1"
            )
            RunRca = $false
        }
        "E2" = @{
            Id = "E2"
            Description = "Sparse + NodeEmbed + VAE + GATv2"
            TrainArgs = @(
                "--use_adaptive_sparse_feat_gat", "True",
                "--use_node_embedding", "True",
                "--node_embed_dim", "16",
                "--recon_model", "vae",
                "--vae_latent_dim", "64",
                "--kl_beta", "0.001",
                "--use_gatv2", "True",
                "--export_sparse_attention", "True",
                "--comment", "E2"
            )
            RunRca = $true
        }
        "E3" = @{
            Id = "E3"
            Description = "Sparse + NodeEmbed + GRU + GATv2"
            TrainArgs = @(
                "--use_adaptive_sparse_feat_gat", "True",
                "--use_node_embedding", "True",
                "--node_embed_dim", "16",
                "--recon_model", "gru",
                "--use_gatv2", "True",
                "--export_sparse_attention", "True",
                "--comment", "E3"
            )
            RunRca = $false
        }
        "E4" = @{
            Id = "E4"
            Description = "Sparse + NodeEmbed + VAE + GAT"
            TrainArgs = @(
                "--use_adaptive_sparse_feat_gat", "True",
                "--use_node_embedding", "True",
                "--node_embed_dim", "16",
                "--recon_model", "vae",
                "--vae_latent_dim", "64",
                "--kl_beta", "0.001",
                "--use_gatv2", "False",
                "--export_sparse_attention", "True",
                "--comment", "E4"
            )
            RunRca = $false
        }
        "E5" = @{
            Id = "E5"
            Description = "Dense + VAE + GATv2"
            TrainArgs = @(
                "--use_adaptive_sparse_feat_gat", "False",
                "--recon_model", "vae",
                "--vae_latent_dim", "64",
                "--kl_beta", "0.001",
                "--use_gatv2", "True",
                "--comment", "E5"
            )
            RunRca = $false
        }
        "E6" = @{
            Id = "E6"
            Description = "Sparse + no NodeEmbed + VAE + GATv2"
            TrainArgs = @(
                "--use_adaptive_sparse_feat_gat", "True",
                "--use_node_embedding", "False",
                "--recon_model", "vae",
                "--vae_latent_dim", "64",
                "--kl_beta", "0.001",
                "--use_gatv2", "True",
                "--export_sparse_attention", "True",
                "--comment", "E6"
            )
            RunRca = $true
        }
    }
}

function Invoke-TrainingExperiment {
    param(
        [hashtable]$Experiment,
        [string[]]$Groups
    )

    $commonArgs = Get-CommonArgs
    foreach ($group in $Groups) {
        $cmdArgs = @("train.py") + $commonArgs + @("--group", $group) + $Experiment.TrainArgs
        Write-Host ""
        Write-Host ("[{0}] group={1} -> {2}" -f $Experiment.Id, $group, $Experiment.Description) -ForegroundColor Cyan
        Write-Host ("python " + ($cmdArgs -join " ")) -ForegroundColor DarkGray
        & python @cmdArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for experiment $($Experiment.Id) on group $group."
        }
    }
}

function Get-LatestRunDirByComment {
    param(
        [string]$Group,
        [string]$Comment
    )

    $baseDir = Join-Path -Path $PSScriptRoot -ChildPath ("output\SMD\" + $Group)
    if (-not (Test-Path -LiteralPath $baseDir)) {
        return $null
    }

    $runDirs = Get-ChildItem -LiteralPath $baseDir -Directory |
        Where-Object { $_.Name -match '^\d{8}_\d{6}$' } |
        Sort-Object LastWriteTime -Descending

    foreach ($dir in $runDirs) {
        $configPath = Join-Path -Path $dir.FullName -ChildPath "config.txt"
        if (-not (Test-Path -LiteralPath $configPath)) {
            continue
        }

        try {
            $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
        } catch {
            continue
        }

        if ($config.comment -eq $Comment) {
            return $dir.FullName
        }
    }

    return $null
}

function Invoke-RcaAnalysis {
    param(
        [string[]]$Groups,
        [string[]]$ExperimentIds = @("E2", "E6")
    )

    foreach ($group in $Groups) {
        foreach ($experimentId in $ExperimentIds) {
            $targetDir = Get-LatestRunDirByComment -Group $group -Comment $experimentId
            if ($null -eq $targetDir) {
                Write-Warning "No run found for $experimentId on group $group. Skipping RCA."
                continue
            }

            Write-Host ""
            Write-Host ("[RCA] {0} group={1}" -f $experimentId, $group) -ForegroundColor Yellow
            Write-Host ("python analyze_results.py --dataset SMD --group {0} --target_dir `"{1}`"" -f $group, $targetDir) -ForegroundColor DarkGray
            & python "analyze_results.py" "--dataset" "SMD" "--group" $group "--target_dir" $targetDir
            if ($LASTEXITCODE -ne 0) {
                throw "RCA analysis failed for experiment $experimentId on group $group."
            }
        }
    }
}

$experiments = Get-ExperimentDefinitions

switch ($Stage) {
    "compare" {
        Invoke-TrainingExperiment -Experiment $experiments["E1"] -Groups $CompareGroups
        Invoke-TrainingExperiment -Experiment $experiments["E2"] -Groups $CompareGroups
        if (-not $SkipRcaAfterTraining) {
            Invoke-RcaAnalysis -Groups $CompareGroups -ExperimentIds @("E2")
        }
    }
    "ablation" {
        Invoke-TrainingExperiment -Experiment $experiments["E2"] -Groups $AblationGroups
        Invoke-TrainingExperiment -Experiment $experiments["E3"] -Groups $AblationGroups
        Invoke-TrainingExperiment -Experiment $experiments["E4"] -Groups $AblationGroups
        Invoke-TrainingExperiment -Experiment $experiments["E5"] -Groups $AblationGroups
        Invoke-TrainingExperiment -Experiment $experiments["E6"] -Groups $AblationGroups
        if (-not $SkipRcaAfterTraining) {
            Invoke-RcaAnalysis -Groups $AblationGroups -ExperimentIds @("E2", "E6")
        }
    }
    "all" {
        Invoke-TrainingExperiment -Experiment $experiments["E1"] -Groups $CompareGroups
        Invoke-TrainingExperiment -Experiment $experiments["E2"] -Groups $CompareGroups
        Invoke-TrainingExperiment -Experiment $experiments["E3"] -Groups $AblationGroups
        Invoke-TrainingExperiment -Experiment $experiments["E4"] -Groups $AblationGroups
        Invoke-TrainingExperiment -Experiment $experiments["E5"] -Groups $AblationGroups
        Invoke-TrainingExperiment -Experiment $experiments["E6"] -Groups $AblationGroups
        if (-not $SkipRcaAfterTraining) {
            Invoke-RcaAnalysis -Groups $CompareGroups -ExperimentIds @("E2")
            Invoke-RcaAnalysis -Groups $AblationGroups -ExperimentIds @("E6")
        }
    }
    "rca" {
        Invoke-RcaAnalysis -Groups $CompareGroups -ExperimentIds @("E2")
        Invoke-RcaAnalysis -Groups $AblationGroups -ExperimentIds @("E6")
    }
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
