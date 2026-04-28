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

    [string]$BranchTag = "",

    [string]$OutputRootBase = "output_branches",

    [string]$ReportsRootBase = "reports_branch",

    [string]$CompareAgainstBranchTag = "",

    [string]$ComparisonOutputDir = "reports_compare",

    [switch]$SkipRcaAfterTraining
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Get-CurrentBranchTag {
    param(
        [string]$ProvidedBranchTag
    )

    if (-not [string]::IsNullOrWhiteSpace($ProvidedBranchTag)) {
        return $ProvidedBranchTag
    }

    $branchName = (& git branch --show-current).Trim()
    if ([string]::IsNullOrWhiteSpace($branchName)) {
        throw "Unable to determine current git branch. Pass -BranchTag explicitly."
    }

    return ($branchName -replace "[^A-Za-z0-9._-]", "_")
}

function Get-OutputRoot {
    param(
        [string]$RootBase,
        [string]$ResolvedBranchTag
    )

    return Join-Path -Path $RootBase -ChildPath $ResolvedBranchTag
}

function Get-ReportsDir {
    param(
        [string]$RootBase,
        [string]$ResolvedBranchTag
    )

    return Join-Path -Path $RootBase -ChildPath $ResolvedBranchTag
}

function Get-CommonArgs {
    return @(
        "--dataset", "SMD",
        "--output_root", $script:OutputRoot,
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
        Write-Host ("[{0}] branch={1} group={2} -> {3}" -f $Experiment.Id, $script:ResolvedBranchTag, $group, $Experiment.Description) -ForegroundColor Cyan
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

    $baseDir = Join-Path -Path $PSScriptRoot -ChildPath (Join-Path -Path $script:OutputRoot -ChildPath ("SMD\" + $Group))
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
                Write-Warning "No run found for $experimentId on group $group under $($script:OutputRoot). Skipping RCA."
                continue
            }

            Write-Host ""
            Write-Host ("[RCA] branch={0} {1} group={2}" -f $script:ResolvedBranchTag, $experimentId, $group) -ForegroundColor Yellow
            Write-Host ("python analyze_results.py --dataset SMD --output_root {0} --group {1} --target_dir `"{2}`"" -f $script:OutputRoot, $group, $targetDir) -ForegroundColor DarkGray
            & python "analyze_results.py" "--dataset" "SMD" "--output_root" $script:OutputRoot "--group" $group "--target_dir" $targetDir
            if ($LASTEXITCODE -ne 0) {
                throw "RCA analysis failed for experiment $experimentId on group $group."
            }
        }
    }
}

function Invoke-SummaryReport {
    param(
        [string[]]$Groups,
        [string[]]$ExperimentIds
    )

    $reportsDir = Join-Path -Path $PSScriptRoot -ChildPath $script:ReportsDir
    if (-not (Test-Path -LiteralPath $reportsDir)) {
        New-Item -ItemType Directory -Path $reportsDir | Out-Null
    }

    $cmdArgs = @(
        "summarize_experiments.py",
        "--dataset", "SMD",
        "--output_root", $script:OutputRoot,
        "--output_dir", $script:ReportsDir,
        "--groups"
    ) + $Groups + @(
        "--experiments"
    ) + $ExperimentIds

    Write-Host ""
    Write-Host ("[REPORT] branch={0}" -f $script:ResolvedBranchTag) -ForegroundColor Green
    Write-Host ("python " + ($cmdArgs -join " ")) -ForegroundColor DarkGray
    & python @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Summary report generation failed."
    }
}

function Invoke-BranchComparison {
    param(
        [string]$BaselineBranchTag,
        [string]$CandidateBranchTag
    )

    if ([string]::IsNullOrWhiteSpace($BaselineBranchTag)) {
        return
    }

    $baselineOutputRoot = Get-OutputRoot -RootBase $OutputRootBase -ResolvedBranchTag $BaselineBranchTag
    $candidateOutputRoot = Get-OutputRoot -RootBase $OutputRootBase -ResolvedBranchTag $CandidateBranchTag
    $baselineReportsDir = Get-ReportsDir -RootBase $ReportsRootBase -ResolvedBranchTag $BaselineBranchTag
    $candidateReportsDir = Get-ReportsDir -RootBase $ReportsRootBase -ResolvedBranchTag $CandidateBranchTag

    $cmdArgs = @(
        "compare_branch_results.py",
        "--baseline_root", $baselineOutputRoot,
        "--candidate_root", $candidateOutputRoot,
        "--baseline_report_dir", $baselineReportsDir,
        "--candidate_report_dir", $candidateReportsDir,
        "--output_dir", $ComparisonOutputDir
    )

    Write-Host ""
    Write-Host ("[COMPARE] baseline={0} candidate={1}" -f $BaselineBranchTag, $CandidateBranchTag) -ForegroundColor Magenta
    Write-Host ("python " + ($cmdArgs -join " ")) -ForegroundColor DarkGray
    & python @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Branch comparison report generation failed."
    }
}

$script:ResolvedBranchTag = Get-CurrentBranchTag -ProvidedBranchTag $BranchTag
$script:OutputRoot = Get-OutputRoot -RootBase $OutputRootBase -ResolvedBranchTag $script:ResolvedBranchTag
$script:ReportsDir = Get-ReportsDir -RootBase $ReportsRootBase -ResolvedBranchTag $script:ResolvedBranchTag

Write-Host ("Using branch tag: {0}" -f $script:ResolvedBranchTag) -ForegroundColor Green
Write-Host ("Output root: {0}" -f $script:OutputRoot) -ForegroundColor Green
Write-Host ("Reports dir: {0}" -f $script:ReportsDir) -ForegroundColor Green

$experiments = Get-ExperimentDefinitions

switch ($Stage) {
    "compare" {
        Invoke-TrainingExperiment -Experiment $experiments["E1"] -Groups $CompareGroups
        Invoke-TrainingExperiment -Experiment $experiments["E2"] -Groups $CompareGroups
        if (-not $SkipRcaAfterTraining) {
            Invoke-RcaAnalysis -Groups $CompareGroups -ExperimentIds @("E2")
        }
        Invoke-SummaryReport -Groups $CompareGroups -ExperimentIds @("E1", "E2")
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
        Invoke-SummaryReport -Groups $AblationGroups -ExperimentIds @("E2", "E3", "E4", "E5", "E6")
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
        Invoke-SummaryReport -Groups @($CompareGroups + $AblationGroups | Select-Object -Unique) -ExperimentIds @("E1", "E2", "E3", "E4", "E5", "E6")
    }
    "rca" {
        Invoke-RcaAnalysis -Groups $CompareGroups -ExperimentIds @("E2")
        Invoke-RcaAnalysis -Groups $AblationGroups -ExperimentIds @("E6")
        Invoke-SummaryReport -Groups @($CompareGroups + $AblationGroups | Select-Object -Unique) -ExperimentIds @("E1", "E2", "E3", "E4", "E5", "E6")
    }
}

Invoke-BranchComparison -BaselineBranchTag $CompareAgainstBranchTag -CandidateBranchTag $script:ResolvedBranchTag

Write-Host ""
Write-Host "Done." -ForegroundColor Green
