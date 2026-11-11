$ErrorActionPreference='Stop'
. 'E:\UTEC\Tesis\TesisNicol\PFC\experiments\local_env.ps1'
$taskOutput=Join-Path (Get-Location) 'runs\controlled_landing_round4'
$taskArchitectures=@('lstm','mamba','mamba2','mamba3','gtrxl','ltc')
while($true){
    $taskReady=@($taskArchitectures | Where-Object { Test-Path -LiteralPath (Join-Path $taskOutput "hpo\${_}_selection.json") }).Count
    if($taskReady -eq 6){break}
    if(Select-String -LiteralPath (Join-Path $taskOutput 'hpo.log') -Pattern '^Traceback' -Quiet){throw 'HPO error: inspect hpo.log'}
    Start-Sleep -Seconds 15
}
Write-Output 'HPO complete;starting two bounded training groups,all30 models.'
python -u -m experiments.round4.parallel_train --workers 2 *> (Join-Path $taskOutput 'final_training.log')
if($LASTEXITCODE -ne 0){throw 'Training failed: inspect final_training.log'}
Write-Output 'Training complete;freezing models and opening final paired tests.'
$env:NICOL_EVAL_WORKERS='8'
python -u -m experiments.round4.parallel_evaluate *> (Join-Path $taskOutput 'final_evaluation.log')
if($LASTEXITCODE -ne 0){throw 'Evaluation failed: inspect final_evaluation.log'}
python -u -m experiments.round4.analyze *> (Join-Path $taskOutput 'analysis.log')
if($LASTEXITCODE -ne 0){throw 'Analysis failed: inspect analysis.log'}
Write-Output 'Round4 training,evaluation and report complete.'
