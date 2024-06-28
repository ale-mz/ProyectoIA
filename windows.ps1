# ANSI color codes
$COLOR_RESET = "`e[0m"
$COLOR_BOLD = "`e[1m"
$COLOR_RED = "`e[91m"
$COLOR_GREEN = "`e[92m"
$COLOR_YELLOW = "`e[93m"
$COLOR_CYAN = "`e[96m"
$COLOR_MAGENTA = "`e[95m"

# Function to show the help message
function Show-Help {
    Write-Host "$COLOR_BOLD`Usage: .\train.ps1 <target>`$COLOR_RESET"
    Write-Host ""
    Write-Host "$COLOR_BOLD`Possible targets:`$COLOR_RESET"
    Write-Host "$COLOR_GREEN`  help`$COLOR_RESET            : Show this usage message"
    Write-Host "$COLOR_GREEN`  clear`$COLOR_RESET           : Clear the terminal"
    Write-Host "$COLOR_GREEN`  adapt`$COLOR_RESET           : Add more resources to the terminal to prevent killing $COLOR_RED`(Careful with this command)`$COLOR_RESET"
    Write-Host "$COLOR_GREEN`  train-tunning`$COLOR_RESET   : Train the model with tuning mode $COLOR_YELLOW`activated`$COLOR_RESET"
    Write-Host "                           $COLOR_CYAN`(Shorter training for testing, parameter tuning, and configuration)`$COLOR_RESET"
    Write-Host "$COLOR_GREEN`  train-tuned`$COLOR_RESET     : Train the model with tuning mode $COLOR_MAGENTA`deactivated`$COLOR_RESET"
    Write-Host "                           $COLOR_CYAN`(Full training to have the model ready to be used)`$COLOR_RESET"
    Write-Host "$COLOR_GREEN`  train`$COLOR_RESET           : Alias for '$COLOR_GREEN`train-tunning`$COLOR_RESET'"
}

# Function to clear the terminal
function Clear-Terminal {
    Clear-Host
}

# Function to assign unlimited resources to the terminal
function Adapt-Terminal {
    # Note: PowerShell does not have a direct equivalent to ulimit
    Write-Host "$COLOR_RED`Warning: PowerShell does not support resource limit management natively`$COLOR_RESET"
}

# Function to train the model with tuning mode on
function Train-Tunning {
    Write-Host "$COLOR_YELLOW`Running program with tuning mode activated`$COLOR_RESET"
    $env:TF_ENABLE_ONEDNN_OPTS = 0
    python src/train/main.py 1
}

# Function to train the model with tuning mode off
function Train-Tuned {
    Write-Host "$COLOR_MAGENTA`Running program with tuning mode deactivated`$COLOR_RESET"
    $env:TF_ENABLE_ONEDNN_OPTS = 0
    python src/train/main.py 0
}

# Main script
if ($args.Count -eq 0) {
    Show-Help
    exit 1
}

switch ($args[0]) {
    "help" {
        Show-Help
    }
    "clear" {
        Clear-Terminal
    }
    "adapt" {
        Adapt-Terminal
    }
    "train-tunning" {
        Train-Tunning
    }
    "train-tuned" {
        Train-Tuned
    }
    "train" {
        Train-Tunning
    }
    default {
        Write-Host "$COLOR_RED`Unknown target: $($args[0])`$COLOR_RESET"
        Show-Help
        exit 1
    }
}
