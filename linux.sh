#!/bin/bash

# ANSI color codes
COLOR_RESET="\033[0m"
COLOR_BOLD="\033[1m"
COLOR_RED="\033[91m"
COLOR_GREEN="\033[92m"
COLOR_YELLOW="\033[93m"
COLOR_CYAN="\033[96m"
COLOR_MAGENTA="\033[95m"

# Function to show the help message
function show_help() {
    echo -e "${COLOR_BOLD}Usage: $0 <target>${COLOR_RESET}"
    echo ""
    echo -e "${COLOR_BOLD}Possible targets:${COLOR_RESET}"
    echo -e "${COLOR_GREEN}  help${COLOR_RESET}            : Show this usage message"
    echo -e "${COLOR_GREEN}  clear${COLOR_RESET}           : Clear the terminal"
    echo -e "${COLOR_GREEN}  adapt${COLOR_RESET}           : Add more resources to the terminal to prevent killing ${COLOR_RED}(Careful with this command)${COLOR_RESET}"
    echo -e "${COLOR_GREEN}  train-tunning${COLOR_RESET}   : Train the model with tuning mode ${COLOR_YELLOW}activated${COLOR_RESET}"
    echo -e "                           ${COLOR_CYAN}(Shorter training for testing, parameter tuning, and configuration)${COLOR_RESET}"
    echo -e "${COLOR_GREEN}  train-tuned${COLOR_RESET}     : Train the model with tuning mode ${COLOR_MAGENTA}deactivated${COLOR_RESET}"
    echo -e "                           ${COLOR_CYAN}(Full training to have the model ready to be used)${COLOR_RESET}"
    echo -e "${COLOR_GREEN}  train${COLOR_RESET}           : Alias for '${COLOR_GREEN}train-tunning${COLOR_RESET}'"
}

# Function to clear the terminal
function clear_terminal() {
    clear
}

# Function to assign unlimited resources to the terminal
function adapt_terminal() {
    ulimit -t unlimited
}

# Function to train the model with tuning mode on
function train_tunning() {
    echo -e "${COLOR_YELLOW}Running program with tuning mode activated${COLOR_RESET}"
    export TF_ENABLE_ONEDNN_OPTS=0
    python3 src/train/main.py 1
}

# Function to train the model with tuning mode off
function train_tuned() {
    echo -e "${COLOR_MAGENTA}Running program with tuning mode deactivated${COLOR_RESET}"
    export TF_ENABLE_ONEDNN_OPTS=0
    python3 src/train/main.py 0
}

# Main script
if [[ $# -eq 0 ]]; then
    show_help
    exit 1
fi

case "$1" in
    help)
        show_help
        ;;
    clear)
        clear_terminal
        ;;
    adapt)
        adapt_terminal
        ;;
    train-tunning)
        train_tunning
        ;;
    train-tuned)
        train_tuned
        ;;
    train)
        train_tunning
        ;;
    *)
        echo -e "${COLOR_RED}Unknown target: $1${COLOR_RESET}"
        show_help
        exit 1
        ;;
esac
