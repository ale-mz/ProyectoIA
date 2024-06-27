# Copyright Oscar Fernández, Jorge Madríz y Kenneth Villalobos
# Makefile to facilitate the usage of the program

# ANSI color codes
COLOR_RESET = \033[0m
COLOR_BOLD = \033[1m
COLOR_RED = \033[91m
COLOR_GREEN = \033[92m
COLOR_YELLOW = \033[93m
COLOR_CYAN = \033[96m
COLOR_MAGENTA = \033[95m

# Phony rules
.PHONY: help adapt train-tunning train-tuned train

# ----------------------------- General rules ---------------------------------
# Rule to show the help message, default for no rule indication
help:
	@echo "$(COLOR_BOLD)Usage: make <target>$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Possible targets:$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)  help$(COLOR_RESET)            : Show this usage message"
	@echo "$(COLOR_GREEN)  clear$(COLOR_RESET)           : Clear the terminal"
	@echo "$(COLOR_GREEN)  adapt$(COLOR_RESET)           : Add more resources to the terminal to prevent killing $(COLOR_RED)(Carefull with this command)$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)  train-tunning$(COLOR_RESET)   : Train the model with tuning mode $(COLOR_YELLOW)activated$(COLOR_RESET)"
	@echo "                           $(COLOR_CYAN)(Shorter training for testing, parameter tuning, and configuration)$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)  train-tuned$(COLOR_RESET)     : Train the model with tunning mode $(COLOR_MAGENTA)deactivated$(COLOR_RESET)"
	@echo "                           $(COLOR_CYAN)(Full training to have the model ready to be used)$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)  train$(COLOR_RESET)           : Alias for '$(COLOR_GREEN)train-tunning$(COLOR_RESET)'"

# Rule to clean the terminal
clear:
	clear

# Rule to assign unlimited resources to the terminal
adapt:
	ulimit -t unlimited

# ----------------------------- Training rules --------------------------------
# Rule to train the model with tunning mode on
train-tunning:
	@echo "$(COLOR_YELLOW)Running program with tuning mode activated$(COLOR_RESET)"
	export TF_ENABLE_ONEDNN_OPTS=0 && \
	python3 src/train/main.py 1
	

# Rule to train the model with tunning mode off
train-tuned:
	@echo "$(COLOR_MAGENTA)Running program with tuning mode deactivated$(COLOR_RESET)"
	export TF_ENABLE_ONEDNN_OPTS=0 && \
	nice -n -5 python3 src/train/main.py 0

# If no indication on the tunning is given, assume tunning mode on
train: train-tunning
