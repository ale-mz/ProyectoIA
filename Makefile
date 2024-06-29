# Copyright Oscar Fernández, Jorge Madríz y Kenneth Villalobos
# Makefile to facilitate the usage of the program

# Detect the operating system (allowd systems: windows and linux)
ifeq ($(OS),Windows_NT)
    detected_OS := Windows
else ifneq ($(findstring cmd.exe,$(COMSPEC)),)
    detected_OS := Windows
else ifdef MSVC
    detected_OS := Windows
else
    detected_OS := $(shell uname)
endif

# ANSI color codes
ifeq ($(detected_OS),Windows)
	COLOR_RESET = [0m
	COLOR_BOLD = [1m
	COLOR_RED = [31m
	COLOR_GREEN = [32m
	COLOR_YELLOW = [33m
	COLOR_CYAN = [36m
	COLOR_MAGENTA = [35m
else
	COLOR_RESET = \033[0m
	COLOR_BOLD = \033[1m
	COLOR_RED = \033[91m
	COLOR_GREEN = \033[92m
	COLOR_YELLOW = \033[93m
	COLOR_CYAN = \033[96m
	COLOR_MAGENTA = \033[95m
endif

# Phony rules
.PHONY: help clear adapt train-tunning train-tuned train evaluate

# ----------------------------- General rules ---------------------------------
# Rule to show the help message, default for no rule indication
help:
	@echo "$(COLOR_BOLD)Usage: make <target>$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Possible targets:$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)  help$(COLOR_RESET)					: Show this usage message"
	@echo "$(COLOR_GREEN)  clear$(COLOR_RESET)					: Clear the terminal"
	@echo "$(COLOR_GREEN)  adapt$(COLOR_RESET)					: Add more resources to the terminal to prevent killing $(COLOR_RED)(Careful with this command)$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)  install-deps$(COLOR_RESET)				: Install all the program related pytbon modules required for the program $(COLOR_RED)(This command might take some time)$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)  train-tunning$(COLOR_RESET)				: Train the model with tuning mode $(COLOR_YELLOW)activated$(COLOR_RESET)"
	@echo "					$(COLOR_CYAN)(Shorter training for testing, parameter tuning, and configuration)$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)  train-tuned$(COLOR_RESET)				: Train the model with tuning mode $(COLOR_MAGENTA)deactivated$(COLOR_RESET)"
	@echo "					$(COLOR_CYAN)(Full training to have the model ready to be used)$(COLOR_RESET)"
	@echo "$(COLOR_GREEN)  train$(COLOR_RESET)					: Alias for '$(COLOR_GREEN)train-tunning$(COLOR_RESET)'"
	@echo "$(COLOR_GREEN)  evaluate$(COLOR_MAGENTA) <inputs/input_file.txt>$(COLOR_RESET)	: Evaluate the currently choose model in: '$(COLOR_GREEN)src/config/configuration.py$(COLOR_RESET)' with the given: '$(COLOR_MAGENTA)input_file.txt$(COLOR_RESET) in the: '$(COLOR_GREEN)inputs folder$(COLOR_RESET)' as input'"

# Rule to clean the terminal
clear:
ifeq ($(detected_OS),Windows)
	@cls
else
	@clear
endif

# Rule to assign unlimited resources to the terminal
adapt:
ifeq ($(detected_OS),Windows)
	@echo "Adapt command not supported on Windows"
else
	@ulimit -t unlimited
endif

install-deps:
	pip install --upgrade pytorch_lightning
	pip install --upgrade pandas
	pip install --upgrade transformers
	pip install --upgrade pathlib
	pip install --upgrade argparse
	pip install --upgrade scikit-learn
	pip install --upgrade scipy
	pip install --upgrade torch
	pip install --upgrade torchmetrics

# ----------------------------- Training rules --------------------------------
# Rule to train the model with tuning mode on
train-tunning:
	@echo "$(COLOR_YELLOW)Running program with tuning mode activated$(COLOR_RESET)"
ifeq ($(detected_OS),Windows)
	set TF_ENABLE_ONEDNN_OPTS=0 && python src/train/main.py 1
else
	export TF_ENABLE_ONEDNN_OPTS=0 && python3 src/train/main.py 1
endif

# Rule to train the model with tuning mode off
train-tuned:
	@echo "$(COLOR_MAGENTA)Running program with tuning mode deactivated$(COLOR_RESET)"
ifeq ($(detected_OS),Windows)
	set TF_ENABLE_ONEDNN_OPTS=0 && python src/train/main.py 0
else
	export TF_ENABLE_ONEDNN_OPTS=0 && python3 src/train/main.py 0
endif

# If no indication on the tuning is given, assume tuning mode on
train: train-tunning

# Rule to evaluate the trained model
evaluate:
	ifeq ($(filter-out $@,$(MAKECMDGOALS)),)
		@echo "$(COLOR_RED)Error: No input file argument provided. Usage: make evaluate <file> $(COLOR_RESET)"
	else
		@echo "$(COLOR_CYAN)Evaluating model with input file: $(filter-out $@,$(MAKECMDGOALS)) $(COLOR_RESET)"
		ifeq ($(detected_OS),Windows)
			set TF_ENABLE_ONEDNN_OPTS=0 && python src/evaluate/label_evaluator.py $(filter-out $@,$(MAKECMDGOALS))
		else
			export TF_ENABLE_ONEDNN_OPTS=0 && python3 src/evaluate/label_evaluator.py $(filter-out $@,$(MAKECMDGOALS))
		endif
	endif

# Rule to ignore undeclared rules
%:
	@: