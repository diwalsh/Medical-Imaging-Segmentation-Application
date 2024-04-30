SHELL := /bin/bash

.PHONY: setup
setup:
	pyenv local 3.11.6
	python -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements_dev.txt
