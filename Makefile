# Makefile
.PHONY: all install test clean

# Default target
all: test

# Install the required Python packages
install:
	python -m pip install --upgrade pip
	python -m pip install numpy matplotlib Pylops Sphinx pytest sphinx-rtd-theme

# Run pytest (depends on install)
test: install
	python -m pytest

# Optional: clean up any __pycache__ or pytest cache
clean:
	rm -rf __pycache__ */__pycache__ .pytest_cache
