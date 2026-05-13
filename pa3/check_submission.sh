#!/bin/bash

echo "Checking submission files for PA3..."
echo "-----------------------------------"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

errors=0

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}OK${NC} $1"
    else
        echo -e "${RED}MISSING${NC} $1"
        errors=$((errors+1))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}OK${NC} $1"
    else
        echo -e "${RED}MISSING${NC} $1"
        errors=$((errors+1))
    fi
}

echo "Checking directories..."
check_dir "part1"
check_dir "part2"
check_dir "part3"

echo -e "\nChecking Part 1 files..."
check_file "part1/moe.py"
check_file "part1/test_moe.py"
check_file "part1/benchmark.py"
check_file "part1/analysis.md"
check_file "part1/rng.py"
check_file "part1/mpi_wrapper/__init__.py"
check_file "part1/mpi_wrapper/comm.py"

echo -e "\nChecking Part 2 files..."
check_file "part2/model_training_cost_analysis.py"
check_file "part2/llama3_8b_config.json"
check_file "part2/my_model_config.json"
check_file "part2/deepseek_v3_config.json"
check_file "part2/moe.md"

echo -e "\nChecking Part 3 files..."
check_file "part3/PA3_Speculative_Decoding.ipynb"

echo -e "\n-----------------------------------"
if [ $errors -eq 0 ]; then
    echo -e "${GREEN}All required files are present.${NC}"
    exit 0
else
    echo -e "${RED}$errors file(s) are missing.${NC}"
    echo "Please make sure all required files are included before submission."
    exit 1
fi
