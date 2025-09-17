# Ensure UTF-8 output and input
$env:PYTHONIOENCODING = 'utf-8'
chcp 65001 | Out-Null

# Create venv if not exists
if (-not (Test-Path ".venv/Scripts/python.exe")) {
    py -m venv .venv
}

# Upgrade pip
& .\.venv\Scripts\python -m pip install --upgrade pip

# Install dependencies
if (Test-Path "requirements.txt") {
    & .\.venv\Scripts\python -m pip install -r requirements.txt
} else {
    & .\.venv\Scripts\python -m pip install numpy matplotlib scikit-learn
}

# Run the script
& .\.venv\Scripts\python .\lab1.py
