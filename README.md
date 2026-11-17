## Installation (Linux/macOs)

```bash
# 1. Clone the repository
git clone https://github.com/CristhianCruzDelgado/26-27-IntelligentSystems-GmailSpamClassifier.git
cd 26-27-IntelligentSystems-GmailSpamClassifier

# 2. Install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# 3. Check the version (the project requires 3.9 or higher)
python3 --version

# 4. (Recommended) Create and activate a virtual environment
sudo apt install python3-venv
python3 -m venv venv
source venv/bin/activate

# 5. Install the dependencies for preprocess enron
pip install pandas torch transformers sentencepiece sacremoses tqdm

# 6. ...
```
