git clone https://github.com/jim-dilkes/bert-encoder
cd bert-encoder
python -m pip install --upgrade pip
pip install -r requirements.txt
apt update
apt install unzip

export HF_DATASETS_CACHE="/workspace/huggingface/datasets"