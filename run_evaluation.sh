export CUDA_VISIBLE_DEVICES=1

source /data/tota_abe/.venv/bin/activate
cd /data/tota_abe/master

python cli/evaluate.py --dims 2,3,4 --no-reasoning --models Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-Math-7B-Instruct --questions-csv data/questions_generated_2000_gpt54.csv
