python train_baseline.py --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 2e-5 --batch_size 8 --output_path exp029/
python train_baseline.py --tokenizer microsoft/deberta-v3-large --model microsoft/deberta-v3-large --lr 2e-5 --batch_size 8 --output_path exp030/
python train_baseline.py --tokenizer microsoft/deberta-base --model microsoft/deberta-base --lr 2e-5 --batch_size 8 --output_path exp031/
python train_baseline.py --tokenizer microsoft/deberta-large --model microsoft/deberta-large --lr 1e-5 --batch_size 8 --output_path exp032/
python train_baseline.py --tokenizer microsoft/deberta-large --model microsoft/deberta-large --lr 2e-5 --batch_size 8 --output_path exp033/