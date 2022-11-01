# python train_baseline.py --model microsoft/deberta-v3-large --lr 1e-5 --batch_size 8 --output_path exp016/
python train_baseline.py --tokenizer microsoft/deberta-v3-base --model deberta-base --lr 1e-5 --batch_size 8 --output_path exp020/
python train_baseline.py --tokenizer microsoft/deberta-v3-base --model deberta-base --lr 2e-5 --batch_size 8 --output_path exp021/