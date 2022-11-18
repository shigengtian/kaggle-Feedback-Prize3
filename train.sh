# python train_baseline.py --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 2e-5 --batch_size 8 --output_path exp029/
# python train_baseline.py --tokenizer microsoft/deberta-v3-large --model microsoft/deberta-v3-large --lr 2e-5 --batch_size 8 --output_path exp030/
# python train_baseline.py --tokenizer microsoft/deberta-base --model microsoft/deberta-base --lr 2e-5 --batch_size 8 --output_path exp031/
# python train_baseline.py --tokenizer microsoft/deberta-large --model microsoft/deberta-large --lr 1e-5 --batch_size 8 --output_path exp032/
# python train_baseline.py --tokenizer microsoft/deberta-large --model microsoft/deberta-large --lr 2e-5 --batch_size 8 --output_path exp033/
# python train_baseline.py --tokenizer microsoft/deberta-v3-base --model deberta-base-2022/checkpoint-1955 --lr 2e-5 --batch_size 8 --output_path exp034/
# python train_baseline.py --tokenizer microsoft/deberta-v3-base --model deberta-base-2022/checkpoint-1955 --lr 1e-5 --batch_size 8 --output_path exp035/
# python train_baseline.py --tokenizer microsoft/deberta-v3-base --model deberta-base-2022/checkpoint-1955 --lr 6e-6 --batch_size 8 --output_path exp036/
# python train_baseline.py --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 1e-5 --batch_size 8 --output_path exp037/
# python train_baseline.py --tokenizer microsoft/deberta-v3-large --model deberta-large-2022/checkpoint-1955 --lr 2e-5 --batch_size 8 --output_path exp036/
# python train_baseline.py --tokenizer microsoft/deberta-v3-large --model deberta-large/checkpoint-9750 --lr 1e-5 --batch_size 8 --output_path exp038/

# python train_baseline_awp.py --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 2e-5 --batch_size 8 --output_path exp039/
# python train_baseline_awp.py --tokenizer microsoft/deberta-v3-large --model microsoft/deberta-v3-large --lr 1e-5 --batch_size 8 --output_path exp040/
# python exp040-fb3-deberta-v3-base.py
# python exp041-fb3-deberta-v3-base_awp.py
# python exp042-fb3-deberta-v3-base_unscale.py
# python exp043-fb3-deberta-v3-base_fgm.py


# python exp044-fb3-deberta-v3-base.py
# python exp045-fb3-deberta-v3-large_awp.py
# python exp046-fb3-deberta-v3-large_unscale.py
# python exp047-fb3-deberta-v3-large_fgm.py

python exp048-fb3-deberta-v3-base.py
python exp049-fb3-deberta-v3-base_awp.py
python exp050-fb3-deberta-v3-base_unscale.py
python exp051-fb3-deberta-v3-base_fgm.py
