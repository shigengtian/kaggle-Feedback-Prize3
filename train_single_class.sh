# python train_baseline.py --model microsoft/deberta-v3-large --lr 1e-5 --batch_size 8 --output_path exp016/
# python train_baseline.py --tokenizer microsoft/deberta-v3-base --model deberta-base --lr 1e-5 --batch_size 8 --output_path exp020/
# python train_baseline.py --tokenizer microsoft/deberta-v3-base --model deberta-base --lr 2e-5 --batch_size 8 --output_path exp021/
# ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
python train_single_class.py --target_cols cohesion --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 2e-5 --batch_size 8 --output_path exp023_cohesion/
python train_single_class.py --target_cols syntax --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 2e-5 --batch_size 8 --output_path exp023_syntax/
python train_single_class.py --target_cols vocabulary --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 2e-5 --batch_size 8 --output_path exp023_vocabulary/
python train_single_class.py --target_cols phraseology --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 2e-5 --batch_size 8 --output_path exp023_phraseology/
python train_single_class.py --target_cols grammar --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 2e-5 --batch_size 8 --output_path exp023_grammar/
python train_single_class.py --target_cols conventions --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 2e-5 --batch_size 8 --output_path exp023_conventions/

python train_single_class.py --target_cols cohesion --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 1e-5 --batch_size 8 --output_path exp024_cohesion/
python train_single_class.py --target_cols syntax --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 1e-5 --batch_size 8 --output_path exp024_syntax/
python train_single_class.py --target_cols vocabulary --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 1e-5 --batch_size 8 --output_path exp024_vocabulary/
python train_single_class.py --target_cols phraseology --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 1e-5 --batch_size 8 --output_path exp024_phraseology/
python train_single_class.py --target_cols grammar --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 1e-5 --batch_size 8 --output_path exp024_grammar/
python train_single_class.py --target_cols conventions --tokenizer microsoft/deberta-v3-base --model microsoft/deberta-v3-base --lr 1e-5 --batch_size 8 --output_path exp024_conventions/

python train_single_class.py --target_cols cohesion --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 2e-5 --batch_size 8 --output_path exp025_cohesion/
python train_single_class.py --target_cols syntax --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 2e-5 --batch_size 8 --output_path exp025_syntax/
python train_single_class.py --target_cols vocabulary --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 2e-5 --batch_size 8 --output_path exp025_vocabulary/
python train_single_class.py --target_cols phraseology --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 2e-5 --batch_size 8 --output_path exp025_phraseology/
python train_single_class.py --target_cols grammar --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 2e-5 --batch_size 8 --output_path exp025_grammar/
python train_single_class.py --target_cols conventions --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 2e-5 --batch_size 8 --output_path exp025_conventions/

python train_single_class.py --target_cols cohesion --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 1e-5 --batch_size 8 --output_path exp026_cohesion/
python train_single_class.py --target_cols syntax --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 1e-5 --batch_size 8 --output_path exp026_syntax/
python train_single_class.py --target_cols vocabulary --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 1e-5 --batch_size 8 --output_path exp026_vocabulary/
python train_single_class.py --target_cols phraseology --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 1e-5 --batch_size 8 --output_path exp026_phraseology/
python train_single_class.py --target_cols grammar --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 1e-5 --batch_size 8 --output_path exp026_grammar/
python train_single_class.py --target_cols conventions --tokenizer microsoft/deberta-v3-base --model deberta-base/checkpoint-9750 --lr 1e-5 --batch_size 8 --output_path exp026_conventions/


