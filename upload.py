# from feedback.fb3-deberta-v3-base-baseline-train import OUTPUT_DIR
from kaggle.api.kaggle_api_extended import KaggleApi
import os
import json

# class Config:
#     Exp = "exp01"
#     OUTPUT_DIR = "exp01"

exp_no = "exp012-fb3-deberta-v3-base-vocabulary"

user_name="shigengtian"
def dataset_create_new(dataset_name, upload_dir):
    dataset_metadata = {}
    dataset_metadata['id'] = f'{user_name}/{dataset_name}'
    dataset_metadata['licenses'] = [{'name': 'CC0-1.0'}]
    dataset_metadata['title'] = dataset_name
    with open(os.path.join(upload_dir, 'dataset-metadata.json'), 'w') as f:
        json.dump(dataset_metadata, f, indent=4)
    api = KaggleApi()
    api.authenticate()
    api.dataset_create_new(folder=upload_dir, convert_to_csv=False, dir_mode='tar')

dataset_create_new(dataset_name=exp_no, upload_dir=exp_no)