python ./src/data_preprocessing/data_conventor_conll.py \
  --original_train_dataset_path ./src/data/conll/conll16st-en-03-29-16-train/relations.json \
  --original_dev_dataset_path ./src/data/conll/conll16st-en-03-29-16-dev/relations.json \
  --original_test_dataset_path ./src/data/conll/conll16st-en-03-29-16-test/relations.json \
  --original_blind_dataset_path ./src/data/conll/conll16st-en-03-29-16-blind-test/relations.json \
  --dataset_dir_path ./src/data/conll_14 \
  --num_rels 14