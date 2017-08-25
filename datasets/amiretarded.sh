python ./check_if_im_retarded.py \
    ./split/json/backup_deusto/backup\ spell/baseline/binary_anger_dataset_test.json \
    ./split/json/backup_deusto/backup\ spell/embeddings/binary_anger_dataset_test.json \
    ./split/binary_anger_dataset_test.csv

python ./check_if_im_retarded.py \
    ./split/json/backup_deusto/backup\ spell/baseline/binary_anger_dataset_train.json \
    ./split/json/backup_deusto/backup\ spell/embeddings/binary_anger_dataset_train.json \
    ./split/binary_anger_dataset_train.csv

python ./check_if_im_retarded.py \
    ./split/json/backup_deusto/backup\ spell/baseline/binary_anger_dataset_validation.json \
    ./split/json/backup_deusto/backup\ spell/embeddings/binary_anger_dataset_validation.json \
    ./split/binary_anger_dataset_validation.csv


python ./check_if_im_retarded.py \
    ./split/json/backup_deusto/backup\ spell/baseline/binary_irony_dataset_test.json \
    ./split/json/backup_deusto/backup\ spell/embeddings/binary_irony_dataset_test.json \
    ./split/binary_irony_dataset_test.csv

python ./check_if_im_retarded.py \
    ./split/json/backup_deusto/backup\ spell/baseline/binary_irony_dataset_train.json \
    ./split/json/backup_deusto/backup\ spell/embeddings/binary_irony_dataset_train.json \
    ./split/binary_irony_dataset_train.csv

python ./check_if_im_retarded.py \
    ./split/json/backup_deusto/backup\ spell/baseline/binary_irony_dataset_validation.json \
    ./split/json/backup_deusto/backup\ spell/embeddings/binary_irony_dataset_validation.json \
    ./split/binary_irony_dataset_validation.csv

python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/backup\ spell/baseline/repressed_anger_prediction_dataset_hit_reprocessed.json \
    ./merged/json/backup_deusto/backup\ spell/embeddings/repressed_anger_prediction_dataset_hit_reprocessed.json \
    ./merged/repressed_anger_prediction_dataset_hit_reprocessed.csv

echo "If you don't see any error you are not retarted"