python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/two_phase_training/second_phase/baseline/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_test.json \
    ./merged/json/backup_deusto/two_phase_training/second_phase/embeddings/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_test.json \
    ./merged/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_test.csv ./merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    ./merged/json/backup_deusto/two_phase_training/embedding_word_index.json

python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/two_phase_training/second_phase/baseline/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_validation.json \
    ./merged/json/backup_deusto/two_phase_training/second_phase/embeddings/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_validation.json \
    ./merged/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_validation.csv ./merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    ./merged/json/backup_deusto/two_phase_training/embedding_word_index.json

python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/two_phase_training/second_phase/baseline/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_train.json \
    ./merged/json/backup_deusto/two_phase_training/second_phase/embeddings/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_train.json \
    ./merged/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_train.csv ./merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    ./merged/json/backup_deusto/two_phase_training/embedding_word_index.json

python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_anger_dataset_test.json \
    ./merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_anger_dataset_test.json \
    ./merged/first_phase_binary_anger_dataset_test.csv ./merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    ./merged/json/backup_deusto/two_phase_training/embedding_word_index.json

python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_anger_dataset_validation.json \
    ./merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_anger_dataset_validation.json \
    ./merged/first_phase_binary_anger_dataset_validation.csv ./merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    ./merged/json/backup_deusto/two_phase_training/embedding_word_index.json

python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_anger_dataset_train.json \
    ./merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_anger_dataset_train.json \
    ./merged/first_phase_binary_anger_dataset_train.csv ./merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    ./merged/json/backup_deusto/two_phase_training/embedding_word_index.json

python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_irony_dataset_test.json \
    ./merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_irony_dataset_test.json \
    ./merged/first_phase_binary_irony_dataset_test.csv ./merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    ./merged/json/backup_deusto/two_phase_training/embedding_word_index.json

python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_irony_dataset_validation.json \
    ./merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_irony_dataset_validation.json \
    ./merged/first_phase_binary_irony_dataset_validation.csv ./merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    ./merged/json/backup_deusto/two_phase_training/embedding_word_index.json

python ./check_if_im_retarded.py \
    ./merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_irony_dataset_train.json \
    ./merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_irony_dataset_train.json \
    ./merged/first_phase_binary_irony_dataset_train.csv ./merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    ./merged/json/backup_deusto/two_phase_training/embedding_word_index.json

echo "If you don't see any error you are not retarted"