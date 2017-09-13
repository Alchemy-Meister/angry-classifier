python ../../deep-learning/cnn_deusto/cnn_with_embeddings.py \
    --embedding_weight=../../datasets/merged/json/backup_deusto/two_phase_training/embedding_weights.npy \
    --train=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_irony_dataset_train.json \
    --validation=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_irony_dataset_validation.json \
    --test=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_irony_dataset_test.json \
    --name=trainable_embeddings_512_filters_double_dense_with_0.4_drop_conv_actv_batch