python ../../deep-learning/second_phase_repressed-anger-embeddings.py \
	--name=trainable_embeddings_512_filters_double_dense_with_0.4_drop_conv_actv_batch \
	--word2vec_file=../../datasets/merged/json/backup_deusto/two_phase_training/second_phase/embeddings/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_test.json \
    --dataset=../../datasets/merged/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_test.csv \
    --anger_dir=../../deep-learning/first_phase_binary_anger_dataset/ \
    --anger_weights_filename=01-0.35-300.hdf5 \
    --anger_distribution=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_anger_dataset_distribution.json \
    --irony_dir=../../deep-learning/first_phase_binary_irony_dataset/ \
    --irony_weights_filename=00-0.29-300.hdf5 \
    --irony_distribution=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/embeddings/first_phase_binary_irony_dataset_distribution.json \
