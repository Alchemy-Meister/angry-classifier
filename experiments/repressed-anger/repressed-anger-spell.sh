#!/bin/sh

python ../../deep-learning/repressed-anger.py \
	--word2vec_file=../../datasets/merged/json/backup_deusto/two_phase_training/second_phase/baseline/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_test.json \
    --dataset=../../datasets/merged/repressed_anger_prediction_dataset_hit_reprocessed_trimmed_test.csv \
    --anger_dir=../../deep-learning/cnn_japan/first_phase_binary_anger_dataset/ \
    --anger_weights_filename=01-0.34-300.hdf5 \
    --anger_distribution=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_anger_dataset_distribution.json \
    --irony_dir=../../deep-learning/cnn_japan/first_phase_binary_irony_dataset/ \
    --irony_weights_filename=01-0.31-300.hdf5 \
    --irony_distribution=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_irony_dataset_distribution.json
