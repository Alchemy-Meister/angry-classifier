python ../../deep-learning/repressed-anger-embeddings.py \
    --dataset=../../datasets/merged/repressed_anger_prediction_dataset_hit_reprocessed.csv \
    --anger_dir=../../deep-learning/binary_anger_dataset/ \
    --anger_weights_filename=01-0.36-300.hdf5 \
    --anger_distribution=../../datasets/split/json/binary_anger_dataset_distribution.json \
    --irony_dir=../../deep-learning/binary_irony_dataset/ \
    --irony_weights_filename=03-0.29-300.hdf5 \
    --irony_distribution=../../datasets/split/json/binary_irony_dataset_distribution.json
