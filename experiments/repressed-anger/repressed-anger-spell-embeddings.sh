python ../../deep-learning/repressed-anger-embeddings.py \
    --dataset=../../datasets/merged/repressed_anger_prediction_dataset_hit_reprocessed.csv \
    --anger_dir=../../deep-learning/binary_anger_dataset/ \
    --anger_weights_filename=02-0.48-300.hdf5 \
    --anger_distribution=../../datasets/split/json/binary_anger_dataset_distribution.json \
    --irony_dir=../../deep-learning/binary_irony_dataset/ \
    --irony_weights_filename=02-0.39-300.hdf5 \
    --irony_distribution=../../datasets/split/json/binary_irony_dataset_distribution.json
