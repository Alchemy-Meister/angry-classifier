#!/bin/sh

workon anger-detection

source ./1.filter-raw-irony-dataset.sh
source ./2.create-irony-dataset.sh
source ./3.irony-word2vec.sh
source ./4.irony-cnn.sh