#!/bin/bash
python import_data.py && \
python train_compression.py && \
python predict_compression.py && \
python train_classification.py && \
python predict_classification.py