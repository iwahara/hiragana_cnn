#!/usr/bin/env bash

python3 prepare_dataset.py

python3 hiragana_cnn.py

#tensorflowjs_converter --input_format keras models/hiragana_cnn_model.h5 models
