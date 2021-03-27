# Training model using Python

## Overview

- Based on [LSTM-Classification](https://github.com/iamhosseindhv/LSTM-Classification).

## Prerequisites

The following prerequisites are required to be installed on your system:

- docker
- docker-compose

## Execution

### Deploy

```sh
./deploy.sh
docker exec -it toxic-comment-classification /bin/bash
```

### Change environment and install missing packages

```bash
conda activate myenv
pip install tensorflowjs
python -m nltk.downloader stopwords wordnet
```

### Train model

```bash
python main.py
```

### Convert Keras model to TF.js Layers format

```bash
tensorflowjs_converter --input_format keras \
                       /app/src/data/model.h5 \
                       /app/src/data/tfjs
```

## Contribution

Your contribution is welcome and greatly appreciated. Please contribute your fixes and new features via a pull request.
Pull requests and proposed changes will then go through a code review and once approved will be merged into the project.

If you like my work, please leave me a star :)
