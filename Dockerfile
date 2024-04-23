FROM mosaicml/pytorch:latest

RUN pip install transformers tokenizers datasets torchinfo schedulefree neptune mosaicml[all]

ENV TOKENIZERS_PARALLELISM="false"

ENV NEPTUNE_API_TOKEN=${NEPTUNE_API_TOKEN}

COPY model.py model.py
COPY train.py train.py
COPY eval.py eval.py
COPY data.zip data.zip

RUN sudo apt update && sudo apt install -y unzip
RUN unzip data.zip

CMD ["python", "train.py", "--run_name", "cluster-test-run-1", "--eval_interval", "6000ba", "--learning_rate", "1e-4", "--batch_size", "32", "--context_window", "4096", "--datafolder", "/users/ddhamani/8156/data-final"]