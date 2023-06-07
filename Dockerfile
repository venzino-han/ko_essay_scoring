FROM huggingface/transformers-pytorch-gpu:4.29.2
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /workspace/kobert
