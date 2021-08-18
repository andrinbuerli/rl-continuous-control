
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

WORKDIR /app

RUN pip install unityagents==0.4.0 --no-deps
COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . /app