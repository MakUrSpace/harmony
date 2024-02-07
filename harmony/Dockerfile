FROM python:3.11

WORKDIR /opt

COPY . /opt
RUN apt update
RUN apt install -y libgl1-mesa-glx
RUN apt install -y zbar-tools
RUN pip install -r requirements.txt

CMD ["uvicorn", "harmonyServer:app", "--host", "0.0.0.0", "--workers", "2"]
