FROM python:3.8.5
WORKDIR /usr/src/vg-transcribe-comprehend
COPY . .
RUN pip install --upgrade -r requirements.txt
CMD ["python", "server.py"]
EXPOSE 5000