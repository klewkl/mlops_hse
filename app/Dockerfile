FROM python:3.9-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV NAME World

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y git

EXPOSE 8000
EXPOSE 8501
EXPOSE 80


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# CMD ["streamlit", "run", "mlops_pipeline/streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
