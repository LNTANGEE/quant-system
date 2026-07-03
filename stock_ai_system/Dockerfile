FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV AKSHARE_TIMEOUT_SECONDS=4
ENV FAST_FALLBACK_FIRST=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
