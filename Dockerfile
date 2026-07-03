FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TZ=Asia/Shanghai
ENV AKSHARE_TIMEOUT_SECONDS=4
ENV FAST_FALLBACK_FIRST=1

COPY stock_ai_system/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY stock_ai_system/ /app/

RUN mkdir -p /app/local_data

EXPOSE 8501

CMD streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
