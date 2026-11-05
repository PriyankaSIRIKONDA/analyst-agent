FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

# api.py computes ROOT = Path(__file__).parents[2], which must be /app, so the package
# is imported from /app/src and not installed into site-packages.
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "analyst_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
