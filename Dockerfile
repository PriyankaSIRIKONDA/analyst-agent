FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

# api.py computes ROOT = Path(__file__).parents[2], which must be /app, so the package
# is imported from /app/src and not installed into site-packages. The MCP server subprocess
# inherits PYTHONPATH, so it finds the package too.
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Never run as root in the container.
RUN useradd --create-home --uid 10001 app
USER app

EXPOSE 8000

CMD ["uvicorn", "analyst_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
