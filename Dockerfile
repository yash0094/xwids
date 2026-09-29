# XWIDS dashboard image. Works as a Hugging Face Docker Space (port 7860), on Render, or anywhere with Docker.
#   python scripts/05_export_deploy.py --dataset synthetic     # (or awid3) creates deploy/ first
#   docker build -t xwids . && docker run -p 7860:7860 -e XWIDS_USERS="analyst:<pw>" xwids
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=7860 HOST=0.0.0.0 XWIDS_DATASET=synthetic \
    XWIDS_CONFIG=/app/config.yaml

# Hugging Face Spaces run containers as uid 1000; the app writes its SQLite DB and retrained models
RUN useradd -m -u 1000 xwids
WORKDIR /app
COPY requirements-deploy.txt deploy/versions.txt ./
# versions.txt = exact library versions the exported model was trained with (written by 05_export_deploy.py)
RUN pip install -r requirements-deploy.txt -r versions.txt

COPY --chown=xwids:xwids xwids/ xwids/
COPY --chown=xwids:xwids app/ app/
COPY --chown=xwids:xwids serve.py ./
COPY --chown=xwids:xwids deploy/config.yaml ./config.yaml
COPY --chown=xwids:xwids deploy/data/ data/
COPY --chown=xwids:xwids deploy/models/ models/
RUN mkdir -p runtime reports && chown -R xwids:xwids /app

USER xwids
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT','7860'))"
CMD ["python", "serve.py"]
