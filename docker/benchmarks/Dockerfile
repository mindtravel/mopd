# IFBench + AutomationBench clients (CPU only): both talk to an OpenAI-compatible model endpoint,
# which mopd's bench service starts on the host (scripts/bench_service.py; this server's Docker has
# no GPU runtime). Pinned sources are copied from the native deployments under /ssd by build.sh.
FROM python:3.13-slim-bookworm
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
COPY uv /usr/local/bin/uv

# IFBench (allenai/IFBench @ 1c40f0c), its wrapper (generate/score with completeness checks), NLTK data
COPY ifbench/repo /opt/ifbench/repo
COPY ifbench/scripts /opt/ifbench/scripts
COPY ifbench/nltk_data /opt/ifbench/cache/nltk_data
RUN cd /opt/ifbench/repo && UV_PROJECT_ENVIRONMENT=/opt/ifbench/venv \
    uv sync --frozen --no-default-groups --python /usr/local/bin/python3 && rm -rf /root/.cache
ENV NLTK_DATA=/opt/ifbench/cache/nltk_data

# AutomationBench (zapier/AutomationBench @ 4a8e106), simulated SaaS world state, no accounts needed
COPY automationbench/repo /opt/automationbench/repo
RUN cd /opt/automationbench/repo && UV_PROJECT_ENVIRONMENT=/opt/automationbench/venv \
    uv sync --frozen --no-dev --python /usr/local/bin/python3 && rm -rf /root/.cache
ENV HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

COPY entrypoint.sh /usr/local/bin/bench
ENTRYPOINT ["bench"]
