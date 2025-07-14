FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# https://docs.docker.com/reference/dockerfile/#shell-and-exec-form
# https://manpages.ubuntu.com/manpages/noble/en/man1/sh.1.html
SHELL ["/bin/sh", "-exc"]

ARG DEBIAN_FRONTEND=noninteractive
ARG python_version=3.9.21

COPY --link --from=ghcr.io/astral-sh/uv:0.7.14 /uv /usr/local/bin/uv

RUN apt-get update --quiet && \
    apt-get upgrade -y && \
    apt-get install --quiet --no-install-recommends -y build-essential git ca-certificates \
    libgl1 libglib2.0-0 libusb-1.0-0-dev && \
    # Forcing http 1.1 to fix https://stackoverflow.com/q/59282476
    git config --global http.version HTTP/1.1 && \
    uv python install $python_version

ENV UV_PYTHON="python$python_version" \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app \
    UV_LINK_MODE=copy \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PYTHONOPTIMIZE=1

WORKDIR /project
COPY pyproject.toml uv.lock README.md /project

# Building deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-dev --no-install-project --frozen

COPY ./fairseq_lib /project/fairseq_lib
# Building fairseq_lib
# cd to /project/fairseq_lib and run uv pip install -e .
RUN cd fairseq_lib && \
    uv pip install -e ./ --python /app/bin/python


# Copying the rest of the project files
COPY ./src /project/src
COPY ./scripts /project/scripts
COPY ./tests /project/tests
COPY ./config /project/config
