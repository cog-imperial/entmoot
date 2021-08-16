FROM registry.roqs.basf.net/base-images/fastapi:latest

# 1) Build and install the ipopt python package
USER root
RUN apt-get update \
  &&  DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
  coinor-libipopt1v5 \
  coinor-libipopt-dev \
  libblas-dev \
  liblapack-dev \
  g++ \
  gfortran \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*
RUN pip3 install ipopt
RUN apt-get remove -y coinor-libipopt-dev libblas-dev g++ gfortran pkg-config

# 2) All other requirements
ADD requirements.txt /app/
RUN pip3 install -U -r requirements.txt && rm requirements.txt

EXPOSE 5000
USER app
ADD . /app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--access-logfile", "-", "--error-logfile", "-", "--capture-output", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "360", "api.main:app"]
