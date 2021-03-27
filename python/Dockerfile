FROM conda/miniconda3:latest
WORKDIR /app
COPY environment.yml ./
RUN conda env create -f environment.yml
RUN conda init bash
