FROM intel/pytorch:latest

EXPOSE 7860 8001

RUN apt update && apt install -y ffmpeg patch

RUN git clone --branch v0.1.8 --depth 1 https://github.com/ace-step/ACE-Step-1.5

WORKDIR /ACE-Step-1.5

RUN pip install -r requirements-xpu.txt

COPY *.patch .
RUN for P in *.patch; do patch -Np1 < $P; done

ENV SYCL_CACHE_PERSISTENT=1
ENV SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
ENV PYTORCH_DEVICE=xpu
ENV TORCH_COMPILE_BACKEND=eager
ENV TORCHAUDIO_USE_BACKEND="soundfile"

CMD ["python", "-m", "acestep.acestep_v15_pipeline"]
