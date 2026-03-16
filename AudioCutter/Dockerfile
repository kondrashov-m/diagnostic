FROM jrottenberg/ffmpeg:latest

RUN apt-get update && apt-get install -y \
    bash \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

    WORKDIR /app
RUN mkdir -p /app/input_files /app/out_files

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]