FROM public.ecr.aws/lambda/python:3.13

# Copy the uv binary from your repository (not downloaded during build!)
COPY uv /bin/uv
RUN chmod +x /bin/uv

COPY pyproject.toml uv.lock ./
RUN uv pip install --system -r <(uv export --format requirements-txt)

COPY lambda_function.py model.bin ./

CMD ["lambda_function.lambda_handler"]
