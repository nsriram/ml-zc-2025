FROM python:3.13.5-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" ".python-version" ./
RUN uv sync --locked

COPY midterm-project/predict.py \
     midterm-project/custom_transformer.py \
     midterm-project/water_potability_prediction_model_v1_0.joblib ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
