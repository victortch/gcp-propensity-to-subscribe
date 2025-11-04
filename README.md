# Economedia Propensity-to-Subscribe (GCP)

This repository contains the production implementation of Economedia's Propensity-to-Subscribe (PTS) engine on Google Cloud Platform. The solution mirrors the legacy on-premise training behavior while introducing the cloud services needed for quarterly retraining and daily batch inference.

## Architecture overview

| Component | Responsibility |
| --- | --- |
| BigQuery | System of record for features, training data, and model outputs under the `propensity_to_subscribe` dataset. |
| Cloud Storage | Stores model artifacts, plots, and run metadata at `gs://economedia-pts-models`. |
| Vertex AI Custom Jobs | Containerized training (quarterly) and inference (daily) workloads. |
| Cloud Workflows & Scheduler | Orchestrate training and inference cadences with explicit parameters. |
| Artifact Registry | Hosts the Docker images used by Vertex AI. |
| Cloud Build | Builds and publishes the training and inference images when the `main` branch updates. |

## Repository layout

```
app/
  common/                 # Shared IO, BigQuery helpers, preprocessing, registry utilities
  training/               # Training data build, model fitting, artifact persistence, metrics loaders
  inference/              # Batch scoring entrypoint, schema helpers
bq/
  ddl/                    # BigQuery table/view DDL scripts
  scheduled_queries/      # SQL for scheduled feature builds
cloudbuild/               # Cloud Build configurations for images
configs/                  # Environment and label configuration files
containers/               # Dockerfiles and requirements for training/inference images
scripts/                  # Helper scripts for bootstrapping the project
workflows/                # Cloud Workflow definitions for training/inference orchestration
```

Refer to `original_files/` for the authoritative legacy scripts used to shape the refactor.

## Getting started locally

1. Copy the environment template and adjust project-specific overrides:
   ```bash
   cp configs/env.example.yaml configs/env.yaml
   # edit configs/env.yaml with project_id, dataset, and bucket overrides
   ```
2. Install dependencies (Python 3.10+ recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r containers/requirements-training.txt
   pip install -r containers/requirements-inference.txt
   ```
3. Run static validation to ensure modules import cleanly:
   ```bash
   python -m compileall app
   ```

## Training workflow

1. `app/training/cv_build.py` executes the BigQuery SQL in `app/training/sql/build_training_dataset.sql` with explicit date parameters to populate `train_data` and `cv_build_metadata`.
2. `app/training/train.py` loads the prepared tables, performs Bayesian optimization over XGBoost hyperparameters, fits one model per label (`cap_90d`, `dne_90d`, `cap_30d`, `dne_30d`), calibrates probabilities, and computes full validation/test metrics.
3. `app/training/artifact_io.py` saves artifacts to Cloud Storage and registers the run in Vertex AI Model Registry.
4. `app/training/metrics_to_bq.py` loads summary and per-split metrics into BigQuery tables.

The training entrypoint (`app/training/entrypoint.py`) stitches these steps for execution inside Vertex AI Custom Jobs. Artifacts are written to `gs://economedia-pts-models/<run_id>/...` and metrics to `propensity_to_subscribe.train_metrics` tables.

## Inference workflow

`app/inference/batch_predict.py` resolves the latest production model version from Vertex AI, loads artifacts from Cloud Storage, scores the `propensity_to_subscribe.features_daily` table for the provided `scoring_date`, and writes probabilities and binary decisions to `propensity_to_subscribe.predictions_daily`.

The inference entrypoint (`app/inference/entrypoint.py`) is wired for Vertex AI Custom Jobs, parameterized by `scoring_date` and optional overrides for project, dataset, or bucket names.

## Deployment pipeline

* **Cloud Build**: `cloudbuild/*.yaml` build and publish container images to Artifact Registry (`europe-west3-docker.pkg.dev/economedia-data-prod-laoy/pts_model`).
* **Cloud Workflows**: `workflows/*.yaml` define orchestration for quarterly training and daily inference, invoked by Cloud Scheduler with explicit parameters.
* **Helper scripts**: The `scripts/` directory contains idempotent shell scripts to bootstrap required infrastructure, deploy workflows, and configure schedules.

## Testing

The repository currently relies on import compilation via `python -m compileall app`. Additional unit tests can be introduced under a `tests/` directory as the project matures.

## License

Proprietary to Economedia. All rights reserved.
