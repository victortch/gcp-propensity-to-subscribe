resource "google_service_account" "sa_ml_train" {
  account_id   = "sa-ml-train"
  display_name = "PTS training service account"
}

resource "google_service_account" "sa_ml_infer" {
  account_id   = "sa-ml-infer"
  display_name = "PTS inference service account"
}

resource "google_service_account" "sa_workflows" {
  account_id   = "sa-workflows"
  display_name = "PTS workflows service account"
}

resource "google_project_iam_member" "train_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.sa_ml_train.email}"
}

resource "google_project_iam_member" "train_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.sa_ml_train.email}"
}

resource "google_project_iam_member" "train_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.sa_ml_train.email}"
}

resource "google_project_iam_member" "infer_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.sa_ml_infer.email}"
}

resource "google_project_iam_member" "infer_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.sa_ml_infer.email}"
}

resource "google_project_iam_member" "infer_artifact_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.sa_ml_infer.email}"
}

resource "google_project_iam_member" "workflows_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.sa_workflows.email}"
}

resource "google_project_iam_member" "workflows_bq_job_user" {
  project = var.project_id
  role    = "roles/bigquery.jobUser"
  member  = "serviceAccount:${google_service_account.sa_workflows.email}"
}

resource "google_service_account_iam_member" "workflows_use_train" {
  service_account_id = google_service_account.sa_ml_train.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.sa_workflows.email}"
}

resource "google_service_account_iam_member" "workflows_use_infer" {
  service_account_id = google_service_account.sa_ml_infer.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.sa_workflows.email}"
}

resource "google_project_iam_member" "cloudbuild_artifact_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${var.project_id}@cloudbuild.gserviceaccount.com"
}
