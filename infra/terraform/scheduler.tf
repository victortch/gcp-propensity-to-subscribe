resource "google_cloud_scheduler_job" "training" {
  name        = "training-quarterly"
  project     = var.project_id
  region      = var.region
  description = "Trigger quarterly training workflow"
  schedule    = "0 6 1 */3 *"
  time_zone   = "Etc/UTC"

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/projects/${var.project_id}/locations/${var.region}/workflows/training-workflow/executions"
    oidc_token {
      service_account_email = "sa-workflows@${var.project_id}.iam.gserviceaccount.com"
    }
    headers = {
      "Content-Type" = "application/json"
    }
    body = base64encode("{\"argument\":\"{\\\"project_id\\\":\\\"${var.project_id}\\\",\\\"region\\\":\\\"${var.region}\\\"}\"}")
  }
}

resource "google_cloud_scheduler_job" "inference" {
  name        = "inference-daily"
  project     = var.project_id
  region      = var.region
  description = "Trigger daily inference workflow"
  schedule    = "0 3 * * *"
  time_zone   = "Etc/UTC"

  http_target {
    http_method = "POST"
    uri         = "https://workflowexecutions.googleapis.com/v1/projects/${var.project_id}/locations/${var.region}/workflows/inference-workflow/executions"
    oidc_token {
      service_account_email = "sa-workflows@${var.project_id}.iam.gserviceaccount.com"
    }
    headers = {
      "Content-Type" = "application/json"
    }
    body = base64encode("{\"argument\":\"{\\\"project_id\\\":\\\"${var.project_id}\\\",\\\"region\\\":\\\"${var.region}\\\",\\\"scoring_date\\\":\\\"${formatdate(\"%Y-%m-%d\", timestamp())}\\\"}\"}")
  }
}
