resource "google_workflows_workflow" "training" {
  name        = "training-workflow"
  project     = var.project_id
  region      = var.region
  description = "Quarterly training workflow for Economedia PTS"

  source_contents = file("${path.module}/../../workflows/training_workflow.yaml")
}

resource "google_workflows_workflow" "inference" {
  name        = "inference-workflow"
  project     = var.project_id
  region      = var.region
  description = "Daily inference workflow for Economedia PTS"

  source_contents = file("${path.module}/../../workflows/inference_workflow.yaml")
}
