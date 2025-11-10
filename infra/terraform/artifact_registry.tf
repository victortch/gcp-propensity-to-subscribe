resource "google_artifact_registry_repository" "pts" {
  location      = var.region
  repository_id = var.artifact_repo
  description   = "Economedia PTS containers"
  format        = "DOCKER"
}
