variable "project_id" {
  type        = string
  description = "GCP project hosting the Economedia PTS platform"
  default     = "economedia-data-prod-laoy"
}

variable "region" {
  type        = string
  description = "Primary region for regional resources"
  default     = "europe-west3"
}

variable "artifact_repo" {
  type        = string
  description = "Artifact Registry repository name"
  default     = "pts_model"
}

variable "gcs_bucket" {
  type        = string
  description = "Bucket for model artifacts"
  default     = "economedia-pts-models"
}

variable "dataset" {
  type        = string
  description = "BigQuery dataset for PTS"
  default     = "propensity_to_subscribe"
}
