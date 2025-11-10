resource "google_storage_bucket" "models" {
  name                        = var.gcs_bucket
  location                    = var.region
  uniform_bucket_level_access = true
  force_destroy               = false

  labels = {
    purpose = "pts-models"
  }
}
