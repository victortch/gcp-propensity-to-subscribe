resource "google_bigquery_dataset" "pts" {
  dataset_id                  = var.dataset
  project                     = var.project_id
  location                    = var.region
  delete_contents_on_destroy  = false
  default_table_expiration_ms = null

  labels = {
    purpose = "pts"
  }
}

resource "google_bigquery_table" "predictions_daily" {
  dataset_id = google_bigquery_dataset.pts.dataset_id
  table_id   = "predictions_daily"

  time_partitioning {
    type  = "DAY"
    field = "scoring_date"
  }

  clustering = ["label", "user_id"]

  schema = file("bq/ddl/create_predictions_daily.sql.json")
}

resource "google_bigquery_table" "train_metrics" {
  dataset_id = google_bigquery_dataset.pts.dataset_id
  table_id   = "train_metrics"

  time_partitioning {
    type  = "DAY"
    field = "created_at"
  }

  clustering = ["label_tag"]

  schema = file("bq/ddl/train_metrics_summary.json")
}

resource "google_bigquery_table" "train_metrics_detail" {
  dataset_id = google_bigquery_dataset.pts.dataset_id
  table_id   = "train_metrics_detail"

  clustering = ["label", "split"]

  schema = file("bq/ddl/train_metrics_detail.json")
}
