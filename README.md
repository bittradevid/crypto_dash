gcloud builds submit --tag gcr.io/morningbriefing-447904/morningbriefing1.3  --project=morningbriefing-447904

gcloud run deploy --image gcr.io/morningbriefing-447904/morningbriefing1.3 --platform managed  --project=morningbriefing-447904 --allow-unauthenticated
