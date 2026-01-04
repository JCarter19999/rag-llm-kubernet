# Local RAG on Kubernetes (Ollama + PVC + Jobs)

A fully local Retrieval-Augmented Generation (RAG) pipeline deployed with Docker + Kubernetes.  
The system separates **(1) document seeding**, **(2) ingestion/index build**, and **(3) query + LLM generation** into distinct Kubernetes Jobs and services.

This repo intentionally keeps everything local:
- No managed cloud services
- No external vector DB
- Reproducible container runtime + Kubernetes orchestration

---

## What this project does

Given a set of PDFs:

1. **Seed docs** into a shared persistent volume  
2. **Chunk + index** those docs and write `data/local_index.pkl` to another persistent volume  
3. **Query** the index to retrieve top-k chunks  
4. **Compose a prompt** (question + retrieved snippets)  
5. **Call Ollama** (`llama3.1:8b`) to generate an answer grounded in retrieved context

---

## High-level architecture

### Components

- **Ollama Deployment + Service**  
  Runs the local model and exposes HTTP endpoints (e.g. `/api/generate`, `/api/tags`)

- **seed-docs Job**  
  Copies PDFs from the container image into a docs PVC

- **rag-ingest Job**  
  Reads PDFs from the docs PVC, chunks them, builds an on-disk index (`local_index.pkl`) in the data PVC

- **rag-query Job**  
  Loads the index from the data PVC, retrieves top-k chunks, calls Ollama, prints answer to logs

---

### Why Kubernetes

- Even though this is local, Kubernetes is used to demonstrate production-shaped patterns:
- Reproducible runtime via containerized Jobs (no “works on my machine” drift)
- Pipeline staging (seed → ingest → query) as discrete jobs with explicit inputs/outputs
- Operational debugging (logs, describe, restart one stage without touching the rest)
- Persistent storage (PVCs for docs, indexes, and model cache)

### Why Docker

- Docker gives one build artifact (rag-local:latest) that includes:
- code + dependencies
- scripts for ingest/query
- (optionally) baked PDFs
- That artifact can run locally or under Kubernetes identically.

### Model choice
LLM: llama3.1:8b (Ollama)

Chosen because it’s a strong local baseline that’s easy to serve via HTTP and practical for a portfolio demo.

Retrieval index: local_index.pkl

This repo deliberately avoids external vector DBs to keep the system minimal and fully local.

Quick start (Kubernetes)
- kubectl apply -f k8s/00-namespace.yaml
- kubectl -n rag-local apply -f k8s/01-pvc.yaml
- kubectl -n rag-local apply -f k8s/01b-ollama-models-pvc.yaml
- kubectl -n rag-local apply -f k8s/02-ollama.yaml

### wait until ollama pod is Running, then pull model once:
- kubectl -n rag-local exec deploy/ollama -- ollama pull llama3.1:8b
- kubectl -n rag-local apply -f k8s/03-seed-docs-job.yaml
- kubectl -n rag-local logs -f job/rag-seed-docs
- kubectl -n rag-local apply -f k8s/04-job-ingest.yaml
- kubectl -n rag-local logs -f job/rag-ingest
- kubectl -n rag-local apply -f k8s/05-job-query.yaml
- kubectl -n rag-local logs -f job/rag-query

