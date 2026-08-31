# ClusterUI

A lightweight web app for running and visualizing clustering algorithms (DBSCAN, K-Means) on 2D data, with a FastAPI backend and a browser-based frontend.

Live demo: [clusterui.vercel.app](https://clusterui.vercel.app)

## Features

- **DBSCAN** and **K-Means** clustering on 2D point data
- Multiple execution modes per algorithm:
  - `Custom` — from-scratch Python implementation (no external ML libraries)
  - `CPU` — scikit-learn implementation
  - `GPU` — cuML implementation (planned / not yet enabled)
- Simple REST API for running clustering and checking status
- Minimal HTML frontend for uploading/visualizing data

## Tech Stack

- **Backend:** FastAPI, Pydantic, scikit-learn, NumPy
- **Frontend:** HTML (`index.html`)
- **Planned:** GPU-accelerated clustering via [cuML](https://docs.rapids.ai/api/cuml/stable/)

## Project Structure

```
Cluster-UI/
├── backend.py         # FastAPI app and API routes
├── Clustering.py       # Clustering algorithm implementations (custom, scikit-learn, cuML)
├── index.html          # Frontend UI
└── requirements.txt    # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Hoksolinvan/Cluster-UI.git
cd Cluster-UI
pip install -r requirements.txt
```

### Running the backend

```bash
uvicorn backend:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### Running the frontend

Open `index.html` in a browser, or serve it with any static file server. Ensure the frontend is configured to point at your running backend URL.

## API Reference

### `POST /clustering/dbscan`

Run DBSCAN clustering on a set of 2D points.

**Body:**
```json
{
  "data": [[0.0, 0.0], [1.0, 1.0]],
  "eps": 0.5,
  "min_samples": 5,
  "mode": "CPU"
}
```

- `mode`: `"CPU"` (scikit-learn), `"Custom"` (from-scratch), or `"GPU"` (not yet enabled)

**Response:**
```json
{
  "message": "DBSCAN clustering completed successfully.",
  "result": [[[x1, y1], [x2, y2]], ...]
}
```

### `POST /clustering/kmeans`

Run K-Means clustering on a set of 2D points.

**Body:**
```json
{
  "data": [[0.0, 0.0], [1.0, 1.0]],
  "k": 3,
  "max_iterations": 100,
  "mode": "CPU"
}
```

**Response:**
```json
{
  "message": "KMeans clustering completed successfully.",
  "clusters": [[[x1, y1], [x2, y2]], ...]
}
```

### `POST /clustering/upload`

Upload a data file for clustering.

### `GET /clustering/available`

Returns available algorithms, execution modes, and supported dimensions.

### `GET /clustering/status`

Returns the most recently run algorithm and mode.

## Roadmap

- [ ] Enable GPU-accelerated clustering (cuML) execution paths
- [ ] Support dimensions beyond 2D
- [ ] Additional clustering algorithms
- [ ] Improved frontend visualization

## License

No license specified yet.
