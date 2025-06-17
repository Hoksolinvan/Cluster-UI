from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import Clustering as clustering


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class DBSCANParams(BaseModel):
    data: list[list[float]]
    eps: float
    min_samples: int
    mode: str


class KMeansParams(BaseModel):
    data: list[list[float]]
    k: int
    #min_samples: int
    max_iterations: int = 100
    mode: str

class DataPoints(BaseModel):
    data: list[list[float]]


global_status = {}
global_json = {}


@app.post("/clustering/dbscan")
def dbscan_clustering(params: DBSCANParams):

    global global_status
    global_status["previous_mode"] = params.mode
    global_status["previous_clustering"] = "DBSCAN"

    if(params.mode == "CPU"):
        result = clustering.Sci_kitDBSCAN(params.data,params.eps, params.min_samples)
    
    # elif(params.mode =="GPU"):
    #     result = clustering.CUML_DBSCAN(params.data,params.eps, params.min_samples)
    else:
        result = clustering.DBSCAN(params.data,params.eps, params.min_samples)


    return {"message": "DBSCAN clustering completed successfully.",
           "result":result}


@app.post("/clustering/kmeans")
def kmeans_clustering(params: KMeansParams):

    global global_status
    global_status['previous_mode'] = params.mode
    global_status['previous_clustering'] = "KMEANS"

    if(params.mode == "CPU"):
        result = clustering.Sci_kitKMeans(params.data, params.k, params.max_iterations)
    
    # elif(params.mode =="GPU"):
    #     result = clustering.CUML_KMeans(params.data, params.k, params.max_iterations)
    else:
        result = clustering.KMeans(params.data, params.k, params.max_iterations)
    
    return {"message": "KMeans clustering completed successfully.",
            "clusters": result
    }


@app.post("/clustering/upload")
async def upload_file(file):
    global global_json
    content = await file.read()
    global_json = json.loads(content)
    return {"message": "File upload successful."}



@app.get("/clustering/available")
def available_algorithms():
    return {
        "algorithms": ["DBSCAN", "KMeans"],
        "modes": ["CPU", "GPU", "Custom"],
        "Dimensions": ["2D"]
    }

@app.get("/clustering/status")
def clustering_status():
    return {
        "previous_mode": global_status.get("previous_mode", "None"),
        "previous_clustering": global_status.get("previous_clustering", "None")
    }


