# Week 10 Notes : Deploy Machine Learning Models with Docker, FastAPI & Kubernetes using kind

### Workshop Video 

----
[![Deploy Machine Learning Models with Docker, FastAPI & Kubernetes using kind](https://img.youtube.com/vi/c_CzCsCnWoU/0.jpg)](https://www.youtube.com/live/c_CzCsCnWoU)

Reference Notes [here](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/10-kubernetes/workshop)


## Goal
- Deploy the ONNX model using FastAPI 
- Containerize the FastAPI app using Docker
- Deploy the Docker container to a local Kubernetes cluster using kind
- Scale horizontally and manage deployments using Kubernetes
- Monitor the deployed application

### Setup Kubectl, kind
1. Install kubectl using brew and check the version (client only):
2. Install kind using brew and check the version:

```
brew install kubectl
kubectl version --client
brew install kind
kind --version
```

### Setup Service
1. download the model from previous week
```
mkdir service
cd service
wget https://github.com/DataTalksClub/machine-learning-zoomcamp/releases/download/dl-models/clothing_classifier_mobilenet_v2_latest.onnx -O clothing-model.onnx
```

2. Install uv dependencies

```
uv add fastapi uvicorn onnxruntime keras-image-helper numpy`
```

3. Run app.py
- Fetch app.py from [here](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/10-kubernetes/workshop/service/app.py)
- Run the app using uvicorn
```
uv run uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

4. Test the app by using the http://127.0.0.1:8080/docs endpoint.
Sample URL used
```
{
  "url": "https://bit.ly/mlbookcamp-pants"
}
```

### Containerize using Docker
- Package the app.py and clothing-model.onnx into a Docker container
- Packaging models with the app helps in consistency across environments and also in scaling while deploying to kubernetes (auto-scaling)
- Refer pyproject.toml, uv.lock, Dockerfile from [here](./service)
- Build the docker image
```
docker build -t clothing-classifier:v1 .
```
- Run the docker container
```
docker run -it --rm -p 8080:8080 clothing-classifier:v1
```

- Test the app using the same endpoint as it was run locally using uvicorn

### K8s concepts
- Cluster: A set of nodes (machines) that run containerized applications managed by Kubernetes.
- Node: A single machine in a Kubernetes cluster (can be a physical or virtual machine).
- Pod: The smallest deployable unit in Kubernetes, which can contain one or more containers.
- Deployment: A higher-level abstraction that manages a set of identical pods, ensuring the desired number
- Service: An abstraction that defines a logical set of pods and a policy to access them, enabling communication between different parts of the application.
 
- Namespace: A virtual cluster within a Kubernetes cluster that provides a way to divide cluster resources between
- ConfigMap: An API object used to store non-confidential configuration data in key-value pairs.
- Secret: An API object used to store sensitive information, such as passwords or API keys,

- Ingress: An API object that manages external access to services within a cluster, typically HTTP/HTTPS.
- Egress: The outbound traffic from the cluster to external services.

- Label: A key-value pair attached to Kubernetes objects, used for organization and selection.
- Selector: A query that selects Kubernetes objects based on their labels.
- ReplicaSet: A controller that ensures a specified number of pod replicas are running at any given time.
- StatefulSet: A controller that manages stateful applications, providing guarantees about the ordering and uniqueness
- DaemonSet: A controller that ensures a copy of a pod is running on all or some nodes in the cluster.

- Job: A controller that creates one or more pods to perform a specific task and ensures that
- CronJob: A controller that creates jobs on a scheduled basis, similar to cron jobs in Unix/Linux systems.

- ServiceAccount: An account used by pods to interact with the Kubernetes API server.
- Role and RoleBinding: Objects that define permissions for accessing resources within a namespace.
- ClusterRole and ClusterRoleBinding: Objects that define permissions for accessing resources across the entire cluster.

- Horizontal Pod Autoscaler (HPA): A controller that automatically scales the number of pod replicas based on observed CPU utilization or other select metrics.
 
- Volume: A storage resource that can be used by containers in a pod to persist data.
- PersistentVolume (PV): A piece of storage in the cluster that has been provisioned by an administrator or dynamically provisioned using StorageClasses.
- PersistentVolumeClaim (PVC): A request for storage by a user that binds to a Persistent
- StorageClass: A way to describe the "classes" of storage available in a cluster, allowing dynamic provisioning of PersistentVolumes.

#### Deployment yaml
```
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```
- liveness sends to `/health` endpoint to check if the app is alive
- readiness sends to `/health` endpoint to check if the app is ready to serve traffic

### Create and Start the local k8s cluster using kind
**create cluster**
```
kind create cluster --name mlzoomcamp
kind get clusters
```

**check using kubectl**
```
kubectl cluster-info
kubectl get nodes
```

**load docker image to kind cluster**
Note : This is done as kind runs its own docker daemon separate from the host machine.

```
kind load docker-image clothing-classifier:v1 --name mlzoomcamp
```

### Deploy the app (model) to k8s cluster
- Fetch deployment.yaml and service.yaml from [here](./k8s)
- Apply the deployment and service yaml files
- Check the pods and services created
```
kubectl apply -f deployment.yaml
kubectl get pods
kubectl get deployments

kubectl apply -f service.yaml
kubectl get services
```

Port forward the service to localhost
```
kubectl port-forward service/clothing-classifier 30080:8080
```
Accessing the app using http://localhost:30080/docs should display the swagger UI for the deployed app.

### Debugging
for getting the event logs

```
(ml-zc-2025) ➜  k8s git:(main) ✗ kubectl get events --sort-by='.lastTimestamp'
LAST SEEN   TYPE      REASON              OBJECT                                            MESSAGE
7m7s        Normal    SuccessfulCreate    replicaset/clothing-classifier-87ccc7864          Created pod: clothing-classifier-87ccc7864-tqxcm
7m7s        Normal    Pulled              pod/clothing-classifier-87ccc7864-7krw9           Container image "clothing-classifier:v1" already present on machine and can be accessed by the pod
7m7s        Normal    Created             pod/clothing-classifier-87ccc7864-7krw9           Container created
7m7s        Normal    ScalingReplicaSet   deployment/clothing-classifier                    Scaled up replica set clothing-classifier-87ccc7864 from 2 to 4
7m7s        Normal    Scheduled           pod/clothing-classifier-87ccc7864-7krw9           Successfully assigned default/clothing-classifier-87ccc7864-7krw9 to mlzoomcamp-control-plane
7m7s        Normal    SuccessfulRescale   horizontalpodautoscaler/clothing-classifier-hpa   New size: 4; reason: cpu resource utilization (percentage of request) above target
7m7s        Normal    Scheduled           pod/clothing-classifier-87ccc7864-tqxcm           Successfully assigned default/clothing-classifier-87ccc7864-tqxcm to mlzoomcamp-control-plane
7m7s        Normal    Pulled              pod/clothing-classifier-87ccc7864-tqxcm           Container image "clothing-classifier:v1" already present on machine and can be accessed by the pod
7m7s        Normal    Created             pod/clothing-classifier-87ccc7864-tqxcm           Container created
7m7s        Normal    SuccessfulCreate    replicaset/clothing-classifier-87ccc7864          Created pod: clothing-classifier-87ccc7864-7krw9
7m6s        Normal    Started             pod/clothing-classifier-87ccc7864-7krw9           Container started
7m6s        Normal    Started             pod/clothing-classifier-87ccc7864-tqxcm           Container started
...
```