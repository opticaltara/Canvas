/**
 * Kubernetes MCP Server
 * A simple Model Control Proxy for Kubernetes
 */

const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const k8s = require('@kubernetes/client-node');
const fs = require('fs');
const path = require('path');

// Create express app
const app = express();
const port = process.env.PORT || 8000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Get kubeconfig from environment variable
const kubeconfig = process.env.KUBECONFIG || path.join(process.env.HOME, '.kube', 'config');
const kubeContext = process.env.KUBE_CONTEXT;

// Create k8s client
let k8sClient;
let k8sAppsClient;

try {
  const kc = new k8s.KubeConfig();
  
  // Check if kubeconfig exists
  if (fs.existsSync(kubeconfig)) {
    kc.loadFromFile(kubeconfig);
    
    // Set context if provided
    if (kubeContext) {
      kc.setCurrentContext(kubeContext);
    }
    
    k8sClient = kc.makeApiClient(k8s.CoreV1Api);
    k8sAppsClient = kc.makeApiClient(k8s.AppsV1Api);
    
    console.log('Kubernetes client initialized successfully');
  } else {
    console.warn(`Kubeconfig not found at ${kubeconfig}. Some features will not work.`);
  }
} catch (error) {
  console.error('Error initializing Kubernetes client:', error);
}

// Standby mode - if enabled, reduces resource usage
const STANDBY_MODE = process.env.STANDBY_MODE === 'true';
console.log(`Standby mode: ${STANDBY_MODE ? 'enabled' : 'disabled'}`);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'ok' });
});

// API endpoint to get namespaces
app.get('/namespaces', async (req, res) => {
  try {
    if (!k8sClient) {
      return res.status(500).json({ error: 'Kubernetes client not initialized' });
    }
    
    const response = await k8sClient.listNamespace();
    const namespaces = response.body.items.map(ns => ({
      name: ns.metadata.name,
      status: ns.status.phase,
      creationTimestamp: ns.metadata.creationTimestamp
    }));
    
    res.json({ namespaces });
  } catch (error) {
    console.error('Error fetching namespaces:', error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint to get pods in a namespace
app.get('/namespaces/:namespace/pods', async (req, res) => {
  try {
    if (!k8sClient) {
      return res.status(500).json({ error: 'Kubernetes client not initialized' });
    }
    
    const namespace = req.params.namespace;
    const response = await k8sClient.listNamespacedPod(namespace);
    
    const pods = response.body.items.map(pod => ({
      name: pod.metadata.name,
      namespace: pod.metadata.namespace,
      status: pod.status.phase,
      podIP: pod.status.podIP,
      nodeName: pod.spec.nodeName,
      creationTimestamp: pod.metadata.creationTimestamp
    }));
    
    res.json({ pods });
  } catch (error) {
    console.error(`Error fetching pods in namespace ${req.params.namespace}:`, error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint to get deployments in a namespace
app.get('/namespaces/:namespace/deployments', async (req, res) => {
  try {
    if (!k8sAppsClient) {
      return res.status(500).json({ error: 'Kubernetes apps client not initialized' });
    }
    
    const namespace = req.params.namespace;
    const response = await k8sAppsClient.listNamespacedDeployment(namespace);
    
    const deployments = response.body.items.map(deployment => ({
      name: deployment.metadata.name,
      namespace: deployment.metadata.namespace,
      replicas: deployment.spec.replicas,
      availableReplicas: deployment.status.availableReplicas,
      creationTimestamp: deployment.metadata.creationTimestamp
    }));
    
    res.json({ deployments });
  } catch (error) {
    console.error(`Error fetching deployments in namespace ${req.params.namespace}:`, error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint to get services in a namespace
app.get('/namespaces/:namespace/services', async (req, res) => {
  try {
    if (!k8sClient) {
      return res.status(500).json({ error: 'Kubernetes client not initialized' });
    }
    
    const namespace = req.params.namespace;
    const response = await k8sClient.listNamespacedService(namespace);
    
    const services = response.body.items.map(service => ({
      name: service.metadata.name,
      namespace: service.metadata.namespace,
      type: service.spec.type,
      clusterIP: service.spec.clusterIP,
      ports: service.spec.ports.map(port => ({
        name: port.name,
        port: port.port,
        targetPort: port.targetPort,
        nodePort: port.nodePort
      })),
      creationTimestamp: service.metadata.creationTimestamp
    }));
    
    res.json({ services });
  } catch (error) {
    console.error(`Error fetching services in namespace ${req.params.namespace}:`, error);
    res.status(500).json({ error: error.message });
  }
});

// API endpoint to get logs from a pod
app.get('/namespaces/:namespace/pods/:pod/logs', async (req, res) => {
  try {
    if (!k8sClient) {
      return res.status(500).json({ error: 'Kubernetes client not initialized' });
    }
    
    const namespace = req.params.namespace;
    const podName = req.params.pod;
    const container = req.query.container;
    const tailLines = req.query.tailLines || 100;
    
    const logs = await k8sClient.readNamespacedPodLog(
      podName,
      namespace,
      container,
      undefined,
      false,
      undefined,
      undefined,
      undefined,
      undefined,
      tailLines
    );
    
    res.json({ logs: logs.body });
  } catch (error) {
    console.error(`Error fetching logs for pod ${req.params.pod}:`, error);
    res.status(500).json({ error: error.message });
  }
});

// Catch-all error handler
app.use((err, req, res, next) => {
  console.error('Unhandled error:', err);
  res.status(500).json({ 
    error: 'Internal server error',
    message: err.message 
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Kubernetes MCP server running on port ${port}`);
});