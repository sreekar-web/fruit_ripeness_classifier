#!/bin/bash

echo "=== Refreshing ECR secret ==="
aws ecr get-login-password --region us-east-1 | sudo kubectl create secret docker-registry ecr-secret \
  --docker-server=664418990900.dkr.ecr.us-east-1.amazonaws.com \
  --docker-username=AWS \
  --docker-password=$(aws ecr get-login-password --region us-east-1) \
  --dry-run=client -o yaml | sudo kubectl apply -f -

echo "=== Restarting deployment ==="
sudo kubectl rollout restart deployment fruit-classifier -n default

echo "=== Waiting 60 seconds for new pod to start ==="
sleep 60

echo "=== Cleaning up old/stuck pods ==="
sudo kubectl delete pods --field-selector=status.phase!=Running -n default 2>/dev/null || true

echo "=== Pod status ==="
sudo kubectl get pods -A

echo "=== Done! App should be available at http://$(curl -s ifconfig.me):30000 ==="