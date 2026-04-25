juju consume -m uk8sx:kubeflow uk8sx:cos.prometheus-receive-remote-write
juju consume -m uk8sx:kubeflow uk8sx:cos.grafana-dashboards
juju consume -m uk8sx:kubeflow uk8sx:cos.loki-logging


juju integrate -m uk8sx:kubeflow grafana-agent-k8s:send-remote-write prometheus-receive-remote-write
juju integrate -m uk8sx:kubeflow grafana-agent-k8s:grafana-dashboards-provider grafana-dashboards
juju integrate -m uk8sx:kubeflow grafana-agent-k8s:logging-consumer loki-logging

juju status -m uk8sx:cos grafana-agent-k8s --relations


juju switch uk8sx:kubeflow
juju integrate argo-controller:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate dex-auth:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate envoy:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate istio-ingressgateway:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate istio-pilot:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate jupyter-controller:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate katib-controller:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate katib-db:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate kfp-api:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate kfp-db:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate knative-operator:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate knative-eventing:otel-collector knative-operator:otel-collector
juju integrate knative-serving:otel-collector knative-operator:otel-collector
juju integrate kserve-controller:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate kubeflow-dashboard:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate kubeflow-profiles:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate metacontroller-operator:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate minio:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate pvcviewer-operator:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate tensorboard-controller:metrics-endpoint grafana-agent-k8s:metrics-endpoint
juju integrate training-operator:metrics-endpoint grafana-agent-k8s:metrics-endpoint

juju switch uk8sx:kubeflow
juju integrate admission-webhook:logging grafana-agent-k8s:logging-provider
juju integrate argo-controller:logging grafana-agent-k8s:logging-provider
juju integrate dex-auth:logging grafana-agent-k8s:logging-provider
juju integrate envoy:logging grafana-agent-k8s:logging-provider
juju integrate jupyter-controller:logging grafana-agent-k8s:logging-provider
juju integrate jupyter-ui:logging grafana-agent-k8s:logging-provider
juju integrate katib-controller:logging grafana-agent-k8s:logging-provider
juju integrate katib-db-manager:logging grafana-agent-k8s:logging-provider
juju integrate katib-db:logging grafana-agent-k8s:logging-provider
juju integrate katib-ui:logging grafana-agent-k8s:logging-provider
juju integrate kfp-api:logging grafana-agent-k8s:logging-provider
juju integrate kfp-db:logging grafana-agent-k8s:logging-provider
juju integrate kfp-metadata-writer:logging grafana-agent-k8s:logging-provider
juju integrate kfp-persistence:logging grafana-agent-k8s:logging-provider
juju integrate kfp-profile-controller:logging grafana-agent-k8s:logging-provider
juju integrate kfp-schedwf:logging grafana-agent-k8s:logging-provider
juju integrate kfp-ui:logging grafana-agent-k8s:logging-provider
juju integrate kfp-viewer:logging grafana-agent-k8s:logging-provider
juju integrate kfp-viz:logging grafana-agent-k8s:logging-provider
juju integrate knative-operator:logging grafana-agent-k8s:logging-provider
juju integrate kserve-controller:logging grafana-agent-k8s:logging-provider
juju integrate kubeflow-dashboard:logging grafana-agent-k8s:logging-provider
juju integrate kubeflow-profiles:logging grafana-agent-k8s:logging-provider
juju integrate kubeflow-volumes:logging grafana-agent-k8s:logging-provider
juju integrate mlmd:logging grafana-agent-k8s:logging-provider
juju integrate oidc-gatekeeper:logging grafana-agent-k8s:logging-provider
juju integrate pvcviewer-operator:logging grafana-agent-k8s:logging-provider
juju integrate tensorboard-controller:logging grafana-agent-k8s:logging-provider
juju integrate tensorboards-web-app:logging grafana-agent-k8s:logging-provider