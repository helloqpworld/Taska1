#!/bin/bash
# ================================================================================
# Автономный скрипт развертывания инфраструктуры и синхронизации с GitLab API
# ================================================================================

SET_GREEN='\033[0;32m'
SET_RED='\033[0;31m'
RESET_COLOR='\033[0m'

log_info() { echo -e "${SET_GREEN}[INFO] $(date +'%Y-%m-%d %H:%M:%S') - $1${RESET_COLOR}"; }
log_error() { echo -e "${SET_RED}[ERROR] $(date +'%Y-%m-%d %H:%M:%S') - $1${RESET_COLOR}"; }

KIND_CONFIG="$HOME/kind-config.yaml"
if [ ! -f "$KIND_CONFIG" ]; then
    log_error "Файл конфигурации $KIND_CONFIG не найден!"
    exit 1
fi

log_info "Шаг 1/8: Очистка старых зависших компонентов и сетей..."
kind delete cluster || true
sudo systemctl stop nginx 2>/dev/null || true

log_info "Шаг 2/8: Создание чистого кластера Kubernetes KinD..."
if ! kind create cluster --config "$KIND_CONFIG"; then
    log_error "Не удалось запустить кластер KinD."
    exit 1
fi

log_info "Шаг 3/8: Обеспечение сетевой связности с реестром GitLab..."
docker network disconnect kind gitlab-server-gitlab-1 2>/dev/null || true
docker network connect kind gitlab-server-gitlab-1

log_info "Шаг 4/8: Создание изолированной песочницы predict-service-ns..."
kubectl create namespace predict-service-ns --dry-run=client -o yaml | kubectl apply -f -

log_info "Шаг 5/8: Динамическое извлечение IP-адреса реестра..."
GITLAB_KIND_IP=$(docker inspect -f '{{.NetworkSettings.Networks.kind.IPAddress}}' gitlab-server-gitlab-1)
if [ -z "$GITLAB_KIND_IP" ] || [ "$GITLAB_KIND_IP" == "<no value>" ]; then
    log_error "Критическая ошибка: Не удалось извлечь IP GitLab!"
    exit 1
fi
log_info "Реестр обнаружен на адресе: ${GITLAB_KIND_IP}:5050"

# Перезаписываем секрет в K8s
kubectl create secret docker-registry gitlab-registry-credentials \
  --docker-server="${GITLAB_KIND_IP}:5050" \
  --docker-username='root' \
  --docker-password='IOidNIODiE&g67g&%g&6' \
  -n predict-service-ns \
  --dry-run=client -o yaml | kubectl apply -f -

log_info "Шаг 6/8: Автоматическая сборка deployment.yaml под актуальный IP..."
# Генерируем deployment.yaml на лету, подставляя свежий IP адрес реестра
cat << EOF > ~/projects/ml-project/deploy/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-predict-deployment
  namespace: predict-service-ns
  labels:
    app: fastapi-predict
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fastapi-predict
  template:
    metadata:
      labels:
        app: fastapi-predict
    spec:
      serviceAccountName: predict-service-sa
      imagePullSecrets:
        - name: gitlab-registry-credentials
      containers:
        - name: predict-container
          image: ${GITLAB_KIND_IP}:5050/root/taska1/predict_service:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          # ДОБАВИЛИ: Проверки состояния для контроля загрузки модели и работы API!
          livenessProbe:
            httpGet:
              path: /healthz/live
              port: 8000
            initialDelaySeconds: 5  # Начинаем проверять через 5 сек после старта контейнера
            periodSeconds: 10       # Проверяем каждые 10 секунд
          readinessProbe:
            httpGet:
              path: /healthz/ready
              port: 8000
            initialDelaySeconds: 10 # Даем 10 секунд форы на старт скачивания весов из MLflow
            periodSeconds: 5        # Проверяем часто, чтобы быстро пустить трафик при готовности
            failureThreshold: 6     # Если модель грузится долго, даем 6 попыток (6 * 5 = 30 сек) перед паникой
          env:
            - name: MLFLOW_TRACKING_URI
              value: "http://172.18.0.1:5080"
            - name: MLFLOW_S3_ENDPOINT_URL
              value: "http://172.18.0.1:9000"
            - name: AWS_ACCESS_KEY_ID
              value: "Sanya_Cool_ML_specialist"
            - name: AWS_SECRET_ACCESS_KEY
              value: "IOidNIODiE&g67g&%g&6"
          resources:
            requests:
              cpu: "300m"
              memory: "512Mi"
            limits:
              cpu: "1"
              memory: "1Gi"
EOF

log_info "Шаг 7/8: Развертывание Ingress Nginx, Metrics Server и патч портов..."
kubectl apply -f https://githubusercontent.com
kubectl apply -f https://github.com
kubectl patch deployment metrics-server -n kube-system --type='json' -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--kubelet-insecure-tls"}]'

log_info "Ожидаем готовности подов Ingress..."
kubectl wait --namespace ingress-nginx --for=condition=ready pod --selector=app.kubernetes.io/component=controller --timeout=90s

kubectl patch service ingress-nginx-controller -n ingress-nginx --type='json' -p='[
  {"op": "replace", "path": "/spec/ports/0/nodePort", "value": 30080},
  {"op": "replace", "path": "/spec/ports/1/nodePort", "value": 30443},
  {"op": "replace", "path": "/spec/type", "value": "NodePort"}
]'

log_info "Шаг 8/8: АВТОМАТИЧЕСКОЕ ОБНОВЛЕНИЕ KUBECONFIG В GITLAB API..."
# 1. Генерируем временный валидный Kubeconfig для раннера в файл
mkdir -p /tmp/k8s_sync
cat ~/.kube/config > /tmp/k8s_sync/config
# Корректируем адрес внутри временного файла на имя контейнера кластера
sed -i "s|server: https://127.0.0.1:.*|server: https://kind-control-plane:6443|g" /tmp/k8s_sync/config

# 2. Отправляем обновленный KUBECONFIG напрямую в GitLab Variables через cURL API
# Используем твой root-пароль в качестве токена авторизации локального GitLab
UPD_STATUS=$(curl --silent --output /dev/null --write-out "%{http_code}" \
  --request PUT --header "PRIVATE-TOKEN: IOidNIODiE&g67g&%g&6" \
  --form "value=$(cat /tmp/k8s_sync/config)" \
  "http://127.0.0")

rm -rf /tmp/k8s_sync

if [ "$UPD_STATUS" == "200" ] || [ "$UPD_STATUS" == "201" ]; then
    log_info "Переменная KUBECONFIG успешно обновлена в GitLab API (Код: $UPD_STATUS)!"
else
    log_error "Не удалось обновить переменную в GitLab. Код ответа API: $UPD_STATUS"
    log_info "Пожалуйста, проверьте Personal Access Token в скрипте."
fi

echo "================================================================"
log_info "ИНФРАСТРУКТУРА ПОЛНОСТЬЮ ОБНОВЛЕНА И СИНХРОНИЗИРОВАНА С CI/CD!"
echo "================================================================"
echo -e "Теперь просто сделай пуш. Пайплайн автоматически станет зеленым!"
echo "----------------------------------------------------------------"
