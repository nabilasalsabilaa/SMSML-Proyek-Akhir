from prometheus_client import start_http_server, Counter, Histogram
import time
import random

# Metrics
REQUEST_COUNT = Counter("request_count", "Total requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency")

# Start server
start_http_server(8000)
print("Prometheus exporter running on http://localhost:8000")

# Simulasi metrik
while True:
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(random.uniform(0.1, 1.0))
    time.sleep(2)