from prometheus_client import start_http_server, Counter, Histogram
import time
import random

REQUEST_COUNT = Counter("request_count", "Total number of requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Latency of requests in seconds")

def process_request():
    REQUEST_COUNT.inc()
    latency = random.uniform(0.1, 1.5)
    REQUEST_LATENCY.observe(latency)
    time.sleep(latency)

if __name__ == "__main__":
    start_http_server(8000)
    while True:
        process_request()