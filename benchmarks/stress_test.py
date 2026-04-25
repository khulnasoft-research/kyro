import asyncio
import aiohttp
import time
import statistics

API_URL = "http://localhost:3000/v1/chat/completions"
CONCURRENT_USERS = 50
REQUESTS_PER_USER = 10

async def send_request(session, user_id):
    payload = {
        "model": "kyro-llama3",
        "messages": [{"role": "user", "content": f"Hello from user {user_id}"}],
        "stream": False
    }
    
    start_time = time.perf_counter()
    async with session.post(API_URL, json=payload) as response:
        await response.text()
        end_time = time.perf_counter()
        return end_time - start_time

async def stress_test():
    print(f"Starting stress test with {CONCURRENT_USERS} concurrent users...")
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(CONCURRENT_USERS):
            for _ in range(REQUESTS_PER_USER):
                tasks.append(send_request(session, i))
        
        start_time = time.perf_counter()
        latencies = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        total_requests = len(latencies)
        rps = total_requests / total_time
        
        print("\n--- Stress Test Results ---")
        print(f"Total Requests: {total_requests}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Requests Per Second (RPS): {rps:.2f}")
        print(f"Average Latency: {statistics.mean(latencies):.4f}s")
        print(f"P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.4f}s")
        print(f"P99 Latency: {statistics.quantiles(latencies, n=100)[98]:.4f}s")

if __name__ == "__main__":
    asyncio.run(stress_test())
