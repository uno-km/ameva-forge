import ameva_tensor as at

async def main():
    print("In Pyodide E2E test!")
    await at.init()
    print(f"Device initialized to: {at.current_device()}")
    
    # Run Matmul
    dev = at.current_device()
    a = at.random((128, 128), device=dev)
    b = at.random((128, 128), device=dev)
    c = a @ b
    d = at.relu(c)
    
    # Readback
    res = await d.numpy_async()
    print(f"GPU computation complete. Result shape: {res.shape}, Mean: {res.mean()}")

import asyncio
asyncio.ensure_future(main())
