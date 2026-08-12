import asyncio
import forge as at

async def main():
    print("Testing AMEVA Tensor Python Wrapper Internally...")
    print(f"Initial Version: {at.__version__}")
    
    # Check if GPU is available (should be False in pure CPython unless pyodide is active)
    await at.init()
    print(f"Device initialized to: {at.current_device()}")

    # CPU Tensor Creation & Ops
    a = at.random((128, 128), device="cpu")
    b = at.random((128, 128), device="cpu")
    print(f"Created tensor A: {a}")
    print(f"Created tensor B: {b}")
    
    # Matmul
    c = a @ b
    print(f"C = A @ B shape: {c.shape}")
    
    # Relu
    d = at.relu(c)
    print(f"D = ReLU(C) shape: {d.shape}")
    
    # Readback
    val = d.numpy()
    print(f"Readback successful, mean value: {val.mean():.4f}")
    print("Integration test passed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
