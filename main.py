from fastapi import FastAPI, HTTPException, Query
from pathlib import Path
import aiofiles
from app.tasks import execute_task

app = FastAPI()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

@app.post("/run")
async def run_task(task: str = Query(..., description="Task description")):
    """
    Runs a given task by processing the task description and executing necessary steps.
    """
    try:
        result = await execute_task(task)
        return {"status": "success", "message": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.get("/read")
async def read_file(file_name: str = Query(..., description="File name inside the /data directory")):
    """
    Securely reads and returns the content of a file inside the /data directory.
    """
    file_path = (DATA_DIR / file_name).resolve()

    # Security Check: Ensure file is within /data and prevent directory traversal attacks
    if not str(file_path).startswith(str(DATA_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
            content = await file.read()
        return {"status": "success", "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

# @app.get("/read")
# async def read_file(file_name: str = Query(..., description="File name inside the /data directory")):
#     """
#     Securely reads and returns the content of a file inside the /data directory.
#     """
#     file_path = DATA_DIR / file_name

#     # Security Check: Prevent accessing files outside /data
#     if not file_path.resolve().is_relative_to(DATA_DIR):
#         raise HTTPException(status_code=400, detail="Invalid file path")

#     if not file_path.exists():
#         raise HTTPException(status_code=404, detail="File not found")

#     try:
#         async with aiofiles.open(file_path, "r", encoding="utf-8") as file:
#             content = await file.read()
#         return {"status": "success", "content": content}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
