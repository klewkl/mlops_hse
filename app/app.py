import uvicorn
from api import app, config


# интерфейс для тестирования http://127.0.0.1:8000/docs#/
if __name__ == "__main__":
    uvicorn.run(app, port=config.PORT, host=config.HOST)
