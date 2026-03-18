import os
import atexit
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from configurator import configurator
from observer.CalibratedObserver import CalibrationObserver, CalibratedCaptureConfiguration

from observer.HexObserver import HexCaptureConfiguration, HexGridConfiguration
from file_lock import FileLock

PORT = int(os.getenv("OBSERVER_PORT", "7000"))

def create_configurator_app():
    app = FastAPI()
    app.state.cc = CalibratedCaptureConfiguration()
    app.state.cc.capture()
    
    # Initialize calibration observer
    app.state.cm = CalibrationObserver(app.state.cc)
    
    # Include new FastAPI configurator router
    app.include_router(configurator)


    @app.get('/')
    async def index():
        return RedirectResponse('/configurator/', status_code=303)

    @app.get('/bootstrap.min.css')
    async def getBSCSS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.css", "r") as f:
            bscss = f.read()
        return Response(bscss, media_type="text/css")
    
    @app.get('/bootstrap.min.js')
    async def getBSJS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.js", "r") as f:
            bsjs = f.read()
        return Response(bsjs, media_type="application/javascript")
    
    @app.get('/htmx.min.js')
    async def getHTMX():
        with open(f"{os.path.dirname(__file__)}/templates/htmx.min.js", "r") as f:
            htmx = f.read()
        return Response(htmx, media_type="application/javascript")
        
    return app

def main():
    lock = FileLock()
    lock.acquire()
    
    # Register shutdown hook
    atexit.register(lock.release)
    
    app = create_configurator_app()
    
    print(f"Launching Configurator Server on {PORT}")
    try:
        uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

    finally:
        lock.release()

if __name__ == "__main__":
    main()
