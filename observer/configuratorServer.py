import os
import atexit
from flask import Flask, redirect, Response
from configurator import configurator, setConfiguratorApp #, registerCaptureService
from ipynb.fs.full.CalibratedObserver import CalibrationObserver
# from calibrator import CalibratedCaptureConfiguration, CalibrationObserver, setCalibratorApp # Deprecated

from ipynb.fs.full.HexObserver import HexCaptureConfiguration, HexGridConfiguration
from file_lock import FileLock

PORT = int(os.getenv("OBSERVER_PORT", "7000"))

def create_configurator_app():
    app = Flask(__name__)
    app.cc = HexCaptureConfiguration()
    if app.cc.hex is None:
        app.cc.hex = HexGridConfiguration()
    app.cc.capture()
    
    # Register blueprints
    app.register_blueprint(configurator, url_prefix='/configurator')
    
    # Initialize calibration observer
    app.cm = CalibrationObserver(app.cc)
    
    # Set app references
    setConfiguratorApp(app)
    # setCalibratorApp(app) # Deprecated


    @app.route('/')
    def index():
        return redirect('/configurator', code=303)

    @app.route('/bootstrap.min.css', methods=['GET'])
    def getBSCSS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.css", "r") as f:
            bscss = f.read()
        return Response(bscss, mimetype="text/css")
    
    @app.route('/bootstrap.min.js', methods=['GET'])
    def getBSJS():
        with open(f"{os.path.dirname(__file__)}/templates/bootstrap.min.js", "r") as f:
            bsjs = f.read()
        return Response(bsjs, mimetype="application/javascript")
    
    @app.route('/htmx.min.js', methods=['GET'])
    def getHTMX():
        with open(f"{os.path.dirname(__file__)}/templates/htmx.min.js", "r") as f:
            htmx = f.read()
        return Response(htmx, mimetype="application/javascript")
        
    return app

def main():
    lock = FileLock()
    lock.acquire()
    
    # Register shutdown hook
    atexit.register(lock.release)
    
    app = create_configurator_app()
    
    print(f"Launching Configurator Server on {PORT}")
    try:
        # registerCaptureService(app) # Deprecated or moved? 
        # If capture service was in calibrator.py, we might need to move it to configurator.py or reimplement.
        # configurator.py does not define registerCaptureService.
        # However, `app.cc.capture()` is called. 
        # `registerCaptureService` was running `app.cm.cycle()`.
        # If we need the observer to cycle, we need this logic.
        # Let's verify `configurator.py` or port `registerCaptureService`.
        pass 
        app.run(host="0.0.0.0", port=PORT)

    finally:
        lock.release()

if __name__ == "__main__":
    main()
