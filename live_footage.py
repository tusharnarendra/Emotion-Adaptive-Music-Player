from flask import Flask, render_template, Response
from picamera2 import Picamera2
import cv2
import time
import joblib
from model_features import xFeatures

# Create the flask app
app = Flask(__name__)

# Load the trained ML_model 
regressor = joblib.load('svr_multioutput_model.pkl')
sc_X = joblib.load('X_scaler.pkl')
sc_y = joblib.load('y_scaler.pkl')

#Features extraction
extractor = xFeatures()

def gen_frames():
    # Initialize Picamera2 inside the generator function
    picam = Picamera2()
    time.sleep(1)  # Allow time for camera initialization
    picam.configure(picam.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)}))
    picam.start()

    while True:
        # Capture a frame from the camera
        frame = picam.capture_array()
        # Check if the camera correctly captured a frame
        if frame is None or frame.size == 0:
            print("Failed to read from the camera")
            break
        else:
            print("Reading from camera")
            # Encode to JPEG and convert to bytes
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode the frame")
            features = extractor.process_frame(frame)
            if features is not None:
                print("Features extracted")
            else:
                print("No face detected")
            X_scaled = sc_X.transform([features])
            y_pred = sc_y.inverse_transform(regressor.predict(X_scaled))
            valence = y_pred[0][0]
            arousal = y_pred[0][1]
            print(f"Valence: {valence} Arousal: {arousal}")
            frame = buffer.tobytes()
            # Yield as part of HTTP response (produce a sequence of values over time one at a time instead of all at once)
            yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Stop the camera after the stream ends
    picam.stop()

# HTML page when root of URL is visited
@app.route('/')
def index():
    return render_template('index.html')

# Stream of JPEG frames is displayed when browser requests /video
@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Check if current script is being run directly or as a module, and then run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
