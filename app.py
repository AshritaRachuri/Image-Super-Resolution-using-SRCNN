from flask import Flask,render_template,url_for,request,redirect,flash
import cv2
import numpy as np
import base64
from app_torch_model import load_model, predicted_img 


app=Flask(__name__, static_folder='static')
app.secret_key="mysecretkey"
model = load_model('model_weights.pth')
img = None
temp_files={}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload_img" , methods=["POST"])
def upload_page():

    if 'image' not in request.files:
        print("No image uploaded, please upload an image.")
        return redirect(url_for("home"))
     
    file =request.files['image']

    if file.filename=="":
        print("no img uploaded pls upload a image")
        return redirect(url_for("home"))
    
    if file:
       # Read the image file in memory as a NumPy array
        file_data=file.read()
        file_bytes = np.frombuffer(file_data, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Convert the image to JPEG format in memory
        _, buffer = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Create base64 string for embedding in HTML
        img_data = f"data:image/jpeg;base64,{img_base64}"
        global temp_files
        temp_files['uploaded_img']=img_base64

        return render_template("upload.html", uploaded_img=img_data)
     
    return redirect(url_for("home")) 
    
@app.route("/display", methods=["GET", "POST"])
def display():
    global temp_files
    if temp_files['uploaded_img'] is None:
        print("No image uploaded. Please upload an image.")
        return redirect(url_for("home"))

    try:
        # Convert base64 back to OpenCV image
        img_data = base64.b64decode(temp_files['uploaded_img'])
        file_bytes = np.frombuffer(img_data, np.uint8)
        uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if uploaded_image is None:
            raise ValueError("Error processing image.")

        # Perform prediction (Replace with actual model)
        predicted = predicted_img(model, uploaded_image)  # Your CNN model function
        predicted_image = cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR)

        # Convert images to base64
        _, buffer1 = cv2.imencode('.jpg', uploaded_image)
        uploaded_image_base64 = base64.b64encode(buffer1).decode('utf-8')

        _, buffer2 = cv2.imencode('.jpg', predicted_image)
        predicted_image_base64 = base64.b64encode(buffer2).decode('utf-8')

        return render_template("result.html",
                               uploaded_image_base64=uploaded_image_base64,
                               predicted_image_base64=predicted_image_base64)

    except Exception as e:
        flash(f"Error: {str(e)}")
        return redirect(url_for("home"))

if __name__=="__main__":
    app.run(debug=True)