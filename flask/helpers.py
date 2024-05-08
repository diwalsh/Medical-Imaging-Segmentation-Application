import cv2
import os
import numpy as np
from flask import redirect, render_template, session
from functools import wraps


# function to display funny apology to user upon login/registration failure
def apology(message, code=400):
    """Render message as an apology to user."""
    def escape(s):
        """
        Escape special characters.

        https://github.com/jacebrowning/memegen#special-characters
        """
        for old, new in [("-", "--"), (" ", "-"), ("_", "__"), ("?", "~q"),
                         ("%", "~p"), ("#", "~h"), ("/", "~s"), ("\"", "''")]:
            s = s.replace(old, new)
        return s
    return render_template("apology.html", top=code, bottom=escape(message)), code


# decorator initiation for login requirement (all pages aside from login/registration)
def login_required(f):
    """
    Decorate routes to require login.

    https://flask.palletsprojects.com/en/1.1.x/patterns/viewdecorators/
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function


# Function to load and convert image from a uint16 to uint8 datatype.
def normalize(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # pefroms mix-max normalization on the image array.
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    # conversion to unit8 (8-bits)
    img = img.astype(np.uint8)
    # checks if the image is grayscale and, if so, converts it to an RGB image by replicating the single-channel
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    return img


# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, segmented_image):
    alpha = 1.0  # Transparency for the original image.
    beta = 0.7  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return np.clip(image, 0.0, 1.0)
    
    
# also Inside model.py
def get_patient_images(case_prefix, day_prefix, folder_path, user_id):
    patient_images = []
    for filename in os.listdir(folder_path):
        # Check if the filename starts with the given case and day prefixes and ends with user_id
        if filename.startswith(f"case{case_prefix}_day{day_prefix}") and filename.endswith(f"_{user_id}.png"):
            patient_images.append(os.path.join(folder_path, filename))
    return patient_images

