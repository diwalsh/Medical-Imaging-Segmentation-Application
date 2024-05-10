import cv2
import os
import zipfile
import numpy as np
from flask import redirect, render_template, session
from functools import wraps


# display funny apology to user upon login/registration failure
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


# Function to create the database and tables if they don't exist
def create_database():
    if not os.path.exists('dats.db'):
        # Connect to the database
        conn = sqlite3.connect('dats.db')
        cursor = conn.cursor()

        # Create the tables
        cursor.execute('''CREATE TABLE users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            username TEXT NOT NULL,
                            hash TEXT NOT NULL
                        )''')

        cursor.execute('''CREATE TABLE renderings (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER,
                            case_name TEXT,
                            case_number INTEGER,
                            day_number INTEGER,
                            cover_image TEXT,
                            mask_paths TEXT,
                            obj_path TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users(id)
                        )''')

        # Commit changes and close the connection
        conn.commit()
        conn.close()

        print("Database 'dats.db' created successfully.")
    else:
        print("Database 'dats.db' already exists.")


def zip_filenames(zip):
    file_like_object = zip.stream._file  
    zipfile_ob = zipfile.ZipFile(file_like_object)
    file_names = zipfile_ob.namelist()
    # Filter names to only include the filetype that you want:
    file_names = [file_name for file_name in file_names if file_name.endswith(".png")]
    return file_names, zipfile_ob

# grabbing relevant data from file name to format for user display
def format_name(name):
    parts = name.split("/")[-1].split("_")
    case_number = ''.join(filter(str.isdigit, parts[0]))
    day_number = ''.join(filter(str.isdigit, parts[1]))
    slice_number = int(parts[3])
    formatted_name = "Case {}; Day {}: Slice {}".format(case_number, day_number, slice_number)
    return formatted_name, case_number, day_number, slice_number

# Define generate_title function
def generate_title_slice(image_path):
    parts = image_path.split("/")[-1].split("_")
    case_number = ''.join(filter(str.isdigit, parts[0]))
    day_number = ''.join(filter(str.isdigit, parts[1]))
    slice_number = int(parts[3].split(".")[0])
    formatted_name = "Case {}; Day {}: Slice {}".format(case_number, day_number, slice_number)
    return formatted_name
                


# load and convert image from a uint16 to uint8 datatype.
def normalize(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # pefroms mix-max normalization on the image array.
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    # conversion to unit8 (8-bits)
    img = img.astype(np.uint8)
    # checks if the image is grayscale and, if so, converts it to an RGB image by replicating the single-channel
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    return img
    
    
# also inside model.py...
def get_patient_images(case_prefix, day_prefix, folder_path, user_id):
    patient_images = []
    for filename in os.listdir(folder_path):
        # Check if the filename starts with the given case and day prefixes and ends with user_id
        if filename.startswith(f"case{case_prefix}_day{day_prefix}") and filename.endswith(f"_{user_id}.png"):
            patient_images.append(os.path.join(folder_path, filename))
    return patient_images


