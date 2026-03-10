from flask import Flask, render_template, request
import cv2
import os
import numpy as np
from filters import *

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():

    results = None

    if request.method == "POST":

        file = request.files["image"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        img = cv2.imread(filepath)

        # thêm nhiễu
        noisy = add_salt_pepper_noise(img)

        # xử lý ảnh xám hoặc màu
        if len(img.shape) == 2:

            median = median_filter(noisy)
            pseudo = pseudo_median_filter(noisy)
            combined = combined_filter(noisy)

        else:

            median = process_color(noisy, median_filter)
            pseudo = process_color(noisy, pseudo_median_filter)
            combined = process_color(noisy, combined_filter)

        # lưu ảnh
        cv2.imwrite("static/noisy.jpg", noisy)
        cv2.imwrite("static/median.jpg", median)
        cv2.imwrite("static/pseudo.jpg", pseudo)
        cv2.imwrite("static/combined.jpg", combined)

        # tính MSE
        mse_median = mse(img, median)
        mse_pseudo = mse(img, pseudo)
        mse_combined = mse(img, combined)

        # tính PSNR
        psnr_median = psnr(img, median)
        psnr_pseudo = psnr(img, pseudo)
        psnr_combined = psnr(img, combined)

        results = {
            "original": filepath,
            "mse_median": mse_median,
            "mse_pseudo": mse_pseudo,
            "mse_combined": mse_combined,
            "psnr_median": psnr_median,
            "psnr_pseudo": psnr_pseudo,
            "psnr_combined": psnr_combined
        }

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)