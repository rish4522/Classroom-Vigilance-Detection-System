from flask import Flask, Response, jsonify, request, session, render_template

from flask_wtf import FlaskForm

from wtforms import FileField, SubmitField, StringField, DecimalRangeField, IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired, NumberRange
import os

import cv2

from YOLOv8_webCam import video_detection

app = Flask(__name__)

app.config['SECRET KEY'] = 'Kusumpathi@121'
app.secret_key = '@alienforce'
app.config['UPLOAD_FOLDER'] = 'static/files'


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    conf_slide = IntegerRangeField('Confidence: ', default=25, validators=[InputRequired()])
    submit = SubmitField("Run")


def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        if detection_ is not None and not detection_.empty():  # Check if frame is not empty
            ref, buffer = cv2.imencode('.jpg', detection_)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



def generate_frames_web(path_x):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    session.clear()
    return render_template('index.html')


@app.route('/webcam', methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('ui.html')


@app.route('/frontpage', methods=['GET', 'POST'])
def front():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                               secure_filename(file.filename)))
        session['video_path'] = os.path.join(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'],
                         secure_filename(file.filename)))

    return render_template('videoproject.html', form=form)


@app.route('/video')
def video():
    return Response(generate_frames(path_x=session.get('video_path', None)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/webapp')
def webapp():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
