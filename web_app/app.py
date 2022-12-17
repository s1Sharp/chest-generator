# coding=utf-8
import sys
import os

# Toch utils
import torch

# Control functions
from controller.generator import Generator, load_generator_from_file
from controller.image_generator import generate_image, generate_image_zip, normalize_seed
from controller.image_generator import GENERATED_PATH

# Flask utils
from flask import Flask, request, render_template, send_file, send_from_directory, url_for, make_response, jsonify
from flask_bootstrap import Bootstrap5


APP_PATH = 'web_app'
app = Flask(__name__)
bootstrap = Bootstrap5(app)

generator = load_generator_from_file(f"{APP_PATH}/models/generator.pth")

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/', methods=['GET', 'POST'])
def index():
    # handle the POST request
    context = ''
    if request.method == 'POST':

        image_seed = int(request.form.get('image_seed'))
        torch_random_generator = torch.random.manual_seed(normalize_seed(image_seed))
        image_size = int(request.form.get('image_size'))
        image_type = str(request.form.get('image_type'))

        if image_size > 1 and image_size < 999:
            zipfilename = generate_image_zip(APP_PATH, generator, torch_random_generator, int(image_seed), int(image_size), str(image_type))
            download_url = f"{url_for('index', _external=True)}downloads/{zipfilename}"
            context = {'size' : image_size, 'format' : image_type, 'url' : download_url}
        elif image_size == 1:
            filename = generate_image(APP_PATH, generator, torch_random_generator, int(image_seed), int(image_size), str(image_type))
            download_url = f"{url_for('index', _external=True)}downloads/{filename}"
            context = {'size' : image_size, 'format' : image_type, 'url' : download_url}
        else:
            context = 'Error'
    return render_template('index.html', context=context)


@app.route('/downloads/<path:filename>', methods=['GET'])
def download(filename):
    uploads = filename
    return send_file(uploads, as_attachment=True)

@app.route('/hello', methods=['GET'])
def helloworld():
    if(request.method == 'GET'):
        data = {"data": "Hello World"}
        res = jsonify(data)
        res.headers.add('Access-Control-Allow-Origin', '*')
        return res

def main(argv):
    from controller.clear_old import init_delete_sheduler_job
    init_delete_sheduler_job(os.path.join(APP_PATH, GENERATED_PATH))
    if "release" in argv:
        from waitress import serve
        serve(app, host="0.0.0.0", port=5000)
    else:
        app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main(sys.argv)
