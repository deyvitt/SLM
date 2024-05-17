"""Before start, please make sure your server has installed the followings:

You can install them all at once:
pip install flask flask_wtf werkzeug flask_login eth_account web3 flask_limiter

or you can install them one by one:
pip install flask
pip install flask_wtf
pip install werkzeug
pip install flask_login
pip install eth_account
pip install web3
pip install flask_limiter"""

import json
import os
from flask import send_from_directory
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField
from werkzeug.utils import secure_filename
from werkzeug.datastructures import CombinedMultiDict
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
"""from eth_account.messages import encode_defunct
from web3 import Web3"""
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

class UploadForm(FlaskForm):
    file = FileField('Dataset')

class User(UserMixin):
    def __init__(self, id):
        self.id = id

# These are codes if you want to use Web3/Blockchain as authentication
"""class MyFlaskApp:
    def __init__(self):
        self.app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
        self.app.config['UPLOAD_FOLDER'] = '/path/to/upload/folder'

        self.setup_routes()

        # Connect to Ethereum
        self.w3 = self.connect_to_blockchain(os.getenv('ETHEREUM_URL'))

        # Connect to Polygon
        self.w3_polygon = self.connect_to_blockchain(os.getenv('POLYGON_URL'))

        self.setup_routes()

    def connect_to_blockchain(self, url):
        try:
            w3 = Web3(Web3.HTTPProvider(url))
            if not w3.isConnected():
                print(f"Failed to connect to {url}")
                return None
            return w3
        except Exception as e:
            print(f"Error connecting to {url}: {e}")
            return None"""

app = Flask(__name__)
login_manager = LoginManager()
login_manager.init_app(app)
limiter = Limiter(app, key_func=get_remote_address)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', '/default/path/to/upload/directory')
@staticmethod
def allowed_file(filename):    
    ALLOWED_EXTENSIONS = {'TXT', 'PDF', 'CSV', 'JPEG', 'PNG', 'GIF', 'BMP', 'TIFF', 'MP3', 'WAV', 'AAC', 'MP4', 'AVI', 'WebM', 'MOV', 'WMV', 'MXF'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_user(user_id):
    return User(user_id)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def get_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return jsonify(config)

def upload_file():
    form = UploadForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST' and form.validate():
        f = form.file.data
        filename = secure_filename(f.filename)
        if allowed_file(filename):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('admin'))
    return render_template('upload.html', form=form)

def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("5/minute")  # limit to 5 attempts per minute
def login():
    if request.method == 'POST':
        try:

            username = request.form.get('username')
            password = request.form.get('password')
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for('admin'))
            else:
                return "Invalid username or password", 400
        except Exception as e:
            app.logger.error(e)
            return "An error occurred", 400
    return render_template('login.html')

#         These are codes when we want to use Web3/blockchain        
#            address = request.form.get('address')
#            signature = request.form.get('signature')
#            message = "Please log in to the SLM Admin Panel"
#            message_encoded = encode_defunct(text=message)
#            signer = Web3().eth.account.recover_message(message_encoded, signature=signature)
#            if signer.lower() == address.lower():
#                user = User(address)
#                login_user(user)
#                return redirect(url_for('admin'))
#            else:
#                return "Invalid signature", 400
#        except Exception as e:
#            app.logger.error(e)
#            return "An error occurred", 400
#    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin():
    # Here you can add code to interact with the database and manage the models
    return render_template('admin.html')

@app.route('/get_config', methods=['GET'])
@login_required
def get_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    return jsonify(config)

@app.route('/update_config', methods=['POST'])
@login_required
def update_config():
    new_config = request.get_json()
    if isinstance(new_config, dict):  # simple validation
        with open('config.json', 'w') as f:
            json.dump(new_config, f)
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'failed', 'message': 'Invalid input'}), 400

if __name__ == "__main__":
#  These are ending codes if you want to use Web3/Blockchain
#    my_flask_app = MyFlaskApp()
#    my_flask_app.
    app.run(debug=True)