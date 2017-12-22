# Web application code

import os
from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import skimage.io as io

# folders for our image
UPLOAD_FOLDER = os.getcwd() + '/static/images/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
MODEL_PATH = os.getcwd() + '/model/frozen_model.pb'


app = Flask(__name__)
wsgi_app = app.wsgi_app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# main function
@app.route('/', methods=['GET','POST'])
def main():
    if request.method == 'GET':
        return render_template('preview.html')
    elif request.method == 'POST':
        origin=request.files['origin'];
        originname=secure_filename(origin.filename)
        originpath=os.path.join(app.config['UPLOAD_FOLDER'], originname)
        origin.save(originpath)

        processedname = 'processed'+ originname
        processed_path = UPLOAD_FOLDER + processedname
        read_transfer_save(originpath,processed_path,model_path=MODEL_PATH,resize=True)

        return render_template('display.html', originname=originname, processed=processedname);
    else:
        return "<h2>Invalid</h2>";

# use model to do colorization
def read_transfer_save(img_path,save_path,model_path,resize=True):
    graph = load_graph(model_path)
    gray_img = graph.get_tensor_by_name('prefix/graph/in_holder:0')
    generate = graph.get_tensor_by_name('prefix/graph/sampler/generator/g_tanh:0')
    float_input = gray_img_read(img_path,resize=resize)
    
    with tf.Session(graph=graph) as sess:
        result = sess.run(generate,feed_dict={gray_img:float_input})
    img = result[0,:,:,:]
    shape = img.shape
    c_img = np.reshape(img,(shape[0],shape[1],-1))
    out_img = Image.fromarray((c_img * 255).astype(np.uint8))
    out_img.save(save_path)

# load our model
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

# load images  
def gray_img_read(path,resize=True):
    gray_read = Image.open(path)
    if resize:
        if gray_read.size[0] > gray_read.size[1]:
            gray_read = gray_read.resize((250,175),Image.ANTIALIAS)
        else:
            gray_read = gray_read.resize((175,250),Image.ANTIALIAS)
    
    gray_read = np.array(gray_read)
    shape = gray_read.shape
    gray_input = np.float32(gray_read)/255.0
    if len(shape) != 2:
        gray_float = np.reshape(gray_input[:,:,0],(1,shape[0],shape[1],1))
    else:
        gray_float = np.reshape(gray_input,(1,shape[0],shape[1],1))
    
    return gray_float


if __name__=='__main__':
	app.run(host=os.getenv('IP','0.0.0.0'),port=int(os.getenv('PORT',1033)))