import os
from flask import Flask, render_template, request,redirect, url_for,jsonify, make_response,send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_script import Manager
from werkzeug import secure_filename
from zipfile import ZipFile

from fonduer_py import config,parser, mention_extractor,candidate_extractor,\
    custom_filter_candidate,featurizer,utils,labeller,model

basedir = os.path.abspath(os.path.dirname(__file__))

# configuring database
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['TRAIN_FOLDER'] = config.train_docs_path
app.config['PREDICT_FOLDER'] = config.predict_docs_path
app.config['FIRST_TIME_TRAIN'] = True
app.config['FIRST_TIME_TRAIN_MODEL'] = True
app.config['FIRST_TIME_PREDICT'] = True


app_data = None
prediction_data = None

# Home Page Route
@app.route('/')
@app.route('/index')
def login():
    return render_template('index.html')

# Route to render view for step 1
@app.route('/step1')
def view():
    return render_template('upload_train.html',data=None)

# Route to handle the file upload for training
@app.route('/uploader',methods = ['POST'])
def upload_train_docs():
    try:
        if request.method == 'POST':
            for f in  request.files.getlist('file'):                    
                if str(f.filename).lower().endswith(('.zip')):
                    f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
                    with ZipFile(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)), 'r') as zipObj:
                        # Extract all the contents of zip file to input_files directory
                        zipObj.extractall(app.config['TRAIN_FOLDER'])
                
                elif str(f.filename).lower().endswith(('.html','.htm')):
                    f.save(os.path.join(app.config['TRAIN_FOLDER'],secure_filename(f.filename)))
                
                else:
                    raise Exception                
                
                
            return render_template('upload_train.html',data={'message':"Upload Successful",'status_code':200})            
    
    except Exception:
        return render_template('upload_train.html', data={'message':"Upload Failed",'status_code':400})

# Route to initiate the parsing of training documents
@app.route('/beginParsing')
def begin_parser():
    global app_data
    # Checking if the train folder is empty
    if is_empty_folder(app.config['TRAIN_FOLDER']):
        return render_template('upload_train.html', data={'message':"Please upload files",'status_code':400})

    parser_output = parser.parse_dataset(train=True,first_time=app.config['FIRST_TIME_TRAIN'])
    mention_extractor_output = mention_extractor.get_mentions(parser_output,first_time=app.config['FIRST_TIME_TRAIN'])
    candidate_extractor_output = candidate_extractor.get_candidates(mention_extractor_output, first_time=app.config['FIRST_TIME_TRAIN'])
    candidate_filtered_output = custom_filter_candidate.filter(candidate_extractor_output,first_time=app.config['FIRST_TIME_TRAIN'])
    featurizer_output = featurizer.get_features(candidate_filtered_output,train=True,first_time=app.config['FIRST_TIME_TRAIN']) 
    # utils.load_section_heading_gold_labels(featurizer_output, annotator_name='gold') 
    labeller_output = labeller.apply_labellling_functions(featurizer_output)


    app_data = labeller_output
    return render_template('train_parse_results.html',data=labeller_output)

# Route to render the view for step 2
@app.route('/step2')
def view_candidates():
    global app_data
    labeller_output = app_data
    if labeller_output == None:
        return redirect('beginParsing')

    candidate_data = []
    session = labeller_output['session']
    candidate_list = session.query(labeller_output['candidate_variable'][0]).all()
    
    for candidate in candidate_list :
        for mention in candidate.get_mentions():
            document = mention.document
            for span in mention:
                sentence = span.get_span()
                candidate_data.append(tuple([document,sentence]))
        
    
    return render_template('view_candidates.html',data=candidate_data)

# Route to render the view for step 3
@app.route('/step3')
def view_train_model():
    global app_data
    labeller_output = app_data
    if labeller_output == None:
        return redirect('beginParsing')
    else:
        return render_template('train_model.html',data=None)

# Route to initiate the training process
@app.route('/beginTraining',methods=['POST'])
def begin_training():
    global app_data
    labeller_output = app_data
    if labeller_output == None:
        return redirect('beginParsing')
    else:
        algorithm_chosen = request.form["algorithm_choice"]
        app_data['algorithm_chosen'] = algorithm_chosen
        (msg,status_code) = model.train_model(algorithm_chosen,labeller_output,first_time=app.config['FIRST_TIME_TRAIN_MODEL'])
    return render_template('train_model.html',data={'message':msg,'status_code':status_code})

# Route to render the view for step 4
@app.route('/step4')
def view_upload_test():
    return render_template('upload_predict.html',data=None)

# Route to handle the file upload for prediction
@app.route('/uploader_prediction',methods = ['POST'])
def upload_prediction_docs():
    try:
        if request.method == 'POST':
            for f in  request.files.getlist('file'):                    
                if str(f.filename).lower().endswith(('.zip')):
                    f.save(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)))
                    with ZipFile(os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename)), 'r') as zipObj:
                        # Extract all the contents of zip file to input_files directory
                        zipObj.extractall(app.config['PREDICT_FOLDER'])
                
                elif str(f.filename).lower().endswith(('.html','.htm')):
                    f.save(os.path.join(app.config['PREDICT_FOLDER'],secure_filename(f.filename)))
                
                else:
                    raise Exception                
                
                
            return render_template('upload_predict.html', data={'message':"Upload Successful",'status_code':200} )            
    
    except Exception:
        return render_template('upload_predict.html', data={'message':"Upload Failed",'status_code':400})        

# Route to initiate parsing of files for prediction
@app.route('/beginParsingForPrediction')
def begin_parser_for_prediction():
    global prediction_data

    # Checking if the predict folder is empty
    if is_empty_folder(app.config['PREDICT_FOLDER']):
        return render_template('upload_predict.html', data={'message':"Please upload files",'status_code':400})
    
    parser_output = parser.parse_dataset(train=False,first_time=app.config['FIRST_TIME_PREDICT'])
    mention_extractor_output = mention_extractor.get_mentions(parser_output,first_time=app.config['FIRST_TIME_PREDICT'])
    candidate_extractor_output = candidate_extractor.get_candidates(mention_extractor_output, first_time=app.config['FIRST_TIME_PREDICT'])
    candidate_filtered_output = custom_filter_candidate.filter(candidate_extractor_output,first_time=app.config['FIRST_TIME_PREDICT'])
    featurizer_output = featurizer.get_features(candidate_filtered_output,train=False,first_time=app.config['FIRST_TIME_PREDICT']) 

    prediction_data = featurizer_output
    
    return render_template('predict_parse_results.html',data=featurizer_output)


# Route to render the view for step 5
@app.route('/step5')
def view_predicted_candidates():
    global prediction_data
    global app_data
    candidate_data = []
    if app_data is None:
        return redirect('step1')
    if 'algorithm_chosen' not in app_data:
        return redirect('step3')

    if prediction_data == None:
        return redirect('beginParsingForPrediction')
    
    candidate_list = model.load_model_and_predict(app_data['algorithm_chosen'],prediction_data)
    
    for candidate in candidate_list :
        for mention in candidate.get_mentions():
            document = mention.document
            for span in mention:
                sentence = span.get_span()
                candidate_data.append(tuple([document,sentence]))        

    return render_template('view_results.html',data=candidate_data)


@app.route('/clearTrain')
def delete_train_files():
    try:
        filelist = [ f for f in os.listdir(app.config['TRAIN_FOLDER']) ]        
        for f in filelist:
            os.remove(os.path.join(app.config['TRAIN_FOLDER'], f))

        return make_response(jsonify({'message':'Deleted all uploaded files','status_code':200}),200)
    
    except Exception as e:
        return make_response(jsonify({'message':'Something went wrong','status_code':400}),400)

@app.route('/clearPredict')
def delete_predict_files():
    try:
        filelist = [ f for f in os.listdir(app.config['PREDICT_FOLDER']) ]        
        for f in filelist:
            os.remove(os.path.join(app.config['PREDICT_FOLDER'], f))

        return make_response(jsonify({'message':'Deleted all uploaded files','status_code':200}),200)
    
    except Exception as e:
        return make_response(jsonify({'message':'Something went wrong','status_code':400}),400)
        
def is_empty_folder(path):
    filelist = [ f for f in os.listdir(path) ]   
    if len(filelist) > 0:
        return False
    else:
        return True

@app.route('/load/<path:filename>')
def load_html(filename):
    return send_from_directory(app.config['PREDICT_FOLDER'],filename)


# main calling function
if __name__ == "__main__":
    app.run(host="10.100.97.176")    