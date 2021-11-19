from flask import Flask,render_template,request,send_file,send_from_directory,jsonify

import pickle
import numpy as np

app = Flask(__name__,static_folder='static',template_folder='templates')

model=pickle.load(open('model_training/pose_classifier.pkl','rb+'))


@app.route('/',methods=['POST','GET'])
def main():
    if request.method=='GET':
        return render_template('main.html')
    if request.method=='POST':
        inputs=[x for x in request.form.values()]
        inputs_parse=list(inputs)
        inputs_parse_float=[float(x) for x in inputs_parse]
        result=model.predict([inputs_parse_float])
        print(result)
        return result[0]
        
    

if __name__=='__main__':
    app.run()
