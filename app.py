from flask import Flask,request,jsonify,render_template
import pandas as pd
import pickle


app=Flask(__name__)

@app.route("/predic",methods=["GET"])
def rend_frm():
    return render_template("index.html")


def clean_data(dt):
    print(type(dt["age"]))
    return { "gestation":[float(dt["gestation"])],
            "parity":[int(dt["parity"])],
            "age":[float(dt["age"])],
            "height":[float(dt["height"])],
            "weight":[float(dt["weight"])],
            "smoke":[float(dt["smoke"])]}


@app.route("/predict",methods=["POST"])
def pred_model():
    #data from use
    #dt=request.get_json() pehle data json tha
    dt=request.form  #ab form se leha
    #convert data fram
    dt_ext=clean_data(dt)
    print(dt_ext)
    # print(dt)
    df=pd.DataFrame(dt_ext)

    # load pickel file
    with open("model/model.pkl",'rb') as f:
        od=pickle.load(f)
    
    ans=od.predict(df)
    return render_template("index.html",resp=ans)

    # return "hello"





if __name__=='__main__':
    app.run(debug=True)