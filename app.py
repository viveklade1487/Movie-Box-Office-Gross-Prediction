from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

scale = pickle.load(open(r'C:/Users/singh/Desktop/ADS Project/scaler.pkl', 'rb'))
model = pickle.load(open(r'C:/Users/singh/Desktop/ADS Project/model.pkl', 'rb'))

@app.route("/")
def initial():
    return render_template("index.html")

@app.route('/result', methods = ['POST'])
def login():
    budg = request.form["budg"]
    genr = request.form["genr"]
    home = request.form["home"]
    lang = request.form["lang"]
    popu = request.form["popu"]
    runt = request.form["runt"]
    stat = request.form["stat"]
    avot = request.form["avot"]
    tvot = request.form["tvot"]
    wday = request.form["wday"]
    mont = request.form["mont"]
    year = request.form["year"]
    t = [[np.cbrt(float(budg)), float(genr), float(home), float(lang), np.cbrt(float(popu)), float(runt), float(stat), float(avot), np.cbrt(float(tvot)), float(wday), float(mont), float(year)]]
    t = scale.transform(t)
    output = model.predict(t)
    output = np.power(output,3)
    return render_template("result.html", y = str(output[0]))

if __name__ == '__main__':
    app.run(debug = False)
