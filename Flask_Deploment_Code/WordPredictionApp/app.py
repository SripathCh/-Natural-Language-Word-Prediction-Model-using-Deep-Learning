from flask import Flask, render_template, request
import model


app = Flask(__name__)

 
@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/senddata', methods=['POST'])
def predictdata():
    predictions=[]
    input_text=request.form['textinp']
    input_text=input_text.strip().lower()
    encoded_text = model.tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = model.pad_sequences([encoded_text], maxlen=model.seq_len, truncating='pre')
    for i in (model.nextwmodel.predict(pad_encoded)[0]).argsort()[-3:][::-1]:
        pred_words = model.tokenizer.index_word[i]
        predictions.append(pred_words)
    return render_template("index.html", textent="The sequence entered is: {}".format(input_text),predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)