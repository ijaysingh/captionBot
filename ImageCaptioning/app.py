from asyncio import run_coroutine_threadsafe
# from crypt import methods
from flask import Flask, render_template, redirect, request
import captionIt


# __name__ == __main__
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template("index.html" )

@app.route('/model')
def hellow():
    return render_template("model.html" )

@app.route('/', methods=['POST'])
def marks():
    if request.method == 'POST':
        f = request.files['userfile']
        
        path = "./static/images/userInput/{}".format(f.filename) # 'http://127.0.0.1:5500/static/images/userinput/images.jpg'
        f.save(path)
        
        caption = captionIt.captionTheImg(path)
        result = {
            'image' : path,
            'caption' : caption
        }
        return render_template("model.html", yourResult=result)
    else:
        result=None
        return render_template("model.html", yourResult=result)





if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True)
