# Flaskのインポート
from flask import Flask,render_template,send_from_directory,url_for

import os

# predict用pythonファイルの読み込み
import melody_predict
import accomp_predict

#Flaskオブジェクトの生成
app = Flask(__name__, static_folder="static")



# Blueprintを読み込む
# import tmp
# app.register_blueprint(tmp.app)

# CSSのキャッシュ削除
@app.context_processor
def add_staticfile():
    def staticfile_cp(fname):
        path = os.path.join(app.root_path, 'static', fname)
        mtime =  str(int(os.stat(path).st_mtime))
        return '/static/' + fname + '?v=' + str(mtime)
    return dict(staticfile=staticfile_cp)



#「/」へアクセスがあった場合に、「index.html」を返す
@app.route("/")
def index():
    return render_template("index.html")

# オーディオ、画像ファイルを正しく読み込む
@app.route("/audio/<path:filename>")
def audio(filename):
    return send_from_directory("templates/audio", filename)
@app.route("/img/<path:filename>")
def img(filename):
    return send_from_directory("templates/img", filename)

# 「作ってみる」を選択すると、YOASOBIっぽい曲を生成して返す
@app.route("/yoasobi")
def yoasobi():
    melody_predict.melody()
    accomp_predict.accomp()
    return render_template("index.html")

# おまじない
if __name__ == "__main__":
    app.run(debug=True)