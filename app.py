from flask import Flask, redirect, render_template, request

import caption_it

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
	return render_template("index.html")

@app.route('/', methods = ['POST'])
def caption_image():
	if request.method == 'POST':

		f = request.files['userfile']
		path = './static/{}'.format(f.filename)
		f.save(path)
		print(path)

		caption = caption_it.caption_this_image(path)

		result_dic = {
			'image' : path,
			'caption': caption
		}

		print(result_dic['caption'])

	return render_template("index.html", your_result = result_dic)


if __name__ == '__main__':
	app.run(debug = True)
