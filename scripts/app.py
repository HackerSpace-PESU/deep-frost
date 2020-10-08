from flask import Flask, render_template, request, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField
from wtforms.validators import InputRequired, Length, Email, EqualTo, ValidationError
from Poem_generator import generatePoem

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisisasecret'

class PoemForm(FlaskForm):
    scheme = StringField('Rhyme Scheme:',
                         validators = [
                             InputRequired(message='Field is required!'),
                             Length(min=4, message='scheme should be minimum of length 4')
                             ])
    starting = StringField('First Line:',
                           validators = [
                               InputRequired(message = 'Field is required!'),
                               Length(min=15, message='Input is too small')
                           ])
    
    def __init__(self, *args, **kwargs):
        super(PoemForm, self).__init__(*args, **kwargs)

@app.route('/', methods=['GET', 'POST'])
def home():
    form = PoemForm()
    poem=0
    if form.validate_on_submit():
        scheme = form.scheme.data
        starting = form.starting.data
        poem = generatePoem(scheme, starting).split('\n')
    return render_template('index.html', form = form, poem = poem)

if __name__ == '__main__':
    app.run(debug=True)
    