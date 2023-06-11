import pickle

from flask import Flask, render_template
from flask import request, url_for, flash, redirect

from . import payment

# App
app = Flask(__name__)
app.config['SECRET_KEY'] = '1234'

# init value
session = {'month': 1, 'curr_usage': 300, 'capacity': 3}

# main
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Get values
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        _month = request.form['month']
        _usage = request.form['usage']
        _capacity = request.form['capacity']

        if not _month:
            flash('월을 입력하세요')
        elif not _usage:
            flash('월간 사용량을 입력하세요')
        elif not _capacity:
            flash('발전용량을 입력하세요')

        try:
            _month = int(_month)
            _usage = float(_usage)
            _capacity = float(_capacity)
        except:
            flash('숫자로 입력하세요')

        if _capacity > 3.3:
            flash('가정용 태양광 발전기는 최대 3.3kW 까지만 지원 가능합니다.')

        else:
            session['month'] = _month
            session['curr_usage'] = _usage
            session['capacity'] = _capacity

            return redirect(url_for('result'))
    return render_template('predict.html')

# Show results
@app.route('/result/', methods=['GET'])
def result():
    # Get model
    with open('./flask_app/model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    # predict
    month = session['month']
    curr_usage = session['curr_usage']
    collected = model.predict(steps=month)[-1]
    generated_power = collected/3600*1000*(session['capacity']*1000/300*2)*0.15

    reduced_power = curr_usage - generated_power

    current_payment = payment.calc_payment(month, curr_usage)
    reduced_payment = payment.calc_payment(month, reduced_power)

    if reduced_power < 0:
        saved_money = str(current_payment - 1150)[:-2]
        reduced_payment = '1150.0'
    else:
        saved_money = str(current_payment - reduced_payment)[:-2]
    
    return render_template('result.html',
        month=12 if month%12==0 else month%12,
        curr_usage=curr_usage,
        generated_power=str(generated_power).split('.')[0],
        current_payment=str(current_payment)[:-2],
        reduced_payment=str(reduced_payment)[:-2],
        saved_money=saved_money)