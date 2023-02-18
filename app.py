from flask import Flask, request
import pandas as pd
from pmdarima import auto_arima

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello world'

@app.route('/', methods=['POST'])
def predict():
    if(request.data):
        data = request.get_json()

        START_DATE = data['start_date']
        END_DATE = data['end_date']
        DATE_RANGE = pd.date_range(START_DATE, END_DATE)

        transactions = pd.DataFrame(data['transactions'], columns=['date', 'transactions'])
        transactions = pd.DataFrame(data['transactions'], columns=['date', 'transactions'])
        transactions['date'] = pd.to_datetime(transactions['date'])
        transactions = transactions[transactions['date'] >= START_DATE][transactions['date'] <= END_DATE]
        transactions.set_index('date', inplace=True)

        empty_rows = pd.DataFrame(0,index=DATE_RANGE, columns=['transactions'])
        empty_rows.loc[transactions.index] = transactions

        model = auto_arima(y=transactions, m=7)
        preds = pd.Series(model.predict(n_periods=data['predictions']))
        preds.name = 'predictions'
        return preds.to_json(orient='split', date_format='iso'), 200
    else:
        return "Please provide data", 400

if __name__ == '__main__':
    app.run(port=3000, debug=True)