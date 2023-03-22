from flask import Flask, request, render_template, jsonify
from model import SentimentRecommenderModel

app = Flask(__name__)

sentiment_rec_model = SentimentRecommenderModel()


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_recommendations():
    # get users_data from html
    users_data = request.form['userName']
    users_data = users_data.lower()
    # obtaining list of recommended products
    items = sentiment_rec_model.getSentimentBasedRecommendations(users_data)
    if not(items is None):
        print(f"retrieving items.Number of items...{len(items)}")
        print('*'*10)
        print(items)
        print('*' * 10)
        return render_template("index.html", column_names=items.columns.values, row_data=list(items.values.tolist()),
                               zip=zip)
    else:
        return render_template("index.html",
                               message=f"{users_data} is not available in database, recommendations can't be given "
                                       "try suggested user name!")


if __name__ == '__main__':
    print('*** App Started ***')
    app.run(debug=True)
