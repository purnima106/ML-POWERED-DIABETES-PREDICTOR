import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session
from flask import * 
import pickle
from sqlite3 import connect
from hashlib import md5
from flask_mail import Mail, Message

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
app.secret_key = "purnima_stays_in_virar"

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'minnie5aj143@gmail.com'
app.config['MAIL_PASSWORD'] = 'zela mxtb ekpg dhvq'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:, [1, 4, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)


@app.route("/", methods=["GET", "POST"])
def home():
    if "un" not in session:
        return redirect(url_for("login"))
    if request.form.get("logout") or request.args.get("logout"):
        session.pop("un")
        return redirect(url_for("login"))
    return render_template('home.html')

@app.route("/map")
def map():
    return render_template("map.html")

@app.route("/tip")
def tip():
    return render_template("tips.html")

@app.route("/appoint", methods=["GET", "POST"])
def appoint():
    approved_patients = ["Mudra", "Purnima", "Payal"]  # Update with actual patient usernames
    
    if "un" not in session:
        return redirect(url_for("home"))

    elif request.method == "POST":
        name = request.form.get("name")
        date = request.form.get("date")
        time = request.form.get("time")
        message = request.form.get("message")

        # Check if all required fields are provided
        if name and date and time and message:
            print("Name entered:", name)  # Debugging statement
            if name in approved_patients:
                try:
                    con = connect("pcusers.db")
                    cursor = con.cursor()
                    cursor.execute("INSERT INTO doctor (name, date, time, message) VALUES (?, ?, ?, ?)", (name, date, time, message))
                    con.commit()
                    return "Thank you! We will reach out to you soon."
                except Exception as e:
                    print("Error:", e)
                    return "An error occurred while processing your request. Please try again later."
                finally:
                    if con:
                        con.close()
            else:
                return "You are not an approved patient."
        else:
            return "Please provide all required information."
    else:
        return render_template("appoint.html")

	

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))

    if prediction == 1:
        pred = "You have Diabetes, please consult a Doctor."
    elif prediction == 0:
        pred = "You don't have Diabetes."
    output = pred
    return render_template('home.html', prediction_text='{}'.format(output))


@app.route("/login", methods=["GET", "POST"])
def login():
    if "un" in session:
        return redirect(url_for("home"))
    elif request.method == "POST":
        un = request.form["un"]
        pw = request.form["pw"]
        epw = md5(pw.encode()).hexdigest()
        con = None
        try:
            con = connect("pcusers.db")
            cursor = con.cursor()
            sql = "SELECT * FROM student WHERE username = '%s' AND password = '%s'"
            cursor.execute(sql % (un, epw))
            data = cursor.fetchall()
            if len(data) == 0:
                msg = "Invalid login"
                return render_template("login.html", msg=msg)
            else:
                session["un"] = un
                return redirect(url_for('home'))
        except Exception as e:
            msg = "Issue: " + str(e)
            return render_template("login.html", msg=msg)
        finally:
            if con is not None:
                con.close()
    else:
        return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if "un" in session:
        return redirect(url_for("home"))
    elif request.method == "POST":
        un = request.form["un"]
        pw1 = request.form["pw1"]
        pw2 = request.form["pw2"]
        email = request.form["email"]
        epw = md5(pw1.encode()).hexdigest()
        if pw1 == pw2:
            con = None
            try:
                con = connect("pcusers.db")
                cursor = con.cursor()
                sql = "INSERT INTO student VALUES ('%s', '%s')"
                cursor.execute(sql % (un, epw))
                con.commit()
                session["un"] = un  # Adding user to session after successful signup
                with app.app_context():
                    message = Message(subject='Welcome to Diabetes App',
                                      sender='minnie5aj143@gmail.com',
                                      recipients=[email])
                    message.body = 'Thank You for signing up to our Diabetes Prediction App, your health buddy for quick and insightful predictions in just a few seconds!!'
                    mail.send(message)
                return redirect(url_for("home"))  # Redirect to home page after successful signup
            except Exception as e:
                con.rollback()
                msg = "Thanks for signup " 
                return render_template("signup.html", msg=msg)
            finally:
                if con is not None:
                    con.close()
        else:
            msg = "Passwords did not match"
            return render_template("signup.html", msg=msg)  # Render signup.html when passwords don't match
    else:
        return render_template("signup.html")


@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for("login"))

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("un", None)  # Remove the "un" key from the session
    return redirect(url_for("login"))  # Redirect to the login page


if __name__ == "__main__":
    app.run(debug=True)
