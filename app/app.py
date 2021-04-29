from flask import Flask, render_template, request, jsonify
import numpy as np

# load Model
import joblib
model = joblib.load('SmartPhone_model.sav')

app = Flask(__name__)


@app.route('/')
def home():
    # age = ["<20", "20-30", "31-40", "41-50", "51-60", ">60"]
    age = [["<20" , 1] , ["20-30", 2] , ["31-40", 3] , ["41-50",4] , ["51-60",5] , [">60",6]]
    sex = [["Male",1] , ["Female",2] , ["LGBTQ+",3]]
    avg_income = [["0",1], ["<5000",2], ["5,000-10,000",3], ["10,001-20,000",4], [">20,000",5]]
    behavior = [["Social Media",2], ["เล่นเกมส์",1], ["ดูหนัง",3], ["ฟังเพลง",8], ["ถ่ายภาพ",4],
                ["ถ่ายวิดีโอ",5], ["คุยโทรศัพท์",6], ["จดเลคเชอร์/วาดรูป",9], ["ใช้นำทาง(ส่งของ)",7]]
    price = [["15,000-19,999",1], ["20,000-40,000",2]]
    battery_1 = [["ต่ำกว่า 6 ชั่วโมง" , 1] , ["6-8 ชั่วโมง" ,2] , ["9-11 ชั่วโมง" ,3]]
    battery_2 = [["ต่ำกว่า 3 ชั่วโมง" , 1] , ["4-6 ชั่วโมง" ,2] , ["7-9 ชั่วโมง" ,3]]
    battery_3 = [["30 นาที - 1 ชั่วโมง" , 1] , ["1 ชั่วโมง 1 นาที  - 1 ชั่วโมง 31 นาที " ,2] , ["1 ชั่วโมง 31 นาที  - 2 ชั่วโมง" ,3]]

    return render_template('index.html', age=age, sex=sex, avg_income=avg_income, behavior=behavior, price=price , battery_1= battery_1 ,  battery_2= battery_2 , battery_3= battery_3)


@app.route('/predict',  methods=["POST"])
def predict():
    # return "hello World"
    # เอามาใส่ไว้ท้ายเพื่อให้ column ตรงกับ model
    behavior1 = request.form['behavior_1']
    behavior2 = request.form['behavior_2']
    behavior3 = request.form['behavior_3']
    #แปลง input จาก form เป็น dict
    user_input= request.form.to_dict()
    user_input.pop("behavior_1")
    user_input.pop("behavior_2")
    user_input.pop("behavior_3")
    print("User input" , user_input)

    # แปลง dict ให้เป็น list
    user_list = []
    for key in user_input:
        user_list.append(user_input[key])
    
    user_list.append(behavior1)
    user_list.append(behavior2)
    user_list.append(behavior3)
    print("userlist",user_list)

    np_data = np.array(user_list).reshape(1,28)					
    y_predict = model.predict(np_data)	
    output = y_predict[0]
    return render_template('result.html' , output=output)

