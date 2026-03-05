from flask import Flask, render_template, request, session, redirect, url_for
# 파이썬 세션(session)은 웹에서 사용자 정보를 서버에 저장하는 기능을 말함(쿠키를 통해 세션 운영)
# 일정 시간동안 동일 사용자(브라우저)와 일련의 요청을 하나의 상태로 보고 그 상태를 유지시키는 기술
# 쿠키에 비해 상대적으로 안전함

# 실습 : 사용자가 os를 선택하면 세션에 저장하고 읽기
from datetime import timedelta    # 날짜/시간을 연산해서 기간 설정 하기에 효과적

app = Flask(__name__)

# Flask는 세션 사용을 위해 secret_key 설정이 필요
app.secret_key = "abcdef123456"   # 위조 방지용 비밀키 값

# 참고 키 값 자동 생성 터미널창에서 > python -c "import secrets; print(secrets.token_hex(32))"
# e73ea80197d775e713352cb55758d65ba0a7235a2082ea84cced4f4a3e10fd53

app.permanent_session_lifetime = timedelta(seconds=5)  # 세션 만료 시간 5초 설정(기본값 30분)

@app.get('/')
def home():
    return render_template("main.html")

@app.route('/setos')
def setos():
    favorite_os = request.args.get('favorite_os')

    if favorite_os:
        session.permanent = True  # 세션 만료 시간 적용
        session["f_os"] = favorite_os   # "f_os" 키로 특정값 저장
        return redirect(url_for("showos"))
    else:
        return render_template('setos.html')
    
@app.route('/showos')
def showos():
    context = {} 

    if "f_os" in session:
        context["f_os"] = session["f_os"]
        context["message"] = f"당신이 선택한 운영체제는 '{session['f_os']}'"
    else:
        context["f_os"] = None
        context["message"] = "운영체재를 선택하지 않았거나 세션이 만료됨"

    return render_template("showos.html", context=context)

if __name__ == '__main__':
    app.run(debug=True)