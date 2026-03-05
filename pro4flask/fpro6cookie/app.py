from flask import Flask, render_template_string, request, make_response, redirect, url_for
# render_template_string : HTML 코드가 담긴 문자열을 Jinja2 엔진으로 렌더링
# request : 클라이언트가 보낸 HTTP 요청 데이터 보유
# make_response: 반환할 응답 객체를 직접 생성
# redirect : 사용자의 브라우저를 다른 주소로 강제 이동
# url_for : 함수의 이름을 기반으로 URL 경로를 생성
app = Flask(__name__)

# Cookie는 브라우저에 저장되는 작은 키-값 데이터이고, 서버가 클라이언트와 연결 유지하는 것 처럼 할 수 있음
# 서버가 설정 -> 브라우저가 저장 -> 다음 요청부터 자동으로 함께 전송

HOME_HTML = '''
<h2>Flask Cookie test</h2>
<form action='/set_cookie' method="post">
    쿠키 값 : <input type="text" name="name" placeholder="예:hong">
    <button type="submit">쿠키 저장</button>
</form>
<p>
    <a href="/read_cookie">쿠키 읽기</a>
    <a href="/delete_cookie">쿠키 삭제</a>
</p>
'''

@app.get('/')
def home():
    return render_template_string(HOME_HTML)

@app.post("/set_cookie")
def set_cookie():
    # 쿠키 저장
    name = request.form.get('name', "anonymous")

    # 클라이언트에 쿠키를 심으랴면 응답 객체가 필요
    # 먼저 "read_cookie" 페이지로 이동하라는 redirect 객체를 만들고 
    # 그 응답에 따라 쿠키를 추가한 뒤 브라우저에 돌려줌

    resp = make_response(redirect(url_for("read_cookie")))
    resp.set_cookie(    # 브라우저에 쿠키 저장
        key="name",     # 쿠키 이름
        value=name,     # 사용자가 입력한 쿠키 값
        max_age=60 * 5, # 쿠키 유효시간 - 5분뒤 만료(안주면 세션 유효기간)
        httponly=True,  # JS에서 document.cookie로 접근 불가
        samesite="Lax"  # CSRF 공격 방지용
    )
    return resp  # 쿠키가 포함된 응답을 브라우저로 반환
    # 브라우저는 쿠키를 저장하고, redirect 요청에 따라 read_cookie로 다시 요청함

@app.get("/read_cookie")  # url_for 이 없었으면 "/read_cookie" 불러옴
def read_cookie():         # url_for 때문에 함수를 불러옴.
    # 브라우저가 요청에 실어 보낸 모든 쿠키 중에서 내 서버가 만든 name 쿠키를 꺼냄
    # - 없으면 None을 반환 (첫방문/만료/삭제된 경우)
    name = request.cookies.get('name')

    # 읽은 쿠키 html로 출력하기
    return f"""
        <h3>쿠키 읽기</h3>
        <p>name 쿠키 값 : {name}</p>
        <a href='/'>홈페이지</a>
    """

@app.get("/delete_cookie")  
def delete_cookie(): 
    # 쿠키 삭제 후 홈(/)으로 이동하기 위한 redirect 응답을 만듦
    resp = make_response(redirect(url_for("home")))
    resp.delete_cookie("name")
    return resp

if __name__ == '__main__':
    app.run(debug=True)