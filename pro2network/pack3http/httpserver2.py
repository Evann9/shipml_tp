# CGIHTTPRequestHandler : SimpleHTTPRequestHandler의 확장 클래스
# get, post 모두 지원 가능
# CGI(Common Gateway Interface): 웹서버와 외부 프로그램 사이에서 정보를 주고 받는 방법이나 규약

from http.server import HTTPServer, CGIHTTPRequestHandler

PORT = 8888

class Handler(CGIHTTPRequestHandler):
    cgi_directories = ['/cgi-bin']   # 여러개 넣을 수 있음.(리스트 형식)

def run():
    serv = HTTPServer(('127.0.0.1', PORT), Handler)

    print('웹서비스 진행중...')
    try:
        serv.serve_forever()   # 무한 루핑
    except Exception as err:
        print('서버종료')
    finally:
        serv.server_close()

if __name__ == '__main__':
    run()