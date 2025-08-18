# wsgi_simple_test.py - Simple test for PythonAnywhere
def application(environ, start_response):
    status = '200 OK'
    headers = [('Content-type', 'text/plain; charset=utf-8')]
    start_response(status, headers)
    return [b'Hello World from PythonAnywhere!']

if __name__ == "__main__":
    print("WSGI test file loaded successfully") 