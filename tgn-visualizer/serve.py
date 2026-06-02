import http.server
import socketserver
import os

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    # Add proper MIME types
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

socketserver.TCPServer.allow_reuse_address = True
httpd = None
while True:
    try:
        httpd = socketserver.TCPServer(("", PORT), Handler)
        break
    except OSError as e:
        if e.errno == 98: # Address already in use
            print(f"Port {PORT} is in use, trying next port...")
            PORT += 1
        else:
            raise

with httpd:
    print(f"Serving 3D TGN Visualizer at http://localhost:{PORT}")
    httpd.serve_forever()
