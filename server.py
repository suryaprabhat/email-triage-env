import http.server
import socketserver
import os

PORT = 7860

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # If the requested file doesn't exist (like a random health check or favicon ping),
        # gracefully serve the main index.html instead of throwing a 404 error page.
        path = self.translate_path(self.path)
        if not os.path.exists(path):
            self.path = '/index.html'
        super().do_GET()

    def do_POST(self):
        # Respond to POST requests identically to GET requests 
        self.do_GET()

print(f"Starting resilient web server on port {PORT}...")
with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    httpd.serve_forever()
