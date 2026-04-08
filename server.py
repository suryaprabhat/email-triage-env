import http.server
import socketserver

PORT = 7860

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        # Respond to POST requests identically to GET requests 
        # This prevents the 501 Unsupported Method error from browsers or Hugging Face health checks
        self.do_GET()

print(f"Starting resilient web server on port {PORT}...")
with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
    httpd.serve_forever()
