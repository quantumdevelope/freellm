"""Fake OpenAI-compatible backend used for the end-to-end smoke test."""

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATE = {"turn": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.endswith("/models"):
            self._json({"object": "list", "data": [{"id": "fake-local-model"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        STATE["turn"] += 1

        if STATE["turn"] == 1:
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"path": "hello.txt"}),
                        },
                    }
                ],
            }
        else:
            message = {"role": "assistant", "content": "The file says: hello from a local model."}

        self._json(
            {
                "id": "chatcmpl-1",
                "model": "fake-local-model",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 42, "completion_tokens": 11},
            }
        )


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8799
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
