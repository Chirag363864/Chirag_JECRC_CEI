import sys
import urllib.request
import json
import time

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or 'utf-8'
        print(text.encode(encoding, errors='replace').decode(encoding))

def test_endpoint(url, data=None):
    req = urllib.request.Request(url)
    if data is not None:
        req.add_header('Content-Type', 'application/json')
        jsondata = json.dumps(data).encode('utf-8')
    else:
        jsondata = None
        
    try:
        response = urllib.request.urlopen(req, data=jsondata, timeout=10)
        status = response.getcode()
        body = response.read().decode('utf-8')
        return status, body
    except Exception as e:
        return 500, str(e)

def run_tests():
    print("=== Testing Flask Web App API Endpoints ===")
    
    # 1. Test GET /
    status, body = test_endpoint("http://127.0.0.1:5000/")
    print(f"GET / : Status {status} | Length {len(body)} characters")
    assert status == 200, "GET / failed"
    assert "<title>Mini GPT-2 Scratch Playground</title>" in body, "GET / did not render correctly"
    
    # 2. Test GET /api/status
    status, body = test_endpoint("http://127.0.0.1:5000/api/status")
    print(f"GET /api/status : Status {status}")
    assert status == 200, "GET /api/status failed"
    status_data = json.loads(body)
    print(f"   Device: {status_data['device']}")
    print(f"   Vocab Size: {status_data['config']['vocab_size']}")
    print(f"   Layers: {status_data['config']['n_layer']} | Heads: {status_data['config']['n_head']}")
    assert not status_data['is_training'], "Should not be training initially"
    
    # 3. Test POST /api/generate
    print("Testing Text Generation & Matrix Weight Extraction...")
    payload = {
        "prompt": "ROMEO:\nWho is there?",
        "temperature": 0.8,
        "top_k": 20,
        "max_tokens": 15
    }
    status, body = test_endpoint("http://127.0.0.1:5000/api/generate", payload)
    print(f"POST /api/generate : Status {status}")
    assert status == 200, "POST /api/generate failed"
    gen_data = json.loads(body)
    safe_print(f"   Generated: {repr(gen_data['generated_text'])}")
    print(f"   Decoded Tokens count: {len(gen_data['tokens'])}")
    print(f"   Attention layer blocks exported: {gen_data['num_layers']} | heads count: {gen_data['num_heads']}")
    assert gen_data['num_layers'] > 0, "No attention weights returned!"
    assert len(gen_data['attention'][0][0]) == len(gen_data['tokens']), "Attention dimension mismatch with token count!"
    
    # 4. Test POST /api/train/start
    print("Testing Background Training Launcher Thread...")
    train_payload = {
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 64,
        "block_size": 64,
        "batch_size": 4,
        "max_iters": 30,
        "learning_rate": 6e-4,
        "pos_emb_type": "rope",
        "norm_type": "rmsnorm",
        "mlp_type": "swiglu",
        "dataset": "dante"
    }
    status, body = test_endpoint("http://127.0.0.1:5000/api/train/start", train_payload)
    print(f"POST /api/train/start : Status {status} | Body {body}")
    assert status == 200, "POST /api/train/start failed"
    
    # Poll status 3 times to see it iterate
    print("Polling Status during background training...")
    for i in range(3):
        time.sleep(1.5)
        status, body = test_endpoint("http://127.0.0.1:5000/api/status")
        status_data = json.loads(body)
        print(f"   Poll {i+1} | is_training: {status_data['is_training']} | iter: {status_data['current_iter']}")
        
    # 5. Test POST /api/train/stop
    print("Testing Asynchronous Training Interrupt...")
    status, body = test_endpoint("http://127.0.0.1:5000/api/train/stop", {})
    print(f"POST /api/train/stop : Status {status} | Body {body}")
    assert status == 200, "POST /api/train/stop failed"
    
    # Check final status
    time.sleep(1.0)
    status, body = test_endpoint("http://127.0.0.1:5000/api/status")
    status_data = json.loads(body)
    print(f"   Final Status | is_training: {status_data['is_training']}")
    assert not status_data['is_training'], "Should have halted training."
    
    print("\n=== ALL Flask API Verification Tests Passed! ===")

if __name__ == "__main__":
    run_tests()
