About Tools  
  
### 1. Checking Diagnostic & Test Tools

The `mode` argument (`debug` or `test`) is required.

```bash
# Debug mode (check internal state of engine)
python examples/run_diagnostics.py debug

# Learning test mode (verify learning of simple word pairs)
python examples/run_diagnostics.py test

```

### 2. Checking Classification Tasks

The `task` argument (`text` or `mnist`) is required.

```bash
# Text classification (completes quickly)
python examples/run_classifier.py text

# MNIST image classification (data download will occur. Skip if time is limited)
python examples/run_classifier.py mnist --epochs 1 --samples 100
python examples/run_classifier.py mnist --epochs 5 --samples 1000
```

### 3. Checking Chat (Completed)

```bash
# Start training
python examples/run_chat.py --train 
  
# Start chat
python examples/run_chat.py
```
---