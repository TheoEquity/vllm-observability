FROM vllm/vllm-openai:v0.8.0@sha256:ce29db4955f06749d951cfa0ffd973741dcf9f1957d3166b6b5d75e86ff554ea
RUN python3 -m pip install --no-cache-dir pandas==2.2.3 datasets==3.3.2
# The base image pins NumPy 1.26; PyArrow 26 requires NumPy 2.
RUN python3 -m pip install --no-cache-dir pyarrow==19.0.1
