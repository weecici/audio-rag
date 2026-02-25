from app.service.internal.chunk import _fallback_chunk_document, _fallback_chunk_transcript, parse_response_into_chunks

# 1. document fallback
docs = _fallback_chunk_document("Hello world. This is a test. Another sentence here. " * 50, max_tokens=10)
print(docs)

# 2. transcript fallback
transcript = """[0.0s - 1.0s] Hello
[1.0s - 2.0s] world
[2.0s - 3.0s] this
[3.0s - 4.0s] is
[4.0s - 5.0s] a test."""
trans_chunks = _fallback_chunk_transcript(transcript, max_tokens=2)
print(trans_chunks)

# 3. parse chunks
resp = """
chunk1 | 0 | 5
++++++++++
Hello world
==========
chunk2 | 5 | 10
++++++++++
this is a test
"""
parsed = parse_response_into_chunks(resp, "transcript")
print(parsed)
