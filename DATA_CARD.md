# Workload and quality data

`workload.py` contains 24 self-authored English document/query pairs about a fictional support/developer knowledge base. Each query has one designated relevant document. Documents cover account access, billing, delivery, API behavior, data operations and model serving. No customer data, external customer dataset or paid data source was used.

The 16 performance texts mix eight shorter documents with eight longer authored concatenations. The exact text list and retrieval pairs are included in the workload SHA-256. All inference variants receive identical inputs. Query/document order is fixed; ranking ties are broken by original document order.

No training, tuning, threshold selection or quantization calibration uses this set. Dynamic quantization derives runtime activation parameters and quantizes supported model weights, without a calibration dataset. Quality results were not used to revise the authored labels. Future test-informed model/data changes need a new evaluation version.

These manually authored labels have not been independently adjudicated. Some real search intents can have multiple relevant documents; this narrow dataset defines only one. Recall and MRR measure performance on this authored catalog only, not general semantic search, multilingual support, production traffic or business outcomes.

The model is Apache-2.0, as documented by its upstream model card. The downloaded model card and immutable revision are kept with local artifacts; no model weights are embedded in the source repository.
