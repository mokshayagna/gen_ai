import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project = "mokshagenai",location = "us-central1")

callingmodel = GenerativeModel("gemini-1.5-flash-001")

response = callingmodel.generate_content("what will the average temparature in tirupati in july")

print(response.text)
