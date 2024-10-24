import vertexai  # type: ignore
from vertexai.generative_models import GenerativeModel # type: ignore

# TODO(developer): Update and un-comment below line
# project_id = "PROJECT_ID"

vertexai.init(project="mokshagenai", location="us-central1")

model = GenerativeModel("gemini-1.5-flash-001")

response = model.generate_content(
    
    "meaning of moksha yagna"
)

print(response.text)


# import google.generativeai as genai
# import os

# genai.configure(api_key="AIzaSyDy9upqnCshBZI97h3C8aMda68Fwgb2IuM")

# model = genai.GenerativeModel('gemini-1.5-flash')

# response = model.generate_content("meaning of mokshaygana")
# print(response.text)    