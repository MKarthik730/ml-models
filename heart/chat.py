from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from text import OPEN_API_KEY  # your key module




#os.environ["OPENAI_API-KEY"]=OPEN_API_KEY
llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.6,
    api_key=OPEN_API_KEY,
)

prompt = PromptTemplate(
    input_variables=["input"],
    template="give details about {input}",
)

chain = prompt | llm


