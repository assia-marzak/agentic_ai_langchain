from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv.ipython import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain.messages import HumanMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool

load_dotenv(override=True)

texts = [
    "Je m'appelle wissal arras, j'ai 21 ans et je suis étudiante en informatique.",
    "Je suis actuellement stagiaire en cybersécurité, où je travaille sur l'implémentation de l'architecture Zero Trust.",
    "Je me spécialise dans les frameworks NIST et la microsegmentation réseau pour remplacer le modèle 'Castle and Moat'.",
    "Je développe une application mobile appelée ShowBay avec React Native, Expo et Firebase pour le suivi de films.",
    "ShowBay est fortement inspirée de Letterboxd et permet aux utilisateurs de suivre leur historique de visionnage.",
    "Je travaille aussi sur un projet d'IA Agentique nommé SMA44 utilisant des modèles LLM locaux avec Ollama.",
    "Mon projet SMA44 se concentre sur les systèmes multi-agents et l'intégration du RAG (Retrieval-Augmented Generation).",
    "J'ai conçu un concept de 'Virtual Stylist' utilisant des capteurs IA pour donner des conseils de mode personnalisés.",
    "Dans le cadre de ma formation, j'utilise régulièrement Wireshark pour l'analyse de trafic réseau et le troubleshooting DNS.",
    "Je maîtrise le développement frontend avec React et le développement d'applications mobiles cross-platform."
]

embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(
    chunks= texts,
    collection_name="cv_tformation",
    embedding=embedding_model

)
retriever= vector_store.as_retriever()

retriever_tool = create_retriever_tool(

    retriever=retriever,
    name="cv_tool",
    description="get information about me ",

)    

@tool
def get_employee_info(name: str) :
    """
    Get information about a given employee (name, salary, seniority)
    """
    print("get_employee_info tool invoked")
   
    return {"name": name, "salary": "12000", "seniority": "5"}
@tool
def send_email(email:str, subject:str, content:str):
    """
    Send an email with subject and content
    """
    
    print(f"sending email to {email}, subject :{subject},content :{content}")
    return f"Email succesfully sent to {email} with subject {subject} and content {content}"

llm=ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(
    model=llm,
    tools=[get_employee_info, send_email,retriever_tool],
    system_prompt="answer to user query using provided tools"
)
#resp=agent.invoke(
# input={"messages": [HumanMessage("c'est quoi le salaire de John Doe ?")]})
#print(resp['messages'][-1].content)