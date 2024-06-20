import gradio as gr
from techRag import TechRAG
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()
vector_db_server = os.getenv('VECTOR_DB_STORE')

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'true'


# ---------------------------- Interface Functions --------------------------- #
def chatbot_response(message: str, history: List[str]) -> str:
    """Chatbot Interface Function

    Args:
        message (str): The new message
        history (List[str]): List of previous chats question and answers

    Returns:
        str: response
    """
    bot_response = tr.ask(message, history)
    return bot_response


def ingest_urls(urls: List[str]) -> str:
    """Ingest data from a list of URL's

    Args:
        urls (list[str]): List of the URL's to ingest

    Returns:
        str: Status of ingestion
    """
    url_list = urls.split("\n")
    return tr.ingest_urls(url_list)


def ingest_fact(title: str, description: str, fact: str) -> str:
    """Ingest manually entered data

    Args:
        title (str): Title of the fact
        description (str): Description of the fact
        fact (str): Detail of the fact

    Returns:
        str: Status of ingestion
    """
    return tr.ingest_fact(title, description, fact)


# ------------------------- Create Interface Elements ------------------------ #
chatbot_interface = gr.ChatInterface(
    fn=chatbot_response,
    #title="Chatbot",
    #description="Enter your message to the chatbot.",
    fill_height=True,
    retry_btn=None,
    chatbot=gr.Chatbot(elem_id="chatbot")
)


url_ingestion_interface = gr.Interface(
    fn=ingest_urls,
    inputs=gr.Textbox(lines=5, placeholder="Enter URLs here, one per line"),
    outputs="text",
    title="URL Ingestion",
    description="Submit URLs for ingestion, one per line.",
    allow_flagging="never"
)

fact_ingestion_interface = gr.Interface(
    fn=ingest_fact,
    inputs=[gr.Textbox(label="Title"), gr.Textbox(label='Description'), gr.Textbox(label="Fact", lines=5)],
    outputs="text",
    title="Fact Ingestion",
    allow_flagging="never"
)


# in the CSS you need to config the full heirarchy of divs to be 'flex' so that the one you want (#chatbot) can be set to grow. 
CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
.tabs { display: flex !important; flex-direction: column; flex-grow: 1}
.tabitem[role="tabpanel"][style="display: block;"] { display: flex !important; flex-direction: column; flex-grow: 1}
.gap {display: flex; flex-direction: column; flex-grow: 1}
#chatbot { flex-grow: 1; }
"""

# create the tabbed interface.
myApp = gr.TabbedInterface([chatbot_interface, url_ingestion_interface, fact_ingestion_interface], 
                           ["Chat", "Ingest URL", "Ingest Fact"],
                           css=CSS)

# create the RAG application
tr = TechRAG(collection_name="techRag", vector_db_host="192.168.0.205", inform=True, debug=True)

# Launch the demo
myApp.launch(share=False, server_name="0.0.0.0")




