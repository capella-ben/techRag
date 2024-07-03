import gradio as gr
from techRag import TechRAG
import os
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI



# ---------------------------- Interface Functions --------------------------- #
def rag_response(message: str, history: List[str]) -> str:
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


def ingest_docx(filenames: List[str]) -> str:
    """Ingest data from a list of word documents

    Args:
        filenames (list[str]): List of the filenames to ingest

    Returns:
        str: Status of ingestion
    """
    return tr.ingest_docx(filenames)


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


def vanilla_chat(message, history):
    """
    Generate a chat response using a language model without RAG.

    This function takes a new message and chat history, formats them for the OpenAI API,
    and streams the response from the language model.

    Args:
        message (str): The current message from the user.
        history (list): A list of tuples, where each tuple contains two strings:
                        (human_message, assistant_response).

    Yields:
        str: Partial responses from the language model, incrementally building
             the complete response.

    Raises:
        Any exceptions raised by the underlying API calls are not caught in this function.
    """

    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})
  
    response = vanilla_llm.chat.completions.create(model='gpt-4o',
    messages= history_openai_format,
    temperature=1.0,
    stream=True)

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              yield partial_message




load_dotenv()
vector_db_server = os.getenv('VECTOR_DB_STORE')
collection_name = os.getenv('COLLECTION_NAME')

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'false'

# setup non-RAG LLM
vanilla_llm = OpenAI()

# Create the RAG application
tr = TechRAG(collection_name=collection_name, vector_db_host=vector_db_server, inform=True, debug=True)

CSS = """
.wrap { display: flex; flex-direction: column; flex-grow: 1}
.contain { display: flex; flex-direction: column; flex-grow: 1}
.gradio-container { height: 100vh !important; }
.tabs { display: flex !important; flex-direction: column; flex-grow: 1}
.tabitem[role="tabpanel"][style="display: block;"] { display: flex !important; flex-direction: column; flex-grow: 1}
.gap {display: flex; flex-direction: column; flex-grow: 1}
#chatbot { display: flex !important; flex-direction: column; flex-grow: 1; }

.fancy-line {
    border: 0;
    height: 1px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
    margin: 20px 0;
}

"""

with gr.Blocks(css=CSS, title="Tech RAG") as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("RAG Chat"):
            chatbot = gr.Chatbot(elem_id="chatbot", layout='panel')
            msg = gr.Textbox(placeholder="Type a message...", label="")
            clear = gr.Button("Clear")

            def respond(message, chat_history):
                bot_message = rag_response(message, chat_history)
                chat_history.append((message, bot_message))
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        with gr.TabItem("Chat"):
            vanilla_chatbot = gr.Chatbot(elem_id="chatbot", layout='panel')
            vanilla_msg = gr.Textbox(placeholder="Type a message...", label="")
            vanilla_clear = gr.Button("Clear")

            def vanilla_respond(message, chat_history):
                bot_message = ""
                for chunk in vanilla_chat(message, chat_history):
                    bot_message = chunk
                    yield "", chat_history + [[message, bot_message]]
                chat_history.append((message, bot_message))
                return "", chat_history
            
            vanilla_msg.submit(vanilla_respond, [vanilla_msg, vanilla_chatbot], [vanilla_msg, vanilla_chatbot])
            vanilla_clear.click(lambda: None, None, vanilla_chatbot, queue=False)

        with gr.TabItem("Ingest"):
            # Ingest URL
            with gr.Group():
                with gr.Row():
                    url_input = gr.Textbox(lines=5, placeholder="Enter URLs here, one per line", label="URLs")
                    url_output = gr.Textbox(label="Output")
                url_button = gr.Button("Ingest URLs")
                url_button.click(ingest_urls, inputs=url_input, outputs=url_output)

            gr.HTML("<hr class='fancy-line'>")

            # Ingest Fact
            with gr.Group():
                with gr.Row():
                    with gr.Group():
                        fact_title = gr.Textbox(label="Fact Title")
                        fact_description = gr.Textbox(label="Description")
                        fact_content = gr.Textbox(label="Fact", lines=5)
                    fact_output = gr.Textbox(label="Output")
                fact_button = gr.Button("Ingest Fact")

            fact_button.click(ingest_fact, inputs=[fact_title, fact_description, fact_content], outputs=fact_output)

            gr.HTML("<hr class='fancy-line'>")

            # Ingest Document
            with gr.Group():
                with gr.Row():
                    docx_input = gr.File(file_count="multiple", file_types=[".docx"], type="filepath")
                    docx_output = gr.Textbox(label="Output")
                docx_button = gr.Button("Ingest Documents")

            docx_button.click(ingest_docx, inputs=docx_input, outputs=docx_output)

demo.launch(share=False, server_name="0.0.0.0", favicon_path="techRag.ico")
