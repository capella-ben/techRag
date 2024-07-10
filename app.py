import gradio as gr
from techRag import TechRAG
import os
from typing import List, Dict
from dotenv import load_dotenv, set_key
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


# Functiosn for the Settings tab.
def get_specific_env_vars():
    return {var: os.getenv(var, '') for var in ENV_VARS}

def update_env_var(key, value):
    if key not in ENV_VARS:
        return f"Error: {key} is not a valid environment variable for this application."
    
    # Update the environment variable
    os.environ[key] = value
    
    # Update the .env file
    set_key('.env', key, value)
    
    return view_env_vars()

def update_langsmith(value):
    os.environ["LANGCHAIN_TRACING_V2"] = str(value).lower()
    # Update the .env file
    set_key('.env', "LANGCHAIN_TRACING_V2", str(value).lower())

def str_to_bool(s):
    return s.lower() == 'true'

def view_env_vars():
    env_vars = get_specific_env_vars()
    return "\n".join([f"{k}: {v}" for k, v in env_vars.items()])

def update_value_input(key):
    return os.getenv(key, '')



# ------------------------------------- - ------------------------------------ #

load_dotenv()
vector_db_server = os.getenv('VECTOR_DB_STORE')
collection_name = os.getenv('COLLECTION_NAME')

os.environ['GRADIO_ANALYTICS_ENABLED'] = 'false'

# setup non-RAG LLM
vanilla_llm = OpenAI()

# Create the RAG application
tr = TechRAG(collection_name=collection_name, vector_db_host=vector_db_server, inform=True, debug=True)


# Environment variables we're interested in for Settings
ENV_VARS = ['VECTOR_DB_STORE', 'COLLECTION_NAME', 'USER_AGENT', 'OPENAI_API_KEY', 'TAVILY_API_KEY', 'LANGCHAIN_ENDPOINT', 'LANGCHAIN_API_KEY']



CSS = """
.wrap { 
    display: flex; 
    flex-direction: column; 
    flex-grow: 1;
}
.contain { 
    display: flex; 
    flex-direction: column; 
    flex-grow: 1;
}
.gradio-container { 
    min-height: 100vh !important; /* Changed from height to min-height */
    overflow-y: auto; /* Added to enable vertical scrolling */
}
.tabs { 
    display: flex !important; 
    flex-direction: column; 
    flex-grow: 1;
}
.tabitem[role="tabpanel"][style="display: block;"] { 
    display: flex !important; 
    flex-direction: column; 
    flex-grow: 1;
}
.gap {
    display: flex; 
    flex-direction: column; 
    flex-grow: 1;
}
#chatbot { 
    display: flex !important; 
    flex-direction: column; 
    flex-grow: 1; 
}

.fancy-line {
    border: 0;
    height: 1px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
    margin: 20px 0;
}
"""
x = """
/* Added to ensure content doesn't overflow horizontally */
body {
    overflow-x: hidden;
}
"""

with gr.Blocks(css=CSS, title="Tech RAG") as demo:
    with gr.Tabs() as tabs:
        with gr.TabItem("RAG Chat"):
            chatbot = gr.Chatbot(elem_id="chatbot", layout='panel')
            with gr.Row():
                with gr.Column(scale=16):
                    msg = gr.Textbox(placeholder="Type a message...", label="")
                with gr.Column(scale=1, min_width=60):
                    clear = gr.Button("Clear")

            def respond(message, chat_history):
                bot_message = rag_response(message, chat_history)
                chat_history.append((message, bot_message))
                return "", chat_history

            msg.submit(respond, [msg, chatbot], [msg, chatbot])
            clear.click(lambda: None, None, chatbot, queue=False)

        with gr.TabItem("Chat"):
            vanilla_chatbot = gr.Chatbot(elem_id="chatbot", layout='panel')
            with gr.Row():
                with gr.Column(scale=16):
                    vanilla_msg = gr.Textbox(placeholder="Type a message...", label="")
                with gr.Column(scale=1, min_width=60):
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
                gr.Markdown("# Ingest URLs")
                with gr.Row():
                    url_input = gr.Textbox(lines=5, placeholder="Enter URLs here, one per line", label="URLs")
                    url_output = gr.Textbox(label="Output")
                with gr.Row():
                    with gr.Column(scale=5):
                        url_button = gr.Button("Ingest URLs", variant='primary')
                        url_button.click(ingest_urls, inputs=url_input, outputs=url_output)
                    with gr.Column(scale=1):
                        url_clear = gr.Button("Clear")
                        url_clear.click(lambda: None, None, url_input, queue=False)

            gr.HTML("<hr class='fancy-line'>")

            # Ingest Fact
            with gr.Group():
                gr.Markdown("# Ingest Fact")
                with gr.Row():
                    with gr.Group():
                        fact_title = gr.Textbox(label="Fact Title")
                        fact_description = gr.Textbox(label="Description")
                        fact_content = gr.Textbox(label="Fact", lines=5)
                    fact_output = gr.Textbox(label="Output")
                with gr.Row():
                    with gr.Column(scale=5):
                        fact_button = gr.Button("Ingest Fact", variant='primary')
                        fact_button.click(ingest_fact, inputs=[fact_title, fact_description, fact_content], outputs=fact_output)
                    with gr.Column(scale=1):
                        fact_clear = gr.Button("Clear")
                        fact_clear.click(lambda: (None, None, None), None, [fact_title, fact_description, fact_content], queue=False)

            gr.HTML("<hr class='fancy-line'>")

            # Ingest Document
            with gr.Group():
                gr.Markdown("# Ingest Documents")
                with gr.Row():
                    docx_input = gr.File(file_count="multiple", file_types=[".docx"], type="filepath")
                    docx_output = gr.Textbox(label="Output", lines=1)
                with gr.Row():
                    with gr.Column(scale=5):
                        docx_button = gr.Button("Ingest Documents", variant='primary')
                        docx_button.click(ingest_docx, inputs=docx_input, outputs=docx_output)
                    with gr.Column(scale=1):
                        docx_clear = gr.Button("Clear")
                        docx_clear.click(lambda: None, None, docx_input, queue=False)

        with gr.TabItem("Settings"):

            with gr.Group():
                gr.Markdown("# Settings")
                with gr.Row():
                    #view_button = gr.Button("View Environment Variables")
                    env_vars_display = gr.Textbox(label="Current Settings", lines=5)


                with gr.Row():
                    with gr.Column():
                        key_input = gr.Dropdown(choices=ENV_VARS, label="Update Setting", value=ENV_VARS[0])
                    with gr.Column():
                        value_input = gr.Textbox(label="Value", value=os.getenv(ENV_VARS[0], ''))
                with gr.Row():
                    update_button = gr.Button("Update Setting", variant='primary')
            
            gr.HTML("<hr class='fancy-line'>")

            with gr.Group():
                gr.Markdown("# Options")
                with gr.Row():
                    langsmith_enable = gr.Checkbox(label="Enable LangSmith", value=str_to_bool(os.getenv("LANGCHAIN_TRACING_V2", "false")))
                    langsmith_update = gr.Button("Update", variant='primary')
                    langsmith_update.click(
                        update_langsmith,  
                        inputs=[langsmith_enable]
                        )
            
            key_input.change(
                update_value_input,
                inputs=[key_input],
                outputs=[value_input]
            )
            
            update_button.click(
                update_env_var,
                inputs=[key_input, value_input],
                outputs=[env_vars_display]
            )


            # Display current values on load
            env_vars_display.value = view_env_vars()

        with gr.TabItem("DB Prune"):
            selected_rows = gr.State([])
            
            gr.Markdown("# Vector Database Pruning")
            
            with gr.Row():
                search_input = gr.Textbox(label="Search Source")
                search_button = gr.Button("Search")
            
            with gr.Row():
                dataframe_output = gr.Dataframe(
                    interactive=False,
                    #col_count=(3, "fixed"),
                    row_count=(50, "dynamic"),
                    #wrap=True,
                )
                
            with gr.Row():
                delete_button = gr.Button("Delete Selected Rows")
                
            message_output = gr.Textbox(label="Message")
            
            def search_and_update(search_term):
                result_df = tr.vector_db_search(search_term)
                return result_df, []  # Return empty list to reset selected_rows
            
            search_button.click(
                search_and_update,
                inputs=[search_input],
                outputs=[dataframe_output, selected_rows]
            )
            
            search_input.submit(
                search_and_update,
                inputs=[search_input],
                outputs=[dataframe_output, selected_rows]
            )

            dataframe_output.select(
                tr.update_selected_rows_in_vector_db,
                [selected_rows],
                selected_rows
            ).then(
                lambda df, rows: tr.highlight_selected_rows(df, rows),
                [dataframe_output, selected_rows],
                dataframe_output
            )
            
            delete_button.click(
                tr.delete_vector_db_rows,
                inputs=[dataframe_output, selected_rows],
                outputs=[dataframe_output, message_output]
            ).then(
                lambda: [],
                None,
                selected_rows
            )




demo.launch(share=False, server_name="0.0.0.0", favicon_path="techRag.ico")
