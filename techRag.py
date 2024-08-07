from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_milvus.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph
import gradio as gr
from pymilvus import connections, Collection
import os
import pypandoc
import pandas as pd


# Required environment variables:
# OPENAI_API_KEY, TAVILY_API_KEY

# Optional environment variables for langsmith
# LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        history: list of strings
    """

    question: str
    generation: str
    documents: List[str]
    history: list[str]



class TechRAG:
   

    def __init__(self, collection_name: str,  vector_db_host: str, vector_db_port = "19530", vector_certainty = 0.85,
                 large_llm_name = "gpt-4o", small_llm_name = "gpt-4o-mini", embedding_model_name = "text-embedding-3-large", 
                 inform=False, debug=False):

        # vector_certainty = Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to maximum relevance. Defaults to 0.5.

        self.collection_name = collection_name
        self.large_llm_Model_name = large_llm_name
        self.small_llm_Model_name = small_llm_name
        self.embedding_model_name = embedding_model_name
        self.inform = inform
        self.debug = debug


        # ----------------------------- Milvus connection ---------------------------- #
        connections.connect("default", host=vector_db_host, port=vector_db_port)
        self.collection = Collection(collection_name)

        # ------------------------------------ LLM ----------------------------------- #
        self.llm = ChatOpenAI(model=self.large_llm_Model_name, temperature=0)
        self.llm_small = ChatOpenAI(model=self.small_llm_Model_name, temperature=0)

        self.embedding_function = OpenAIEmbeddings(model=self.embedding_model_name)

        # ------------------------------- Vector Store ------------------------------- #
        self.vectorstore = Milvus(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            connection_args={"host": vector_db_host, "port": vector_db_port}, 
            auto_id=True
        )
        self.retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6 , "lambda_mult": vector_certainty})
        # lambda_mult (float) – Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to maximum relevance. Defaults to 0.5.

        self.vs_length = self.__get_vector_db_text_length()


        
        # ----------------------------- Retreival Grader ----------------------------- #
        # LLM with function call
        structured_llm_grader = self.llm_small.with_structured_output(GradeDocuments)

        # Prompt
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to all or part of the user question, grade it as relevant. \n
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        self.retrieval_grader = grade_prompt | structured_llm_grader
        

        # --------------------------------- Generate --------------------------------- #
        # Prompt
        #prompt = hub.pull("rlm/rag-prompt")
        prompt = hub.pull("quacktheduck/rag-prompt-history-lang")       # this prompt has support for history

        # Chain
        self.rag_chain = prompt | self.llm | StrOutputParser()


        
        # ---------------------------------- Search ---------------------------------- #

        self.web_search_tool = TavilySearchResults(k=3)


        # -------------------------------- Build Graph ------------------------------- #
        self.workflow = StateGraph(GraphState)

        # Define the nodes
        self.workflow.add_node("web_search", self.__web_search)  # web search
        self.workflow.add_node("retrieve", self.__retrieve)  # retrieve
        self.workflow.add_node("grade_documents", self.__grade_documents)  # grade documents
        self.workflow.add_node("generate", self.__generate)  # generatae

        # Build graph
        self.workflow.set_entry_point("retrieve")
        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.__decide_to_generate,
            {
                "web_search": "web_search",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("web_search", "generate")
        self.workflow.add_edge("generate", END)


        # Compile
        self.app = self.workflow.compile()








    # -------------------------------- Graph Nodes -------------------------------- #

    def __retrieve(self, state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        # Retrieval
        documents = self.retriever.invoke(question)
        print(f'   {len(documents)} found')
        return {"documents": documents, "question": question}


    def __generate(self, state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        history = state['history']

        # RAG generation
        generation = self.rag_chain.invoke({"context": documents, "question": question, "chat_history": history, "language": "English"})
        return {"documents": documents, "question": question, "generation": generation}


    def __grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                print("   ---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("   ---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        return {"documents": filtered_docs, "question": question}


    def __web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        print(f'   {len(docs)} found')
        # todo: handle what to do if there are no results.  Maybe return no results and then use a null context???
        web_results = "\n".join([d["content"] for d in docs if d['content']])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}


    # -------------------------------- Graph Edges ------------------------------- #


    def __decide_to_generate(self, state):
        """
        Determines whether to generate an answer, or do a web search.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        question = state["question"]
        filtered_documents = state["documents"]

        if not filtered_documents:
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "   ---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION. PERFORM WEB SEARCH---"
            )
            return "web_search"
        else:
            # We have relevant documents, so generate answer
            print("   ---DECISION: GENERATE---")
            return "generate"


    def __get_vector_db_text_length(self) -> int:
        schema = self.collection.schema
        schema_dict = schema.to_dict()
        # Extract the max_length of the "text" field
        max_length = None
        for field in schema_dict['fields']:
            if field['name'] == 'text':
                max_length = field['params'].get('max_length')
                break

        return max_length
  

    def ingest_docs(self, docs: List[Document]) -> str:
        return_str = ""
        
        # Make sure all the metadata is populated.  If it is not then Milvus won't be able to import it. 
        for doc in docs:
            for d in doc:
                source = d.metadata.get('source', '').split('/')[-1]
                
                if d.metadata.get('title') is None:
                    d.metadata['title'] = source
                if d.metadata.get('description') is None:
                    d.metadata['description'] = source
                if d.metadata.get('language') is None:
                    d.metadata['language'] = 'en'
            

        docs_list = [item for sublist in docs for item in sublist]

        print("Splitting documents...")
        if self.inform and self.debug: gr.Info(f"Splitting documents...")
        # Split
        text_splitter = SemanticChunker(self.embedding_function)

        doc_splits = text_splitter.split_documents(docs_list)

        # check that the split is not too big for the vector store. 
        for split in doc_splits:
            if len(split.page_content) > self.vs_length:
                # delete the element
                doc_splits.remove(split)
                print(f"Split removed to to excessive length (over {self.vs_length})")
                return_str = return_str + f"Split removed to to excessive length (over {self.vs_length})\n"

       
        print("Adding to vector store...")
        if self.inform and self.debug: gr.Info(f"Adding to vector store...")
        # Add to vectorstore
        ids = self.vectorstore.add_documents(documents=doc_splits)
        print(f"Done!  {len(doc_splits)} records added")
        if self.inform: gr.Info(f"Done!  {len(doc_splits)} records added")
        
        return return_str + str(len(doc_splits)) + " records ingested"


    
    # ----------------------------- Public functions ----------------------------- #

    def ingest_urls(self, urls) -> str:
        return_str = ""     # build a results string with info on such things as the url's that are already in the vector store.

        urls = [s for s in urls if s.strip()]       # clean out any empty entries

        print("Loading documents...")
        if self.inform and self.debug: gr.Info(f"Loading documents...")

        # check to see if the URL is already in the vector store
        clean_urls = []
        for u in urls:
            results = self.collection.query(expr=f'source == "{u}"')
            if not results: 
                clean_urls.append(u)
            else:
                return_str = return_str + u  + " is already in vector store" + "\n"

        print(f"filtered: {len(clean_urls)}")
        if len(clean_urls) == 0: return return_str       # if there are no urls left in the list then 


        # Load
        docs = [WebBaseLoader(url).load() for url in clean_urls]

        # Make sure all the metadata is populated.  If it is not then Milvus won't be able to import it. 
        for doc in docs:
            for d in doc:
                source = d.metadata.get('source', '').split('/')[-1]
                
                if d.metadata.get('title') is None:
                    d.metadata['title'] = source
                if d.metadata.get('description') is None:
                    d.metadata['description'] = source
                if d.metadata.get('language') is None:
                    d.metadata['language'] = 'en'
            

        docs_list = [item for sublist in docs for item in sublist]

        print("Splitting documents...")
        if self.inform and self.debug: gr.Info(f"Splitting documents...")
        # Split
        text_splitter = SemanticChunker(self.embedding_function)

        doc_splits = text_splitter.split_documents(docs_list)

        # check that the split is not too big for the vector store. 
        for split in doc_splits:
            if len(split.page_content) > self.vs_length:
                # delete the element
                doc_splits.remove(split)
                print(f"Split removed to to excessive length (over {self.vs_length})")
                return_str = return_str + f"Split removed to to excessive length (over {self.vs_length})\n"

       
        print("Adding to vector store...")
        if self.inform and self.debug: gr.Info(f"Adding to vector store...")
        # Add to vectorstore
        ids = self.vectorstore.add_documents(documents=doc_splits)
        print(f"Done!  {len(doc_splits)} records added")
        if self.inform: gr.Info(f"Done!  {len(doc_splits)} records added")
        
        return return_str + str(len(doc_splits)) + " records ingested"


    def ingest_docx(self, filenames) -> str:
        return_str = ""     # build a results string with info on such things as the url's that are already in the vector store.

        filenames = [s for s in filenames if s.strip()]       # clean out any empty entries


        # check to see if the filename is already in the vector store
        clean_filenames = []
        for u in filenames:
            results = self.collection.query(expr=f'source == "{os.path.basename(u)}"')
            if not results: 
                clean_filenames.append(u)
            else:
                return_str = return_str + u  + " is already in vector store" + "\n"
        filenames = clean_filenames


        md_filenames = []
        # convert to markdown
        for filename in filenames:
            md_filename = str(filename).replace(".docx", ".md")
            pypandoc.convert_file(filename, 'md', outputfile=md_filename, encoding='utf-8')
            md_filenames.append(md_filename)


        print("Loading and Splitting Word documents...")
        if self.inform and self.debug: gr.Info(f"Loading and Splitting Word documents...")


        print(f"filtered: {len(md_filenames)}")
        if len(md_filenames) == 0: return return_str       # if there are no urls left in the list then 


        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2")
        ]

        # Load and split
        doc_splits=[]
        for md_filename in md_filenames:
            
            with open(md_filename, 'r', encoding="utf8", errors="ignore") as mdf:
                md = mdf.read()
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line=False, strip_headers=False)
            docs = markdown_splitter.split_text(md)
            
            # remove any splits that don't have a header and are the Table of Contents
            docs = [d for d in docs if d.metadata.get('Header 1') and d.metadata['Header 1'] != "Table of Contents"]

            # Make sure all the metadata is populated.  If it is not then Milvus won't be able to import it.
            for d in docs:
                source = os.path.basename(md_filename)
                title = d.metadata.get('Header 1', '') + ', ' + d.metadata.get('Header 2', '')

                d.metadata['source'] = source
                if d.metadata.get('title') is None:
                    d.metadata['title'] = source
                if d.metadata.get('description') is None:
                    d.metadata['description'] = title
                if d.metadata.get('language') is None:
                    d.metadata['language'] = 'en'

            doc_splits = doc_splits + docs


        secondary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.vs_length - 512,
            chunk_overlap=200,
            length_function=len,
        )

        new_doc_splits = []
        for split in doc_splits:
            if len(split.page_content) > self.vs_length:
                # Split the document if it's too long
                split_docs = secondary_splitter.split_documents([split])
                new_doc_splits.extend(split_docs)
                print(f"Split document into {len(split_docs)} smaller chunks due to excessive length (over {self.vs_length})")
                return_str = return_str + f"Split document into {len(split_docs)} smaller chunks due to excessive length (over {self.vs_length})\n"
            else:
                new_doc_splits.append(split)

        doc_splits = new_doc_splits
       
        print("Adding to vector store...")
        if self.inform and self.debug: gr.Info(f"Adding to vector store...")
        # Add to vectorstore
        ids = self.vectorstore.add_documents(documents=doc_splits)
        print(f"Done!  {len(doc_splits)} records added")
        if self.inform: gr.Info(f"Done!  {len(doc_splits)} records added")
        
        return return_str + str(len(doc_splits)) + " records ingested"



    def ingest_fact(self, title: str, description: str, fact: str) -> str:
        """Load a fact into the vector store

        Args:
            title (str): Title of the fact
            description (str): Description of the fact (short summary)
            fact (str): Details of the fact.

        Returns:
            int: Number of records ingested
        """

        print("Splitting documents...")
        if self.inform and self.debug: gr.Info(f"Splitting documents...")
        # Split
        text_splitter = SemanticChunker(self.embedding_function)

        facts = text_splitter.split_text(fact)

        # check that the split is not too big for the vector store
        for split in facts:
            if len(split) > self.vs_length:
                # delete the element
                facts.remove(split)
                print(f"Split removed to to excessive length (over {self.vs_length})")
                return_str = return_str + f"Split removed to to excessive length (over {self.vs_length})\n"


        metadata = {
            'source': 'Fact', 
            'title': title, 
            'description': description, 
            'language': 'No language found.'}

        metadatas = [metadata for _ in range(len(facts))]

        print("Adding fact to vector store...")
        if self.inform and self.debug: gr.Info(f"Adding fact to vector store...")
        # Add to vectorstore
        
        ids = self.vectorstore.add_texts(texts=facts, metadatas=metadatas)
        print(f"Done!  {len(facts)} records added")
        if self.inform: gr.Info(f"Done!  {len(facts)} facts added")
        return str(len(facts)) + " facts ingested"

        



    def ask(self, query:str, history: List[str]) -> str:
        """Ask a question

        Args:
            query (str): The question to ask
            history (List[str]): List of previos questions and answers

        Returns:
            str: The generated reaponse
        """

        inputs = {"question": query,
                  "history": history}
        
        for output in self.app.stream(inputs):
            for key, value in output.items():
                # Node
                print(f"Node '{key}' complete.")
                if self.inform and self.debug: 
                    if key == 'retrieve':
                        gr.Info(f"{len(value['documents'])} documents retreived")

                    if key == 'grade_documents':
                        gr.Info(f"{len(value['documents'])} documents are relevant")

                    if key == 'web_search':
                        gr.Info(f"Web searched")

            print("\n---\n")

        # Final generation
        return value["generation"]
        

    def list_collection_sources(self) -> List[str]:
        results = self.collection.query(
            expr='source != "Fact"',
            output_fields=["source", "title"]
        )

        sources = list({item["source"] for item in results})

        return sources


    def vector_db_search(self, search_term: str):
        """Search the source field of the vector databas

        Args:
            search_term (str): The string to search for 

        Returns:
            DataFrame: DataFrame of results
        """
        results = self.collection.query(
            expr=f'source like "%{search_term}%"',
            output_fields=["source", "title", "pk"],
            limit=50
        )
        return pd.DataFrame(results)

    
    def update_selected_rows_in_vector_db(self, evt: gr.SelectData, current_selections: List):
        """Maintain the list of rows selected in the dataframe

        Args:
            evt (gr.SelectData): Selected data
            current_selections (List): List of rows selected

        Returns:
            List: List of rows selected
        """
        if current_selections is None:
            current_selections = []
        
        if evt.index[0] in current_selections:
            current_selections.remove(evt.index[0])
        else:
            current_selections.append(evt.index[0])
        
        return sorted(current_selections)


    def delete_vector_db_rows(self, df, selected_rows):
        """Delete rows from the vector database

        Args:
            df (DataFrame): Dataframe to delete from
            selected_rows (List): Selected rows

        Returns:
            Dataframe: Dataframe with rows removed
            str: Result text
        """
        if not selected_rows:
            return df, "No rows selected. Please select one or more rows to delete."
        for selected_row in selected_rows:
            print(df.iloc[selected_row]['pk'])
            self.collection.delete(f'pk == {df.iloc[selected_row]["pk"]}')
        df = df.drop(df.index[selected_rows]).reset_index(drop=True)
        return df, f"Deleted {len(selected_rows)} row(s) successfully."


    def highlight_selected_rows(self, df, selected_rows):
        """Highlight selected rows

        Args:
            df (DataFrame): Dataframe to update
            selected_rows (List): List of selcted rows

        Returns:
            DataFrame: Dataframe with rows highlighted 
        """
        def highlight_row(row):
            if row.name in selected_rows:
                return ['background-color: yellow'] * len(row)
            return [''] * len(row)
        
        return df.style.apply(highlight_row, axis=1)
