if __name__ == '__main__':

    # importing python modules:S1
    try:
        from pathlib import Path
        import os
        import json
        from typing import List
        from dotenv import load_dotenv
    except Exception as error:
        print(f'ERROR - [Main:S1] - {error}')


    # define folder path:S2
    try:
        parent_folder_path = Path.cwd()
        models_folder_path = parent_folder_path / "models"
        input_folder_path = parent_folder_path / "input"
        pdf_document_path = input_folder_path / "Sample-DOC.pdf"
    except Exception as error:
        print(f'ERROR - [Main:S2] - {error}')


    # load environment variables:S3
    try:
        load_dotenv()
        AZURE_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
        AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
        AZURE_LLM_DEPLOYMENT = os.getenv("AZURE_OPENAI_API_LLM_MODEL_NAME")
    except Exception as error:
        print(f'ERROR - [Main:S3] - {error}')


    # redirect model caches:S4
    try:
        os.environ["UNSTRUCTURED_HOME"] = str(models_folder_path)
        os.environ["HF_HOME"] = str(models_folder_path / "hf_cache") #type: ignore
        os.environ["TRANSFORMERS_CACHE"] = str(models_folder_path / "hf_cache") #type: ignore
        os.environ["TORCH_HOME"] = str(models_folder_path / "torch_cache") #type: ignore
    except Exception as error:
        print(f'ERROR - [Main:S4] - {error}')


    # importing unstructured modules:S5
    try:
        from unstructured.partition.pdf import partition_pdf
        from unstructured.chunking.title import chunk_by_title
        from langchain_core.documents import Document
        from langchain_openai import AzureChatOpenAI
        from langchain_core.messages import HumanMessage
    except Exception as error:
        print(f'ERROR - [Main:S5] - {error}')


    # partition pdf document:S6
    try:
        pdf_doc_elements = partition_pdf(
            filename = str(pdf_document_path),
            strategy = "hi_res",
            languages=["eng"],
            infer_table_structure = True,
            extract_image_block_types = ["Image"],
            extract_image_block_to_payload = True
        )
    except Exception as error:
        print(f'ERROR - [Main:S6] - {error}')


    # creating chunks by title:S7
    try:
        title_chunks = chunk_by_title(
            pdf_doc_elements,
            max_characters = 3000,
            new_after_n_chars = 2400,
            combine_text_under_n_chars=500
        )
    except Exception as error:
        print(f'ERROR - [Main:S7] - {error}')


    # preparing multimodal content extraction:S8
    try:
        separated_chunk_data = []
        for chunk in title_chunks:
            content_data = {
                "text": chunk.text,
                "tables": [],
                "images": [],
                "types": ["text"]
            }
            if hasattr(chunk, "metadata") and hasattr(chunk.metadata, "orig_elements"):
                for element in chunk.metadata.orig_elements: #type: ignore
                    element_type = type(element).__name__
                    if element_type == "Table":
                        content_data["types"].append("table")
                        table_html = getattr(
                            element.metadata,
                            "text_as_html",
                            element.text
                        )
                        content_data["tables"].append(table_html)
                    elif element_type == "Image":
                        if hasattr(element.metadata, "image_base64"):
                            content_data["types"].append("image")
                            content_data["images"].append(
                                element.metadata.image_base64
                            )
            content_data["types"] = list(set(content_data["types"]))
            separated_chunk_data.append(content_data)
    except Exception as error:
        print(f'ERROR - [Main:S8] - {error}')


    # initialize Azure OpenAI LLM for multimodal summarization:S9
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=AZURE_API_ENDPOINT,
            api_key=AZURE_API_KEY, #type: ignore
            api_version=AZURE_API_VERSION,
            azure_deployment=AZURE_LLM_DEPLOYMENT,
            temperature=0
        )
    except Exception as error:
        print(f'ERROR - [Main:S9] - {error}')


    # processing chunks and generating AI summaries:S10
    try:
        rag_documents = []
        total_chunks = len(separated_chunk_data)
        print("🧠 Processing chunks with AI summaries...")
        for index, content_data in enumerate(separated_chunk_data):
            current_chunk = index + 1
            print(f"Processing chunk {current_chunk}/{total_chunks}")
            text_content = content_data["text"]
            tables = content_data["tables"]
            images = content_data["images"]
            if tables or images:
                print("Creating AI summary for mixed content...")
                try:
                    prompt_text = f"""You are creating a searchable description for document content retrieval.
                    CONTENT TO ANALYZE:

                    TEXT CONTENT:
                    {text_content}
                    """
                    if tables:
                        prompt_text += "TABLES:\n"
                        for i, table in enumerate(tables):
                            prompt_text += f"Table {i+1}:\n{table}\n\n"
                    prompt_text += """
                    YOUR TASK:
                    Generate a comprehensive, searchable description that covers:

                    1. Key facts, numbers, and data points from text and tables
                    2. Main topics and concepts discussed
                    3. Questions this content could answer
                    4. Visual content analysis (charts, diagrams, patterns in images)
                    5. Alternative search terms users might use

                    Make it detailed and searchable.

                    SEARCHABLE DESCRIPTION:"""
                    message_content = [
                        {"type": "text", "text": prompt_text}
                    ]
                    for image_base64 in images:
                        message_content.append({ #type: ignore
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        })
                    message = HumanMessage(content=message_content) #type: ignore
                    response = llm.invoke([message])
                    enhanced_content = str(response.content)
                except Exception as ai_error:
                    print(f"❌ AI summary failed: {ai_error}")
                    enhanced_content = text_content
            else:
                enhanced_content = text_content
            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "original_content": json.dumps({
                        "raw_text": text_content,
                        "tables_html": tables,
                        "images_base64": images
                    })
                }
            )
            rag_documents.append(doc)
        print(f"✅ Processed {len(rag_documents)} chunks")
        # DEBUG: print processed chunk
        print("\n---------------- CHUNK OUTPUT ----------------")
        print(f"Chunk Number: {current_chunk}")
        print(f"Text Length: {len(text_content)}")
        print(f"Tables Found: {len(tables)}")
        print(f"Images Found: {len(images)}")
        print("\nAI Enhanced Content Preview:\n")
        print(enhanced_content[:800])  # prevent huge console spam
        print("\n----------------------------------------------\n")
    except Exception as error:
        print(f'ERROR - [Main:S10] - {error}')