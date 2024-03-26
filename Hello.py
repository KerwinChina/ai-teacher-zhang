import ast
import uuid
import spacy
from openai import OpenAI
import streamlit as st
from supabase import create_client, Client
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import PostgresChatMessageHistory
from langchain.chains import ConversationChain

from ChatZhipuAI import ChatZhipuAI



with st.sidebar:
    selected_option = st.selectbox('请选择一个公司', ['zhipuai', 'openai'])
    if selected_option == 'zhipuai':
        model_selected_option = st.selectbox('请选择一个模型', ['glm-4', 'glm-3-turbo'])
        custom_openai_api_key = st.secrets["zhipu_ai_key"]
    else:
        model_selected_option = st.selectbox('请选择一个模型', ['gpt-4-0125-preview', 'gpt-3.5-turbo-0125'])
        custom_openai_api_key = st.text_input("API Key", key="chatbot_api_key", type="password")
    # supabase_url = st.text_input("supabase URL", key="supabase URL", type="password")
    # supabase_key = st.text_input("supabase KEY", key="supabase KEY", type="password")
    supabase_url = st.secrets["supabase_url"]
    supabase_key = st.secrets["supabase_key"]
    open_ai_key = st.secrets["open_ai_key"]
    connection_string = st.secrets["connection_string"]
    
# 获取session_id
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())
session_id = st.session_state['session_id']
# session_id = 'session_id_14'



# if custom_openai_api_key:
#     if selected_option=='zhipuai':
#         client = ZhipuAI(api_key=custom_openai_api_key)  # 填写您自己的APIKey
#     else:
#         chat = ChatOpenAI(openai_api_key=custom_openai_api_key, model_name=model_selected_option)
#     embedding1536 = OpenAIEmbeddings(openai_api_key=open_ai_key
#                                  ,
#                                  model="text-embedding-3-large", dimensions=1536)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


st.title("💬 knowledge_base")
st.caption("🚀 A streamlit knowledge_base powered by LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]



for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if custom_openai_api_key:
        if selected_option=='zhipuai':
            chat = ChatZhipuAI(api_key=custom_openai_api_key,model_name=model_selected_option)
            # client = ZhipuAI(api_key=custom_openai_api_key)  # 填写您自己的APIKey
            # chat = client.chat

        else:
            chat = ChatOpenAI(openai_api_key=custom_openai_api_key, model_name=model_selected_option)
        embedding1024 = OpenAIEmbeddings(openai_api_key=open_ai_key,
                                    model="text-embedding-3-large", dimensions=1024)

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url,supabase_key)





def query_knowledge(query: str) -> str :
    request_content = []
    # rpc的方式
    vector = embedding1024.embed_query(query)
    result2 = supabase.rpc('match_yeeha_documents_v3', {
        "query_embedding": vector,
        "match_count": 10,
        "match_threshold": 0.1
    }).execute()


    for item in result2.data:
        request_content.append(item['content'])


    print('截取before的相似性内容=', request_content)

    docs = '\n'.join([doc for doc in request_content])

    text_splitter = TokenTextSplitter(
        chunk_size=2000, chunk_overlap=0, encoding_name="cl100k_base"
    )

    texts = text_splitter.split_text(docs)
    similar_text = ''
    if len(texts) == 1:
        similar_text = texts[0]
    elif len(texts) > 1:
        similar_text = texts[0] + texts[1]

    token_num = num_tokens_from_string(similar_text, "cl100k_base")
    print('截取after的相似性内容=', similar_text)

    print("input token num=", token_num)
    content = f"'''{similar_text}'''"
    content += f'\n问题：{query}'
    return content

def get_result(prompt: str) -> str:

    content = query_knowledge(prompt)

    # if selected_option == 'zhipuai':
    #     message_list = [
    #         {"role": "system", "content": "我会将文档内容以三引号(''')引起来发送给你。"},
    #         {"role": "user", "content": content},
    #     ]
    # else:
    message_list = [
        SystemMessage(
            content="我会将文档内容以三引号(''')引起来发送给你。"
        ),
        HumanMessage(
            content=content

        ),
    ]
    if selected_option == 'zhipuai':
        # response = client.chat.completions.create(
        #     model=model_selected_option,  
        #     messages=message_list,
        # )
        # msg = response.choices[0].message.content

        msg = chat(messages=message_list).content
    else:
        msg = chat(message_list).content

    return msg


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if not custom_openai_api_key or not supabase_url or not supabase_key:
        st.info("Please add your OpenAI API key and supabase_url and supabase_key to continue.")
        st.stop()

    

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                msg = get_result(prompt)
                placeholder = st.empty()
                # full_response = ''
                # for item in response:
                #     full_response += item
                #     placeholder.markdown(full_response)
                placeholder.markdown(msg)
    message = {"role": "assistant", "content": msg}
    st.session_state.messages.append(message)

    # with get_openai_callback() as cb:
    #     if selected_option == 'zhipuai':
    #         response = client.chat.completions.create(
    #             model=model_selected_option,  
    #             messages=message_list,
    #         )
    #         msg = response.choices[0].message.content
    #     else:
    #         msg = chat(message_list).content
    #     st.session_state.messages.append({"role": "assistant", "content": msg})
    #     st.chat_message("assistant").write(msg)
    #     print(cb)








    # client = OpenAI(api_key=openai_api_key)
    # st.session_state.messages.append({"role": "user", "content": prompt})
    # st.chat_message("user").write(query_content)
    # response = client.chat.completions.create(model="gpt-4-0125-preview", messages=st.session_state.messages)
    # msg = response.choices[0].message.content
    # st.session_state.messages.append({"role": "assistant", "content": msg})
    # st.chat_message("assistant").write(msg)