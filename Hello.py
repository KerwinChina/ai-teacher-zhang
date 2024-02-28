import ast
from openai import OpenAI
import streamlit as st
from supabase import create_client, Client
import tiktoken
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback



with st.sidebar:
    custom_openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    supabase_url = st.text_input("supabase URL", key="supabase URL", type="password")
    supabase_key = st.text_input("supabase KEY", key="supabase KEY", type="password")


if custom_openai_api_key:
    chat = ChatOpenAI(openai_api_key=custom_openai_api_key, model_name="gpt-4-0125-preview")
    embedding1536 = OpenAIEmbeddings(openai_api_key=custom_openai_api_key
                                 ,
                                 model="text-embedding-3-large", dimensions=1536)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def get_school_name_and_year(question):

    content = f"'''{question}'''"
    # content += f'\n问题：按照如下的格式给出：{{"school_name":"大学名称","year":"年份"}}，不要输出其它结果'
    messages = [
        SystemMessage(
            content="找出文档内容中的大学名称和年份，我会将文档内容以三引号(''')引起来发送给你，按照如下的格式给出：{'school_name':'大学名称','year':'年份'}"
                    "，如果没有年份，就不要输出year字段，"
                    "如果没有大学名称，就不要输出school_name字段，如果年份和大学名称都没有，就直接返回{}，不要输出其它无关的内容"
        ),
        HumanMessage(
            content=content

        ),
    ]
    return chat(messages).content

st.title("💬 Chatbot")
st.caption("🚀 A streamlit chatbot powered by OpenAI LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url,supabase_key)



for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def queryKnowedge(query):
    request_content = []
    info_source = []
    filter_condition = get_school_name_and_year(query)
    # 查询知识库
    result2 = supabase.rpc('match_documents_v3', {
            "query_embedding": embedding1536.embed_query(query),
            "filter": ast.literal_eval(filter_condition),
            "match_count": 4,
            "match_threshold": 0.1
        }).execute()
    if result2 and len(result2.data) > 0:
        info_source.append(result2.data[0]['metadata']['info_source'])

    for item in result2.data:
        request_content.append(item['content'])

    # request_content = list(set(request_content))

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
        similar_text = texts[0]+texts[1]

    token_num = num_tokens_from_string(similar_text, "cl100k_base")
    print('截取after的相似性内容=', similar_text)

    print("input token num=", token_num)
    content = f"'''{similar_text}'''"
    content += f'\n问题：{query}'
    return content



if prompt := st.chat_input():
    if not custom_openai_api_key or not supabase_url or not supabase_key:
        st.info("Please add your OpenAI API key and supabase_url and supabase_key to continue.")
        st.stop()
    content = queryKnowedge(prompt)


    message_list = [
        SystemMessage(
            content="我会将文档内容以三引号(''')引起来发送给你。请使用中文回答问题。"
        ),
        HumanMessage(
            content=content

        ),
    ]
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with get_openai_callback() as cb:
        msg = chat(message_list).content
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
        print(cb)








    # client = OpenAI(api_key=openai_api_key)
    # st.session_state.messages.append({"role": "user", "content": prompt})
    # st.chat_message("user").write(query_content)
    # response = client.chat.completions.create(model="gpt-4-0125-preview", messages=st.session_state.messages)
    # msg = response.choices[0].message.content
    # st.session_state.messages.append({"role": "assistant", "content": msg})
    # st.chat_message("assistant").write(msg)