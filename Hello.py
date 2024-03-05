import ast
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
from zhipuai import ZhipuAI



with st.sidebar:
    selected_option = st.selectbox('请选择一个选项', ['zhipuai', 'openai'])
    if selected_option == 'zhipuai':
        model_selected_option = st.selectbox('请选择一个选项', ['glm-4', 'glm-3-turbo'])
    else:
        model_selected_option = st.selectbox('请选择一个选项', ['gpt-4-0125-preview', 'gpt-3.5-turbo-0125'])
    custom_openai_api_key = st.text_input("API Key", key="chatbot_api_key", type="password")
    # supabase_url = st.text_input("supabase URL", key="supabase URL", type="password")
    # supabase_key = st.text_input("supabase KEY", key="supabase KEY", type="password")
    supabase_url = st.secrets["supabase_url"]
    supabase_key = st.secrets["supabase_key"]
    open_ai_key = st.secrets["open_ai_key"]

if custom_openai_api_key:
    if selected_option=='zhipuai':
        client = ZhipuAI(api_key=custom_openai_api_key)  # 填写您自己的APIKey
    else:
        chat = ChatOpenAI(openai_api_key=custom_openai_api_key, model_name=model_selected_option)
    embedding1536 = OpenAIEmbeddings(openai_api_key=open_ai_key
                                 ,
                                 model="text-embedding-3-large", dimensions=1536)

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


st.title("💬 AI版张老师")
st.caption("🚀 A streamlit chatbot powered by LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url,supabase_key)

def get_school_name(question):
    nlp = spacy.load("zh_core_web_md")
    # Process the sentence with spaCy
    doc = nlp(question)

    # Extract organization entities
    school_name = ''
    year = None
    for ent in doc.ents:
        if ent.label_ == "ORG":
           school_name = ent.text
        if ent.label_ == "DATE":
            year = ent.text
            year = year.replace('年','')
    if not year:
        return "{'school_name':'"+school_name+"'}"
    else:
        return "{'school_name':'"+school_name+"','year':'"+year+"'}"


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


def queryKnowedge(query):
    request_content = []
    info_source = []
    filter_condition = get_school_name(query)
    # if selected_option == 'zhipuai':
    #     filter_condition = get_school_name_and_year_by_zhipu(query)
    # else:
    #     filter_condition = get_school_name_and_year_by_openai(query)

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
        request_content.append(item['metadata']['school_name']+"; "+item['content'])

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

    if selected_option == 'zhipuai':
        message_list = [
            {"role": "system", "content": "我会将文档内容以三引号(''')引起来发送给你。请使用中文回答问题。"},
            {"role": "user", "content": content},
        ]
    else:
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
        if selected_option == 'zhipuai':
            response = client.chat.completions.create(
                model=model_selected_option,  
                messages=message_list,
            )
            msg = response.choices[0].message.content
        else:
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