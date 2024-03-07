import ast
from uuid import uuid4
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
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import PostgresChatMessageHistory
from langchain.chains import ConversationChain



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
    st.session_state['session_id'] = uuid4
session_id = st.session_state['session_id']


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


st.title("💬 AI版张老师")
st.caption("🚀 A streamlit chatbot powered by LLM")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]


def get_school_name(question):
    nlp = spacy.load("zh_core_web_md")
    # Process the sentence with spaCy
    doc = nlp(question)

    # Extract organization entities
    school_name_list = []
    year_list = []
    filter_dict_list = []
    year = None
    for ent in doc.ents:
        if ent.label_ == "ORG":
           school_name_list.append(ent.text)
        if ent.label_ == "DATE":
            year = ent.text
            year = year.replace('年','')
            year_list.append(year)
    if len(school_name_list)>0:
        for school_name in school_name_list:
            if len(year_list)>0:
                for year in year_list:
                    filter_dict_list.append({"school_name":school_name,"year":year})
        return filter_dict_list
    elif len(year_list)>0:
        for year in year_list:
            if len(school_name_list)>0:
                for school_name in school_name_list:
                    filter_dict_list.append({"school_name":school_name,"year":year})
        return filter_dict_list
    else:
        return [{}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if custom_openai_api_key:
        if selected_option=='zhipuai':
            client = ZhipuAI(api_key=custom_openai_api_key)  # 填写您自己的APIKey
            chat = client.chat
        else:
            chat = ChatOpenAI(openai_api_key=custom_openai_api_key, model_name=model_selected_option)
        embedding1536 = OpenAIEmbeddings(openai_api_key=open_ai_key,
                                    model="text-embedding-3-large", dimensions=1536)

if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url,supabase_key)


def queryKnowedge(query):



    history = PostgresChatMessageHistory(
        connection_string=connection_string,
        session_id=session_id,
        # table_name='history_messages'
    )



    system_msg_template = SystemMessagePromptTemplate.from_template(
        template="""根据input和history中的Human message，制定一个最相关的问题。不是让你回答问题，是让你生成一个问题，只返回问题本身，不要返回其它内容""")

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages(
        [system_msg_template,
         MessagesPlaceholder(variable_name="history"),
         human_msg_template])

    conversation_with_memory = ConversationChain(
        llm=chat,
        prompt=prompt_template,
        memory=ConversationSummaryBufferMemory(llm=chat, max_token_limit=2000,
                                               chat_memory=history, return_messages=True),
        verbose=True
    )

    response = conversation_with_memory.predict(input=query)
    print('制定的相关问题',response,query)
    query = response
    request_content = []
    info_source = []
    filter_condition = get_school_name(query)
    # if selected_option == 'zhipuai':
    #     filter_condition = get_school_name_and_year_by_zhipu(query)
    # else:
    #     filter_condition = get_school_name_and_year_by_openai(query)

    # 查询知识库
    if filter_condition:
        result2 = supabase.rpc('match_documents_v3', {
                "query_embedding": embedding1536.embed_query(query),
                "filter": filter_condition,
                "match_count": 4,
                "match_threshold": 0.1
            }).execute()
    else:
        result2 = supabase.rpc('match_documents_v3', {
        "query_embedding": embedding1536.embed_query(query),
        "filter":[{}],
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

def get_result_chain(prompt:str):
    content = queryKnowedge(prompt)

    system_msg_template = SystemMessagePromptTemplate.from_template(template="""我会将文档内容以三引号(''')引起来发送给你。请使用中文回答问题。""")


    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    history = PostgresChatMessageHistory(
        connection_string="postgres://postgres.dusjejrppguxkatsyzkp:Jy*#357692577@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres",
        session_id="session_id",
        # table_name='history_messages'
    )

    conversation_with_memory = ConversationChain(
        llm=chat,
        prompt=prompt_template,
        memory=ConversationSummaryBufferMemory(llm=chat, max_token_limit=2000, chat_memory=history),
        verbose=True
    )

    response = conversation_with_memory.predict(input=content)
    return response

    # conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)



def get_result(prompt: str) -> str:

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
    if selected_option == 'zhipuai':
        response = client.chat.completions.create(
            model=model_selected_option,  
            messages=message_list,
        )
        msg = response.choices[0].message.content
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
                msg = get_result_chain(prompt)
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