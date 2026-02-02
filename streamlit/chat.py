import streamlit as st

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

st.set_page_config(page_title="소득세 챗봇", page_icon="🤖")

st.title("소득세 챗봇 🤖") #제목 
st.caption("소득세 관련 질문을 해보세요.")  #캡션 설명 

# st.chat_input(placeholder="질문을 입력하세요.") #채팅 입력 창 
# st.chat_message("user"): #사용자 메시지 창 
# st.chat_message("ai"): #ai 메시지 창 
# st.chat_message("assistant"): #assistant 메시지 창 
# st.chat_message("system"): #system 메시지 창 
# st.chat_message("error"): #error 메시지 창 

# with 문을 사용하면 아래 들여쓰기에 있는 내용을 이 창 안에 넣어줌
# with st.chat_message("user"): #사용자 메시지 창 
#     st.write("Hello, how are you?") #사용자 메시지 창에 메시지 출력

#st.session_state
#streamlit은 채팅을 입력할 때마다 코드가 전체적으로 다시 실행된다.
#st.session_state는 코드가 다시 실행되어도 데이터를 유지해주는 특수 저장소. (새로고침 전까지 히스토리 유지)

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

#기존 채팅 기록 출력
for message in st.session_state.message_list: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

def get_ai_message(user_message):
    # 1. 임베딩, 벡터 DB 객체 생성 
    load_dotenv()
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    index_name = 'index-2'
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding  # embedding 객체는 필요
    )

    # 2. 쿼리 변환 체인 생성
    llm = ChatOpenAI(model="gpt-4o-mini")

    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    query_transform_prompt = ChatPromptTemplate.from_template(
        """사용자의 질문을 보고, 키워드 사전을 참고해서 사용자의 질문을 변경해주세요. 
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다. 
        그런 경우에는 질문만 리턴해주세요.
        사전: {dictionary}
        사용자의 질문: {question}
        """
    )
    query_transform_chain = (
        query_transform_prompt 
        | llm 
        | StrOutputParser()
    )

    # 3. RAG 검색 체인 생성
    def format_docs(docs):
        """검색된 문서들을 하나의 context 문자열로 포맷팅"""
        return "\n\n---\n\n".join([doc.page_content for doc in docs])

    retriever = database.as_retriever(
        search_kwargs={"k": 3}
    )
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 최고의 한국 소득세 전문가입니다. 
                    주어진 context를 기반으로 질문에 답변하세요."""
        ),
        ("human", """Context: {context}
                    Question: {question}"""
        )
    ])
    rag_chain = (
        {
            "context": retriever | format_docs,  # 검색 후 포맷팅
            "question": RunnablePassthrough()     # 질문 그대로 전달
        }
        | rag_prompt                              # 프롬프트 생성
        | llm                                     # LLM 호출
        | StrOutputParser()                       # 문자열 추출
    )

    # 4. 전체 체인 생성
    full_chain = query_transform_chain | rag_chain

    ai_message = full_chain.stream({
        "question": user_message, 
        "dictionary": dictionary
    })
    return ai_message

#채팅 입력 창에 질문이 입력되면 새로운 메세지 창 추가 
if user_question := st.chat_input(placeholder="질문을 입력하세요."): 
    with st.chat_message("user"): # 사용자 메시지 창 생성 
        st.write(user_question)  
    st.session_state.message_list.append({"role": "user", "content": user_question}) #채팅 기록 추가

    ai_message = get_ai_message(user_question)
    with st.chat_message("ai"): # ai 메시지 창 생성 
        st.write(ai_message)   
    st.session_state.message_list.append({"role": "ai", "content": ai_message}) #채팅 기록 추가
