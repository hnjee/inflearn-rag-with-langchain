import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

def get_llm(model='gpt-4o-mini'):
    llm = ChatOpenAI(model=model)
    return llm

def get_retriever():
    load_dotenv()
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    index_name = 'index-2'
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    database = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding  # embedding 객체는 필요
    )
    retriever = database.as_retriever(
        search_kwargs={"k": 3}
    )
    return retriever


def get_query_transform_chain():
    llm = get_llm()

    query_transform_prompt = ChatPromptTemplate.from_template(
        """사용자의 질문을 보고, 키워드 사전을 참고해서 사용자의 질문을 변경해주세요. 
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다. 
        그런 경우에는 질문만 리턴해주세요.
        사전: ["사람을 나타내는 표현 -> 거주자"]
        사용자의 질문: {question}
        """
    )
    query_transform_chain = (
        query_transform_prompt 
        | llm 
        | StrOutputParser()
    )
    return query_transform_chain


def format_docs(docs):
    """검색된 문서들을 하나의 context 문자열로 포맷팅"""
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


def get_rag_chain():
    llm = get_llm()
    retriever = get_retriever()

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
    return rag_chain
    
def get_ai_message(user_message):
    query_transform_chain = get_query_transform_chain()
    rag_chain = get_rag_chain()
    
    full_chain = query_transform_chain | rag_chain

    ai_message = full_chain.stream({
        "question": user_message
    })

    return ai_message