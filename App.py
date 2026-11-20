"""
====================================================================
과목명: 생성형AI활용프로그래밍 (기말 팀프로젝트 — 사회문제를 해결하는 AI 에이전트)
주제(B): 남서울대학교 엘림생활관 외국인 유학생의 하루(아침·낮·저녁) 한국 & 캠퍼스 적응 도우미
페르소나: 2026년 9월 학기 남서울대학교에 입학하여 엘림생활관(기숙사)에 처음 입사한
         한국어가 서툰 외국인 유학생 (법률·비자·의료·외부 주소 없이 기숙사 규칙 & 한국 초기 생활 주의사항 안내)

구현 기술 (총 3개 기술 연결 — 기본 2개 + 추가 기술 가점 항목 충족):
  1. Local LLM (Ollama: qwen2.5:3b / 학교 GPU 서버용 exaone3.5:7.8b, qwen2.5:7b 전환 지원)
  2. 기술 1 — RAG (HuggingFaceEmbeddings + FAISS: 엘림생활관 수칙 및 한국 생활 적응 가이드 문서 검색)
  3. 기술 2 — 도구 호출 (Tool Calling 3종: 캠퍼스 가이드 RAG 검색, 한국 초기 생활 주의사항 체크리스트 조회, 기숙사 외박 신청 & 한국어 예절 메시지 생성)
  4. 기술 3 — LangGraph (StateGraph 상태 분기: router_node -> clarification_node[모호한 질문 되묻기] / campus_agent_node[RAG+도구 실행])
  5. Gradio 웹 데모 (아침·낮·저녁 시나리오 버튼 + 슬라이드 7·8쪽 필수 테스트 '예외 1·2' 버튼 탑재)
====================================================================
"""

import os
import re
from typing import TypedDict, List, Literal

import gradio as gr
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END


# ====================================================================
# 0. 전역 설정 및 보안·윤리·환각 방지 상수 정의
# ====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, "data", "campus_guide.txt")

# [로컬 LLM 모델 설정]
# 로컬 PC에서는 가벼운 "qwen2.5:3b" 모델을 사용합니다.
# ※ 학교 GPU 서버에서 시연할 때는 아래 주석을 해제하여 EXAONE 또는 Qwen 7B로 쉽게 변경하세요!
MODEL_NAME = "qwen2.5:3b"
# MODEL_NAME = "exaone3.5:7.8b"    # 학교 GPU 서버 권장 모델 1 (한국어 특화 EXAONE)
# MODEL_NAME = "qwen2.5:7b"        # 학교 GPU 서버 권장 모델 2 (Qwen 7B)

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# [보안·윤리·오작동 방지 안내 문구] (슬라이드 3쪽: 최악의 오작동 방지책 & 범위 한정)
SAFETY_NOTICE = (
    "📌 [안내] 본 답변은 외국인 유학생의 한국 생활 및 엘림생활관 적응을 돕기 위한 AI 안내이며, "
    "실제 기숙사 외박 신청 및 행정 절차는 반드시 남서울대학교 공식 학생포털에서 직접 진행해야 합니다."
)

ANTI_HALLUCINATION_SYSTEM_PROMPT = """당신은 남서울대학교 엘림생활관(기숙사)에 처음 입사한 2026년 9월 입학 외국인 유학생을 돕는 친절한 AI 캠퍼스 & 한국 생활 도우미입니다.

[반드시 지켜야 할 규칙]
1. 쉬운 한국어 말투: 한국어가 서툰 외국인 유학생이 이해하기 쉽도록 짧고 친절한 한국어(해요체)와 글머리 기호(-)로만 답변하세요. 중국어나 영어를 섞지 마세요.
2. 문서 근거 기반 답변(환각 방지): 오직 아래 제공된 [도구 실행 및 문서 검색 결과]에 있는 내용만 바탕으로 답변하세요.
3. 문서에 없는 정보 추측 금지: 질문한 시설(예: 기숙사 수영장, 반려동물 사육 등)이나 규칙이 검색 결과에 없으면 절대 지어내지 말고, "죄송해요. 해당 내용은 현재 학교 및 기숙사 안내 문서에서 확인할 수 없습니다. 엘림생활관 행정실(041-580-2600)에 직접 확인해 주세요."라고 솔직하게 답하세요.
4. 의료·법률 제외 및 개인정보 보호: 의료(약 추천/진단)나 법률 자문은 제공하지 않으며, 유학생의 개인정보(학번, 전화번호, 신분증 번호)를 요구하거나 저장하지 마세요.
"""


# ====================================================================
# 1. 기술 1 — RAG (Retrieval-Augmented Generation) 초기화
# ====================================================================
def build_rag_retriever():
    """data/campus_guide.txt 문서를 임베딩하여 FAISS 벡터 검색기를 생성합니다."""
    if not os.path.exists(DATA_FILE_PATH):
        raise FileNotFoundError(
            f"데이터 파일을 찾을 수 없습니다: {DATA_FILE_PATH}\n"
            "data/campus_guide.txt 파일이 존재하는지 확인해 주세요."
        )

    with open(DATA_FILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=120,
        separators=["\n----", "\n====", "\n\n", "\n", " "],
    )
    chunks = text_splitter.split_text(raw_text)
    docs = [
        Document(page_content=chunk, metadata={"source": "data/campus_guide.txt"})
        for chunk in chunks
    ]

    print(f"[RAG 초기화] 다국어 임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # 학교 GPU 서버에서는 "cuda"로 변경 가능
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"[RAG 초기화 완료] 총 {len(docs)}개의 캠퍼스·한국생활 가이드 청크가 FAISS에 인덱싱되었습니다.")
    return vectorstore.as_retriever(search_kwargs={"k": 3})


CAMPUS_RETRIEVER = build_rag_retriever()


# ====================================================================
# 2. 기술 2 — 도구 호출 (Tool Calling / Function Calling) 3종 정의
# ====================================================================
@tool
def search_campus_guide(query: str) -> str:
    """
    [아침 시나리오 및 캠퍼스 공통 RAG 도구]
    남서울대학교 엘림생활관(기숙사) 아침 식사(조식) 시간, 출입 통금 시간(01:00~05:00),
    세탁실 이용 시간(06:00~23:00), 야간 정숙 시간, 반입 금지 물품, 교내 시설 및 스쿨버스 정보를 검색합니다.
    """
    docs = CAMPUS_RETRIEVER.invoke(query)
    if not docs:
        return "안내 문서(RAG)에서 관련 정보를 찾지 못했습니다."

    results = []
    for idx, doc in enumerate(docs, start=1):
        results.append(f"[캠퍼스 가이드 문서 {idx}]\n{doc.page_content.strip()}")
    return "\n\n".join(results)


@tool
def search_korea_life_tips(topic: str) -> str:
    """
    [낮 시나리오 — 한국에 처음 온 유학생을 위한 한국 생활 주의사항 체크리스트 도구]
    한국 입국 초기 유학생이 가장 많이 실수하는 생활 주의사항(쓰레기 분리배출, 버스/지하철 교통카드 하차 태그,
    대학 식당 셀프 반납 및 전자출결, 220V 콘센트 및 기숙사 택배 수령법)을 카테고리별로 상세히 조회합니다.
    topic 예시: '쓰레기 분리배출', '교통카드 버스 환승', '식당 수업 문화', '전압 택배'
    """
    tips_db = {
        "쓰레기": (
            "♻️ [한국 생활 주의사항 1 — 쓰레기 분리배출 규칙]\n"
            "1. 재활용 쓰레기 (플라스틱·페트병·캔·종이상자):\n"
            "   - 페트병과 플라스틱 용기는 내용물을 비우고 물로 깨끗이 헹군 뒤, 겉의 비닐 라벨을 반드시 떼서 버려야 해요.\n"
            "   - 택배 종이상자는 테이프와 운송장 스티커를 모두 제거한 뒤 납작하게 접어서 버려요.\n"
            "2. 음식물 쓰레기 vs 일반 쓰레기 구분 (유학생이 가장 많이 실수하는 점!):\n"
            "   - 부드러운 남은 밥, 반찬, 과일 과육만 '음식물 쓰레기'예요.\n"
            "   - ⚠️ 주의: **달걀 껍데기, 치킨·돼지고기 뼈, 생선 가시, 복숭아·망고의 딱딱한 씨앗, 양파·옥수수 껍질, 티백**은 동물이 먹을 수 없어서 음식물 쓰레기가 아니라 **'일반 쓰레기'**로 버려야 해요!"
        ),
        "교통": (
            "🚌 [한국 생활 주의사항 2 — 대중교통 & 교통카드(T-money) 이용 규칙]\n"
            "1. 버스·전철 내릴 때 '하차 태그' 필수:\n"
            "   - 한국 버스를 탈 때는 탈 때(승차)뿐만 아니라 **내릴 때(하차)도 반드시 출입문 단말기에 교통카드를 찍어야(태그해야)** 해요.\n"
            "   - 내릴 때 카드를 찍지 않으면 30분 이내 다른 버스나 전철로 갈아탈 때 **무료 환승 할인**을 받을 수 없고, 다음 탑승 시 추가 요금(페널티)이 빠져나가요.\n"
            "2. 버스 내 음식물 반입 금지:\n"
            "   - 뚜껑이 없는 테이크아웃 커피잔이나 냄새나는 음식을 들고 시내버스에 탈 수 없어요."
        ),
        "식당_수업": (
            "🍽️ [한국 생활 주의사항 3 — 대학 식당(학식) 및 강의실 수업 문화]\n"
            "1. 학생식당·기숙사 식당 셀프 서비스(Self-Service):\n"
            "   - 물과 기본 반찬(김치, 단무지)은 셀프 코너에서 먹을 만큼만 직접 가져다 먹어요.\n"
            "   - 식사가 끝나면 식탁에 그릇을 두고 가지 말고, 반드시 본인이 먹은 식판과 수저를 **퇴식구(식기 반납 코너)**에 직접 반납해야 해요.\n"
            "2. 대학 강의실 전자출결 및 지각 규정:\n"
            "   - 수업 시작 전에 스마트폰 전자출결 앱(블루투스)으로 출석을 체크하며, **지각 3번은 결석 1번**으로 처리되니 수업 5분 전까지 도착하는 것이 좋아요."
        ),
        "전압_택배": (
            "🔌 [한국 생활 주의사항 4 — 220V 전기 콘센트 & 기숙사 택배 수령]\n"
            "1. 220V 전압 및 둥근 2구 콘센트(돼지코 어댑터):\n"
            "   - 한국은 **220V 전압(60Hz)과 둥근 2구 콘센트(C/F 타입)**를 사용해요. 본국에서 가져온 충전기나 전자제품 플러그 모양이 다르면 변환 어댑터(돼지코)를 끼워야 하고, 110V 전용 제품을 변압기 없이 꽂으면 고장이나 화재 위험이 있으니 꼭 확인하세요.\n"
            "2. 기숙사 택배 주소 작성법:\n"
            "   - 온라인 쇼핑몰 주문 시 주소 끝에 반드시 **'엘림생활관 제1관(또는 제2관) OOO호 + 본인 이름'**을 정확히 적어야 1층 무인 택배보관함에서 분실 없이 찾을 수 있어요."
        ),
    }

    q = topic.lower()
    matched = []
    if any(k in q for k in ["쓰레기", "분리", "재활용", "음식물", "뼈", "달걀", "페트병"]):
        matched.append(tips_db["쓰레기"])
    if any(k in q for k in ["버스", "교통", "카드", "환승", "하차", "태그", "전철", "지하철"]):
        matched.append(tips_db["교통"])
    if any(k in q for k in ["식당", "학식", "식판", "퇴식구", "반납", "수업", "출석", "출결", "지각", "강의"]):
        matched.append(tips_db["식당_수업"])
    if any(k in q for k in ["전기", "전압", "220", "콘센트", "어댑터", "돼지코", "택배", "우편"]):
        matched.append(tips_db["전압_택배"])

    if not matched:
        matched = [
            tips_db["쓰레기"],
            tips_db["교통"],
            tips_db["식당_수업"],
            tips_db["전압_택배"],
        ]

    return "\n\n".join(matched)


@tool
def dorm_and_korean_helper(request_text: str) -> str:
    """
    [저녁 시나리오 — 기숙사 외박 신청 규정 체크 & 한국어 예절 메시지 작성 도구]
    엘림생활관 외박 신청 규정(밤 22:00 마감, 무단외박 벌점, 외부인 출입 금지)을 확인하여 체크리스트를 제공하고,
    기숙사 사감 선생님이나 교수님께 보낼 정중한 한국어 메시지(존댓말 템플릿)와 연락 예절을 안내합니다.
    """
    q = request_text.lower()
    outputs = []

    if any(k in q for k in ["외박", "주말", "토요일", "일요일", "집", "친구", "벌점"]):
        outputs.append(
            "📝 [엘림생활관 외박 신청 규정 & 체크리스트 도구 실행 결과]\n"
            "- 신청 마감 시간: 외박 당일 **밤 22:00(오후 10시) 전**까지 신청 필수!\n"
            "- 주요 주의사항:\n"
            "  * 밤 22:00까지 외박 신청을 하지 않고 통금 시간(밤 01:00) 이후까지 귀관하지 않으면 **무단외박 벌점 5점**이 부과돼요 (누적 15점 이상 시 퇴사).\n"
            "  * 기숙사 호실 안에 외부인(친구나 가족)을 데려오면 절대 안 돼요.\n"
            "- 신청 절차: 남서울대 학생포털 → [생활관] → [외박 신청] → 외박 날짜 및 행선지 입력 후 저장\n"
            "- 💬 [사감 선생님께 드리는 정중한 한국어 인사말 예시]\n"
            "  \"안녕하세요, 사감 선생님! 엘림생활관 OOO호에 거주하는 외국인 유학생 OOO입니다. "
            "학생포털 시스템에 이번 주말 외박 신청을 완료하였습니다. 일요일 저녁 통금 시간(밤 01:00) 전까지 안전하게 귀관하겠습니다. 감사합니다!\""
        )

    if any(k in q for k in ["교수", "수업", "과제", "질문", "메일", "메시지", "한국어", "인사", "예절", "사감"]):
        outputs.append(
            "✉️ [한국 대학 연락 예절 & 정중한 한국어 메시지 템플릿]\n"
            "- 한국 연락 예절 주의사항:\n"
            "  1) 밤 22:00 이후 늦은 밤이나 이른 아침에는 교수님·사감 선생님께 문자나 메신저 연락을 피하는 것이 예의예요.\n"
            "  2) 메시지 첫 줄에는 항상 **[소속 학과 / 학번 또는 기숙사 호실 / 본인 이름]**을 먼저 밝혀야 해요.\n"
            "- [교수님께 수업/과제 질문할 때 템플릿]\n"
            "  \"안녕하세요, 교수님. OOO 강의를 수강 중인 외국인 유학생 OOO(학번)입니다. "
            "한국어가 아직 서툴러 메시지로 여쭙게 되어 죄송합니다. 이번 주 과제 제출 방법과 관련하여 확인하고 싶은 점이 있어 연락드렸습니다. 감사합니다.\""
        )

    if not outputs:
        outputs.append(
            "💬 [캠퍼스 한국어 & 기숙사 외박 도우미]\n"
            "- 엘림생활관 외박 신청은 당일 밤 22:00 전까지 학생포털에서 완료해야 무단외박 벌점(5점)을 받지 않습니다.\n"
            "- 사감 선생님이나 교수님께 연락할 때는 소속(기숙사 호실/학번)과 이름을 먼저 밝히고 정중한 높임말(해요체/하십시오체)을 사용하세요."
        )

    return "\n\n".join(outputs)


# ====================================================================
# 3. 기술 3 — LangGraph 상태(State) 및 노드(Nodes) 설계
#    (슬라이드 4·7쪽: 상태에 따라 다음 작업을 분기하거나 모호한 질문 시 사람에게 되묻기)
# ====================================================================
class AgentState(TypedDict):
    question: str
    intent: str               # "greeting", "clarify", "normal"
    route_path: List[str]     # LangGraph 실행 노드 경로
    used_tools: List[str]     # 실제 호출된 @tool 목록
    tool_context: str         # 도구 실행 결과
    answer: str               # 최종 답변


def get_llm():
    """로컬 Ollama LLM 인스턴스를 반환합니다 (temperature=0.0으로 환각 방지)."""
    return ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,
    )


def sanitize_personal_info(text: str) -> tuple[str, bool]:
    """[보안·개인정보 미수집] 전화번호, 주민/외국인등록번호 등 민감 패턴을 즉시 마스킹합니다."""
    pii_pattern = r"(\b\d{6}[-~\s]?[1-8]\d{6}\b|\b01[016789][-~\s]?\d{3,4}[-~\s]?\d{4}\b)"
    if re.search(pii_pattern, text):
        masked = re.sub(pii_pattern, "[개인정보 마스킹됨]", text)
        return masked, True
    return text, False


def is_greeting_only(text: str) -> bool:
    """단순 인사말인지 판별합니다."""
    cleaned = re.sub(r"[!?.~\s]+", "", text).lower()
    return cleaned in {
        "안녕", "안녕하세요", "안녕하십니까", "반가워", "반가워요", "반갑습니다",
        "하이", "헬로", "hello", "hi", "고마워", "고마워요", "감사합니다",
    }


def is_ambiguous_life_question(text: str) -> bool:
    """
    [슬라이드 4·7쪽 공통 실습 핵심 테스트 케이스 — 모호한 질문 되묻기]
    '한국에 처음 왔는데 주의할 점이 뭐야?' 또는 '한국 생활할 때 뭘 조심해야 해?'처럼
    주의사항이나 규칙을 묻지만 구체적인 주제(쓰레기 분리배출, 버스/교통카드, 식당/수업, 전기/택배, 기숙사 통금/외박 등)를
    전혀 지정하지 않은 포괄적·모호한 질문인지 검사합니다.
    """
    asks_precaution = any(
        k in text for k in ["주의할 점", "주의사항", "조심해야", "유의할 점", "알아야 할 점", "뭘 조심", "무엇을 조심"]
    )
    specific_topics = [
        "쓰레기", "분리", "재활용", "음식물", "버스", "교통", "환승", "태그", "전철",
        "식당", "학식", "퇴식구", "수업", "출결", "전기", "220", "콘센트", "택배",
        "기숙사", "엘림", "통금", "조식", "아침", "세탁", "외박", "사감", "교수"
    ]
    has_specific = any(k in text.lower() for k in specific_topics)
    return asks_precaution and (not has_specific)


# --------------------------------------------------------------------
# Node 1: router_node (질문 상태 분석 및 분기 결정)
# --------------------------------------------------------------------
def router_node(state: AgentState) -> AgentState:
    question = state["question"]
    route_path = list(state.get("route_path", [])) + ["router_node"]

    # 1) 단순 인사말 분기
    if is_greeting_only(question):
        intent = "greeting"
    # 2) 예외 테스트 1: 어떤 주제의 주의사항인지 특정하지 않은 모호한 질문 -> 되묻기 분기
    elif is_ambiguous_life_question(question):
        intent = "clarify"
    else:
        intent = "normal"

    return {
        **state,
        "intent": intent,
        "route_path": route_path,
    }


def route_by_intent(state: AgentState) -> Literal["clarification_node", "campus_agent_node"]:
    """Conditional Edge: 모호한 질문이면 clarification_node로, 일반 요청이면 campus_agent_node로 분기합니다."""
    if state["intent"] == "clarify":
        return "clarification_node"
    return "campus_agent_node"


# --------------------------------------------------------------------
# Node 2: clarification_node (모호한 질문 시 사용자에게 되묻기 / 사람 확인 요청)
# --------------------------------------------------------------------
def clarification_node(state: AgentState) -> AgentState:
    """
    [슬라이드 4쪽 & 7쪽 필수 동작]
    사용자가 '한국에 처음 왔는데 주의할 점이 뭐야?'처럼 주제를 특정하지 않고 질문했을 때,
    임의로 하나만 추측해 답하지 않고 어떤 주제의 주의사항을 원하는지 친절하게 되묻습니다.
    """
    route_path = list(state.get("route_path", [])) + ["clarification_node"]

    clarify_answer = (
        "🤔 **한국에 처음 온 유학생을 위한 주의사항이 여러 분야로 나뉘어 있어요! 어느 항목을 먼저 알려드릴까요?**\n\n"
        "- **① 쓰레기 분리배출 주의사항** (재활용·음식물쓰레기·일반쓰레기 구분법)\n"
        "- **② 대중교통(버스·전철) 교통카드 주의사항** (내릴 때 하차 태그 및 환승 할인 규칙)\n"
        "- **③ 한국 대학 식당 및 수업 주의사항** (식판 셀프 반납 및 전자출결 지각 규정)\n"
        "- **④ 220V 전기 콘센트 및 기숙사 택배 수령 주의사항**\n"
        "- **⑤ 엘림생활관(기숙사) 통금 시간·세탁실·외박 신청 주의사항**\n\n"
        "위 항목 중 궁금한 주제를 골라서 구체적으로 다시 물어봐 주세요 😊"
    )

    return {
        **state,
        "route_path": route_path,
        "used_tools": [],
        "tool_context": "모호한 입력 감지 -> 사용자 확인(되묻기) 요청",
        "answer": clarify_answer,
    }


# --------------------------------------------------------------------
# Node 3: campus_agent_node (아침·낮·저녁 시나리오 RAG + Tool Calling 실행)
# --------------------------------------------------------------------
def campus_agent_node(state: AgentState) -> AgentState:
    question = state["question"]
    route_path = list(state.get("route_path", [])) + ["campus_agent_node"]

    # 단순 인사말 처리
    if state.get("intent") == "greeting":
        return {
            **state,
            "route_path": route_path,
            "used_tools": [],
            "tool_context": "",
            "answer": (
                "안녕하세요! 😊 남서울대학교 엘림생활관 외국인 유학생 AI 도우미예요.\n"
                "- **🌅 아침**: 기숙사 아침 식사 시간, 통금 시간, 세탁실·반입 금지 물품 안내\n"
                "- **☀️ 낮**: 한국에 처음 왔을 때 꼭 알아야 할 생활 주의사항 (분리수거, 버스 환승, 대학 식당·수업 문화)\n"
                "- **🌙 저녁**: 기숙사 외박 신청 방법 및 사감 선생님·교수님께 보낼 정중한 한국어 메시지 작성\n"
                "아래 예시 버튼을 누르거나 궁금한 점을 편하게 물어보세요!"
            ),
        }

    used_tools: List[str] = []
    tool_outputs: List[str] = []

    available_tools = [search_campus_guide, search_korea_life_tips, dorm_and_korean_helper]
    tool_map = {t.name: t for t in available_tools}

    try:
        llm = get_llm()
        llm_with_tools = llm.bind_tools(available_tools)

        # 1단계: 로컬 LLM이 질문에 맞는 도구(Tool Calling)를 선택
        tool_call_msg = llm_with_tools.invoke([
            SystemMessage(content=(
                "유학생의 질문을 해결하기 위해 알맞은 도구를 호출하세요.\n"
                "1. 엘림생활관(기숙사) 조식/통금/세탁실/금지물품 및 캠퍼스 시설 검색 -> search_campus_guide\n"
                "2. 한국 초기 생활 주의사항(쓰레기 분리배출, 버스 교통카드 환승/하차 태그, 식당 셀프 반납, 220V 콘센트/택배) -> search_korea_life_tips\n"
                "3. 기숙사 외박 신청 방법/체크리스트 또는 사감실·교수님께 보낼 정중한 한국어 메시지 작성 -> dorm_and_korean_helper"
            )),
            HumanMessage(content=question),
        ])

        if getattr(tool_call_msg, "tool_calls", None):
            for tc in tool_call_msg.tool_calls:
                t_name = tc["name"]
                if t_name in tool_map and t_name not in used_tools:
                    used_tools.append(t_name)
                    output = tool_map[t_name].invoke(question)
                    tool_outputs.append(f"### [{t_name} 실행 결과]\n{output}")

        # [소형 3B 모델 도구 호출 보완 로직]
        q_lower = question.lower()
        needs_life_tips = any(
            k in q_lower for k in [
                "쓰레기", "분리", "재활용", "음식물", "버스", "교통카드", "환승", "하차",
                "식판", "퇴식구", "셀프", "전자출결", "220", "콘센트", "어댑터", "택배"
            ]
        )
        needs_helper = any(
            k in q_lower for k in ["외박", "인사말", "한국어", "메시지", "사감", "교수님", "예절"]
        )

        if "search_campus_guide" not in used_tools:
            used_tools.insert(0, "search_campus_guide")
            rag_out = search_campus_guide.invoke(question)
            tool_outputs.insert(0, f"### [search_campus_guide 실행 결과]\n{rag_out}")

        if needs_life_tips and "search_korea_life_tips" not in used_tools:
            used_tools.append("search_korea_life_tips")
            tips_out = search_korea_life_tips.invoke(question)
            tool_outputs.append(f"### [search_korea_life_tips 실행 결과]\n{tips_out}")

        if needs_helper and "dorm_and_korean_helper" not in used_tools:
            used_tools.append("dorm_and_korean_helper")
            helper_out = dorm_and_korean_helper.invoke(question)
            tool_outputs.append(f"### [dorm_and_korean_helper 실행 결과]\n{helper_out}")

        combined_context = "\n\n".join(tool_outputs)

        # 2단계: 도구 실행 결과를 바탕으로 쉬운 한국어 최종 답변 생성
        synthesis_prompt = (
            f"[도구 실행 및 문서 검색 결과]\n{combined_context}\n\n"
            f"[유학생의 질문]\n{question}\n\n"
            "[답변 작성 지침]\n"
            "1. 위 [도구 실행 및 문서 검색 결과]에 있는 규칙과 주의사항만 정확히 사용해 답변하세요.\n"
            "2. 만약 질문한 내용(예: 기숙사 반려동물 사육, 기숙사 옥상 수영장 등)이 위 검색 결과에 없다면 절대 추측하거나 지어내지 말고 "
            "'죄송해요. 질문하신 내용은 현재 학교 및 기숙사 안내 문서에서 확인할 수 없습니다. 엘림생활관 행정실(041-580-2600)에 직접 문의해 주세요.'라고 명확히 안내하세요.\n"
            "3. 한국어가 서툰 외국인 유학생이 읽기 쉽게 짧은 문장(해요체)과 글머리 기호(-)로 친절하게 정리해 주세요."
        )

        final_res = llm.invoke([
            SystemMessage(content=ANTI_HALLUCINATION_SYSTEM_PROMPT),
            HumanMessage(content=synthesis_prompt),
        ])
        body_answer = final_res.content.strip()

    except Exception as e:
        if not tool_outputs:
            used_tools = ["search_campus_guide"]
            tool_outputs = [search_campus_guide.invoke(question)]
        combined_context = "\n\n".join(tool_outputs)
        body_answer = (
            f"질문하신 내용에 대해 검색된 캠퍼스 및 한국 생활 가이드를 정리해 드릴게요 😊\n\n"
            f"{combined_context}\n\n"
            f"*(참고: 로컬 Ollama `{MODEL_NAME}` 연결 확인이 필요합니다: {e})*"
        )

    final_answer = f"{body_answer}\n\n---\n{SAFETY_NOTICE}"

    return {
        **state,
        "route_path": route_path,
        "used_tools": used_tools,
        "tool_context": combined_context,
        "answer": final_answer,
    }


# ====================================================================
# 4. LangGraph 워크플로우(StateGraph) 빌드 및 컴파일
# ====================================================================
def build_langgraph_app():
    workflow = StateGraph(AgentState)

    # 노드 등록
    workflow.add_node("router_node", router_node)
    workflow.add_node("clarification_node", clarification_node)
    workflow.add_node("campus_agent_node", campus_agent_node)

    # 시작 -> router_node
    workflow.add_edge(START, "router_node")

    # 상태 기반 조건부 분기 (Conditional Edges)
    workflow.add_conditional_edges(
        "router_node",
        route_by_intent,
        {
            "clarification_node": "clarification_node",
            "campus_agent_node": "campus_agent_node",
        },
    )

    # 종료 연결
    workflow.add_edge("clarification_node", END)
    workflow.add_edge("campus_agent_node", END)

    return workflow.compile()


LANGGRAPH_APP = build_langgraph_app()


# ====================================================================
# 5. Gradio 웹 데모 인터페이스 구성 (슬라이드 3·6·7·8쪽 시연 요건 충족)
# ====================================================================
def chat_fn(message: str, history: list) -> str:
    """Gradio 채팅 입력을 받아 LangGraph를 실행하고, 하단에 실행 경로와 호출된 도구를 표시합니다."""
    if not message or not message.strip():
        return "질문을 입력해 주세요 😊"

    # 개인정보 마스킹 검사 (슬라이드 3·5쪽: 개인정보 미수집·미저장)
    clean_message, had_pii = sanitize_personal_info(message.strip())
    pii_notice = (
        "🔒 **[개인정보 보호 알림]** 입력하신 내용에 개인식별정보(전화번호/등록번호 등) 패턴이 감지되어 즉시 마스킹 처리되었으며 저장되지 않았습니다.\n\n"
        if had_pii else ""
    )

    initial_state: AgentState = {
        "question": clean_message,
        "intent": "",
        "route_path": ["START"],
        "used_tools": [],
        "tool_context": "",
        "answer": "",
    }

    result_state = LANGGRAPH_APP.invoke(initial_state)

    full_path = " → ".join(result_state["route_path"] + ["END"])
    tools_str = ", ".join(result_state["used_tools"]) if result_state["used_tools"] else "도구 미호출 (되묻기/인사 분기)"
    intent_badge = (
        "❓ 모호한 질문 되묻기 분기 (Clarify)"
        if result_state["intent"] == "clarify"
        else "🏫 한국생활·기숙사 에이전트 분기 (RAG + Tool)"
    )

    # 슬라이드 7쪽 '확인 방법: 최종 답변만 보지 말고 실제로 호출된 도구를 함께 확인하세요' 충족
    trace_log = (
        f"\n\n---\n"
        f"🔧 **LangGraph 실행 경로 & 호출 도구**: `[{intent_badge}]` `{full_path}` | "
        f"**호출된 도구(@tool)**: `{tools_str}` | **로컬 LLM**: `{MODEL_NAME}`"
    )

    return f"{pii_notice}{result_state['answer']}{trace_log}"


# [슬라이드 3쪽 '하루 시나리오(아침·낮·저녁)' + 슬라이드 7·8쪽 '예외 1·2 테스트' 버튼]
EXAMPLE_QUESTIONS = [
    "🌅 [아침 시나리오 - 기숙사 규칙] 엘림생활관 아침 식사(조식) 시간과 기숙사 통금 시간, 세탁실 이용 시간 및 반입 금지 물품을 알려주세요.",
    "☀️ [낮 시나리오 - 한국 생활 주의사항] 한국에 처음 온 유학생인데 쓰레기 분리배출할 때 주의할 점과 버스 교통카드(환승) 이용할 때 꼭 지켜야 할 점을 알려주세요.",
    "🌙 [저녁 시나리오 - 외박 신청 & 한국어 예절] 이번 주말에 엘림생활관 외박 신청을 하고 싶어요. 신청 마감 시간과 사감 선생님께 드릴 정중한 한국어 인사말을 알려주세요.",
    "🧪 [예외 테스트 1 - 모호한 질문 되묻기] 한국에 처음 왔는데 주의할 점이 뭐야?",
    "🧪 [예외 테스트 2 - 문서에 없는 내용(환각 방지)] 남서울대학교 기숙사에서 강아지나 고양이를 키워도 되나요? 기숙사 옥상 수영장 이용 시간도 알려주세요.",
]

CUSTOM_CSS = """
.gradio-container { max-width: 980px !important; margin: auto; }
.header-box {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
    color: white;
    padding: 20px 24px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.header-box h1 { color: white !important; margin: 0 0 8px 0; font-size: 1.45rem; }
.header-box p { color: #e0e7ff !important; margin: 0; font-size: 0.93rem; line-height: 1.55; }
"""

with gr.Blocks(css=CUSTOM_CSS, title="남서울대학교 유학생 AI 한국생활 & 기숙사 도우미") as demo:
    gr.HTML(
        f"""
        <div class="header-box">
            <h1>🏫 남서울대학교 엘림생활관 외국인 유학생 AI 한국 생활 & 기숙사 도우미</h1>
            <p>
                <b>사용자 & 하루 시나리오:</b> 2026년 9월 입학 엘림생활관 거주 외국인 유학생의 <b>아침(기숙사 규칙) · 낮(한국 초기 생활 주의사항) · 저녁(외박 신청 & 한국어 예절)</b><br>
                <b>연결 기술 (3종):</b> Local LLM (Ollama <code>{MODEL_NAME}</code>) + <b>RAG</b> (FAISS 문서 검색) + <b>도구 호출</b> (한국 생활 체크리스트·외박 도우미) + <b>LangGraph</b> (모호한 질문 되묻기 분기)
            </p>
        </div>
        """
    )

    gr.ChatInterface(
        fn=chat_fn,
        examples=EXAMPLE_QUESTIONS,
        title=None,
        description=(
            "아래 **아침·낮·저녁 시나리오 버튼** 또는 **예외 테스트 1·2 버튼**(슬라이드 7·8쪽 평가 기준)을 클릭해 시연해 보세요! (개인정보 미수집)"
        ),
        chatbot=gr.Chatbot(
            height=480,
            placeholder=(
                "👋 안녕하세요! 남서울대학교 엘림생활관 유학생 AI 도우미입니다.\n"
                "기숙사 조식·통금·세탁 규칙(아침), 한국 생활 필수 주의사항(낮), 외박 신청 및 한국어 예절 메시지 작성(저녁)을 도와드려요!"
            ),
        ),
        textbox=gr.Textbox(
            placeholder="예: 한국에서 쓰레기 분리배출할 때 뭘 조심해야 해? / 버스 탈 때 교통카드 어떻게 써? / 기숙사 통금 시간이 언제야?",
            container=False,
            scale=7,
        ),
    )


if __name__ == "__main__":
    # [실행 환경 설정]
    # 1. 로컬 PC(Windows / Linux) 실행 시: server_name="127.0.0.1"
    # 2. 학교 GPU 서버에서 실행 시: server_name="0.0.0.0", share=True 로 변경
    demo.launch(
        server_name="127.0.0.1",  # 학교 GPU 서버 실행 시: server_name="0.0.0.0", share=True 로 변경
        server_port=7860,
        share=False,
    )
