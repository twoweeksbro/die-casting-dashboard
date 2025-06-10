import streamlit as st
import sqlite3
from datetime import datetime,date
import os
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1) DB 연결 및 테이블 생성 함수
# ─────────────────────────────────────────────────────────────────────────────
def get_connection():
    return sqlite3.connect("notice.db", check_same_thread=False)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS notices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_pinned INTEGER DEFAULT 0,
            tag TEXT,
            updated_at TEXT,
            file_path TEXT,
            writer TEXT DEFAULT '관리자'
            -- 추가 필드: is_pinned (고정 여부), tag (태그), updated_at (수정 시간), file_path (첨부 파일 경로), writer (작성자
            -- 기본값은 '관리자'로 설정, 필요시 사용자 인증 시스템과 연동 가능
        )
        """
    )
    conn.commit()
    conn.close()

create_table()

# ─────────────────────────────────────────────────────────────────────────────
# 2) CRUD 함수
# ─────────────────────────────────────────────────────────────────────────────

upload_dir = 'uploads'

def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        file_path = os.path.join(upload_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None


def insert_notice(title, content, is_pinned=0, tag=None, file_path=None, writer=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO notices (title, content, created_at, is_pinned, tag, file_path, writer, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (title, content, now, is_pinned, tag, file_path, writer, now),
    )
    conn.commit()
    conn.close()

def get_notices():
    conn = get_connection()
    cursor = conn.cursor()
    # 중요글 우선, 최신순
    cursor.execute(
        """
        SELECT id, title, content, created_at, is_pinned, tag, file_path
        FROM notices
        ORDER BY is_pinned DESC, created_at DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_notice_by_id(nid):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, title, content, created_at, is_pinned, tag, file_path, writer, updated_at
        FROM notices WHERE id = ?
        """,(nid,)
    )
    row = cursor.fetchone()
    conn.close()
    return row # (id, title, content, created_at, is_pinned, tag, file_path, writer, updated_at)

def update_notice(nid, title, content, is_pinned, tag, file_path, writer):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE notices
        SET title = ?, content = ?, is_pinned = ?, tag = ?, file_path = ?, writer = ?, updated_at = ?
        WHERE id = ?
        """,
        (title, content, is_pinned, tag, file_path, writer, now, nid)
    )
    conn.commit()
    conn.close()

def delete_notice(nid):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notices WHERE id = ?", (nid,))
    conn.commit()
    conn.close()

# ─────────────────────────────────────────────────────────────────────────────
# 3) 페이지 UI 및 스타일링
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # --- 페이지 상단 CSS 삽입 (Customize) ---
    st.set_page_config(page_title="공지사항", layout="wide")
    # --- 스타일: 글로벌 CSS ---
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: 'NanumSquare', 'Pretendard', 'Apple SD Gothic Neo', 'sans-serif';
            background: #f5f7fa;
        }
        .notice-banner {
            background: linear-gradient(90deg, #5E7A8A 90%, #FFEB3B 100%);
            color: #fff;
            border-radius: 16px;
            padding: 1.4rem 2rem;
            font-size: 1.15rem;
            font-weight: bold;
            box-shadow: 0 6px 24px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
            letter-spacing: 0.02em;
        }
        .notice-card {
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.4rem;
            transition: box-shadow 0.2s;
            border: 2px solid #EAF1F8;
        }
        .notice-card:hover {
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            border-color: #e6e100;
        }
        .notice-header {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: #2A3F4A;
            letter-spacing: 0.01em;
        }
        .notice-date {
            font-size: 0.92rem;
            color: #aaa;
            margin-bottom: 0.6rem;
            text-align: right;
        }
        .notice-content {
            font-size: 1.02rem;
            color: #222;
            margin-top: 0.3rem;
            word-break: break-all;
        }
        .new-badge {
            display: inline-block;
            background: #F3E22F;
            color: #222;
            border-radius: 8px;
            padding: 2px 10px;
            font-size: 0.88rem;
            font-weight: bold;
            margin-left: 10px;
            vertical-align: middle;
        }
        .stButton > button {
            min-width: 120px;
            margin-right: 0.5rem;
            background: #2A3F4A;
            color: #fff;
            border-radius: 8px;
            border: none;
            font-weight: bold;
            font-size: 1rem;
            padding: 0.6em 1.4em;
        }
        .stButton > button:hover {
            background: #f7cb20;
            color: #222;
        }
        .btn-delete { background: #e74c3c; }
        .btn-edit   { background: #3498db; }
        .btn-back   { background: #95a5a6; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if 'page_mode' not in st.session_state:
        st.session_state.page_mode = 'list'
    if 'selected_notice' not in st.session_state:
        st.session_state.selected_notice = None


    st.title("공지 사항")
    st.markdown('<div class="notice-banner"> 6월 12일(수) 서버 점검 예정 - 오전 6:00~8:00 서비스가 중단됩니다.</div>', unsafe_allow_html=True)
    with st.expander("새 공지 등록", expanded=False):
        with st.form(key="notice_form"):
            title = st.text_input("제목", placeholder="공지 제목을 입력하세요")
            content = st.text_area("내용", height=120, placeholder="공지 내용을 입력하세요 ")
            is_pinned = st.checkbox("중요(상단 고정)", value=False)
            tag = st.text_input("태그 (예: [공지], [업데이트], [이벤트] 등)", placeholder="[공지]")
            uploaded_file = st.file_uploader("첨부파일 (선택)", type=["jpg", "jpeg", "png", "pdf", "xlsx", "csv"])
            writer = st.text_input("작성자", placeholder="이름/아이디 (선택)")
            submit = st.form_submit_button("등록")
            if submit:
                if not title or not content:
                    st.error("제목과 내용을 모두 입력해주세요.")
                else:
                    file_path = save_uploaded_file(uploaded_file) if uploaded_file else None
                    insert_notice(title, content, int(is_pinned), tag, file_path, writer)
                    st.success("공지사항이 등록되었습니다.")
                    st.rerun()

    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""
    if 'tag_filter' not in st.session_state:
        st.session_state.tag_filter = "전체"
    if 'is_pinned_filter' not in st.session_state:
        st.session_state.is_pinned_filter = "전체"
    if 'sort_option' not in st.session_state:
        st.session_state.sort_option = "최신순"

    if st.session_state.page_mode == "list":
        st.subheader("공지 목록")

        # 1. 🔥 [검색/필터/정렬 UI] ---------------------
        tag_list = list({row[5] for row in get_notices() if row[5]})  # 고유 tag 목록
        cols = st.columns([3, 2, 2, 2])

        with cols[0]:
            prev_query = st.session_state.get("search_query", "")
            new_query = st.text_input("검색어 (제목/내용/작성자/태그)", value=prev_query, key="search_query_input")
            if new_query != prev_query:
                st.session_state.search_query = new_query
                st.session_state.current_page = 1  # 변경시 첫 페이지

        with cols[1]:
            prev_tag = st.session_state.get("tag_filter", "전체")
            new_tag = st.selectbox("태그", options=["전체"] + tag_list,
                                index=(["전체"] + tag_list).index(prev_tag) if prev_tag in (["전체"] + tag_list) else 0,
                                key="tag_filter_select")
            if new_tag != prev_tag:
                st.session_state.tag_filter = new_tag
                st.session_state.current_page = 1

        with cols[2]:
            prev_pin = st.session_state.get("is_pinned_filter", "전체")
            new_pin = st.selectbox("중요공지", options=["전체", "중요만", "일반만"],
                                index=["전체", "중요만", "일반만"].index(prev_pin),
                                key="is_pinned_filter_select")
            if new_pin != prev_pin:
                st.session_state.is_pinned_filter = new_pin
                st.session_state.current_page = 1

        with cols[3]:
            prev_sort = st.session_state.get("sort_option", "최신순")
            new_sort = st.selectbox("정렬", options=["최신순", "과거순", "제목순"],
                                    index=["최신순", "과거순", "제목순"].index(prev_sort),
                                    key="sort_option_select")
            if new_sort != prev_sort:
                st.session_state.sort_option = new_sort
                st.session_state.current_page = 1
        

        # 2. 🔥 [공지 필터링/정렬 처리] ---------------------
        notices = get_notices()
        filtered = []

        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1
        
        #POST_PER_PAGE = 10  # 페이지당 공지사항 수


        for notice in notices:
            nid, title, content, created_at, is_pinned, tag, file_path = notice
            q = st.session_state.search_query.strip()
            if q and not (q in title or q in content or q in (tag or "")):
                continue
            if st.session_state.tag_filter != "전체" and tag != st.session_state.tag_filter:
                continue
            if st.session_state.is_pinned_filter == "중요만" and not is_pinned:
                continue
            if st.session_state.is_pinned_filter == "일반만" and is_pinned:
                continue
            filtered.append(notice)

        # 정렬
        if st.session_state.sort_option == "최신순":
            filtered = sorted(filtered, key=lambda x: x[3], reverse=True)
        elif st.session_state.sort_option == "과거순":
            filtered = sorted(filtered, key=lambda x: x[3])
        elif st.session_state.sort_option == "제목순":
            filtered = sorted(filtered, key=lambda x: x[1])

        POST_PER_PAGE = 10
        total = len(filtered)
        total_pages = math.ceil(total / POST_PER_PAGE)

        # 현재 페이지 유지 (검색/필터 변경 시 1페이지로 리셋 권장)
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 1

        start = (st.session_state.current_page - 1) * POST_PER_PAGE
        end = start + POST_PER_PAGE
        page_items = filtered[start:end]



        # 3. 🔥 [카드 for문 → filtered로 변경] ---------------------
        if not filtered:
            st.info("조건에 맞는 공지사항이 없습니다.")
        else:
            for nid, title, content, created_at, is_pinned, tag, file_path in page_items:

                is_today = (created_at[:10] == date.today().strftime("%Y-%m-%d"))
                badges = []
                if is_pinned:
                    badges.append('<span style="background:#ffe066;color:#222;padding:2px 8px;border-radius:8px;font-size:0.88em;font-weight:bold;margin-right:5px;">📌 중요</span>')
                if tag:
                    badges.append(f'<span style="background:#e9ecef;color:#333;padding:2px 8px;border-radius:8px;font-size:0.88em;margin-right:5px;">{tag}</span>')
                if is_today:
                    badges.append('<span style="background:#e64980;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.88em;font-weight:bold;">NEW</span>')
                if file_path:
                    badges.append('<span title="첨부파일 있음" style="margin-left:7px;font-size:1.05em;">📎</span>')
                badge_str = " ".join(badges)
                content_preview = content.replace("\n", " ")[:65] + ("..." if len(content) > 65 else "")
                with st.container():
                    cols = st.columns([7, 2, 2, 1])
                    with cols[0]:
                        # 제목 버튼(상세진입)
                        if st.button(title, key=f"title_{nid}"):
                            st.session_state.selected_notice = nid
                            st.session_state.page_mode = "detail"
                        st.markdown(f"{badge_str}", unsafe_allow_html=True)
                        st.markdown(
                            f"<div style='font-size:0.98em;color:#444;margin-top:0.3em;'>{content_preview}</div>",
                            unsafe_allow_html=True
                        )
                    with cols[1]:
                        st.markdown(f"<span style='color:#888;font-size:0.98em;'>{created_at[:10]}</span>", unsafe_allow_html=True)
                    with cols[2]:
                        if st.button("상세보기", key=f"detail_{nid}"):
                            st.session_state.selected_notice = nid
                            st.session_state.page_mode = "detail"
                    with cols[3]:
                        pass  # (추후) 삭제/수정 자리
            
            # --- 스마트 페이지네이션 UI ----------------
            if total_pages > 1:
                cur = st.session_state.current_page
                page_block = 10
                block_start = ((cur - 1) // page_block) * page_block + 1
                block_end = min(block_start + page_block - 1, total_pages)
                nav_cols = st.columns(page_block + 2)  # ◀️ + 번호 + ▶️

                # ◀️ 이전
                if cur > 1 and nav_cols[0].button("◀️", key="page_prev"):
                    st.session_state.current_page = max(1, block_start - 1)
                    st.rerun()

                # 페이지 번호들
                for i, page_num in enumerate(range(block_start, block_end + 1)):
                    if nav_cols[i + 1].button(str(page_num), key=f"page_{page_num}"):
                        st.session_state.current_page = page_num
                        st.rerun()

                # ▶️ 다음
                if block_end < total_pages and nav_cols[page_block + 1].button("▶️", key="page_next"):
                    st.session_state.current_page = block_end + 1
                    st.rerun()

                                    
            st.divider()
        



    # ────────── 공지 목록 or 상세보기 ──────────
    # if st.session_state.page_mode == "list":
    #     st.subheader("📋 공지 목록")

        # if 'search_query' not in st.session_state:
        #     st.session_state.search_query = ""
        # if 'tag_filter' not in st.session_state:
        #     st.session_state.tag_filter = "전체"
        # if 'is_pinned_filter' not in st.session_state:
        #     st.session_state.is_pinned_filter = "전체"
        # if 'sort_option' not in st.session_state:
        #     st.session_state.sort_option = "최신순"

        # notices = get_notices()
        # if not notices:
        #     st.info("등록된 공지사항이 없습니다.")
        # else:
        #     for nid, title, content, created_at, is_pinned, tag, file_path in notices:
        #         # 오늘 날짜 확인
        #         is_today = (created_at[:10] == date.today().strftime("%Y-%m-%d"))
        #         # NEW, 중요, 태그, 첨부 뱃지
        #         badges = []
        #         if is_pinned:
        #             badges.append('<span style="background:#ffe066;color:#222;padding:2px 8px;border-radius:8px;font-size:0.88em;font-weight:bold;margin-right:5px;">📌 중요</span>')
        #         if tag:
        #             badges.append(f'<span style="background:#e9ecef;color:#333;padding:2px 8px;border-radius:8px;font-size:0.88em;margin-right:5px;">{tag}</span>')
        #         if is_today:
        #             badges.append('<span style="background:#e64980;color:#fff;padding:2px 8px;border-radius:8px;font-size:0.88em;font-weight:bold;">NEW</span>')
        #         if file_path:
        #             badges.append('<span title="첨부파일 있음" style="margin-left:7px;font-size:1.05em;">📎</span>')
        #         badge_str = " ".join(badges)
        #         # 본문 2줄 요약
        #         content_preview = content.replace("\n", " ")[:65] + ("..." if len(content) > 65 else "")
        #         # 카드형 목록
        #         with st.container():
        #             cols = st.columns([7, 2, 2, 1])

        #             with cols[0]:
        #                 # 버튼(하이퍼링크처럼) → 클릭 시 상세 진입
        #                 if st.button(title, key=f"title_{nid}"):
        #                     st.session_state.selected_notice = nid
        #                     st.session_state.page_mode = "detail"
        #                 # badge만 별도 마크다운 출력
        #                 st.markdown(f"{badge_str}", unsafe_allow_html=True)
        #                 st.markdown(
        #                     f"<div style='font-size:0.98em;color:#444;margin-top:0.3em;'>{content_preview}</div>",
        #                     unsafe_allow_html=True
        #                 )

                ########>>>
                    # with cols[0]:
                    #     btn_title = st.button(
                    #         label=title,
                    #         key=f"title_{nid}",
                    #         help="상세보기로 이동"
                    #     )
                    #     if btn_title:
                    #         st.session_state.selected_notice = nid
                    #         st.session_state.page_mode = "detail"
                    #     # 하이퍼링크 스타일 (CSS)
                    #     st.markdown(f"""
                    #         <style>
                    #         .stButton > button[kind="secondary"] {{
                    #             color: #2471e5 !important;
                    #             background: none !important;
                    #             border: none !important;
                    #             text-decoration: underline;
                    #             font-size:1.10em !important;
                    #             font-weight:600;
                    #         }}
                    #         </style>
                    #     """, unsafe_allow_html=True)
                    #     st.markdown(
                    #         f"""
                    #         <span style='font-size:1.12em;font-weight:600;'>{badge_str}</span>
                    #         <div style='font-size:0.98em;color:#444;margin-top:0.3em;'>{content_preview}</div>
                    #         """,
                    #         unsafe_allow_html=True
                    #     ) 

                        # st.markdown(f"""
                        #     <div style="font-size:1.12em;font-weight:600;cursor:pointer;" 
                        #         onclick="window.location.href='#{nid}'">
                        #         <a href="#" onclick="return false;">{title}</a> {badge_str}
                        #     </div>
                        #     <div style="font-size:0.98em;color:#444;margin-top:0.3em;">{content_preview}</div>
                        # """, unsafe_allow_html=True)
                        #############
                #     with cols[1]:
                #         st.markdown(f"<span style='color:#888;font-size:0.98em;'>{created_at[:10]}</span>", unsafe_allow_html=True)
                #     with cols[2]:
                #         if st.button("상세보기", key=f"detail_{nid}"):
                #             st.session_state.selected_notice = nid
                #             st.session_state.page_mode = "detail"
                #     with cols[3]:
                #         pass  # (추후) 삭제/수정 등 기능 자리
                # st.divider()



        # ────────── 상세보기 ──────────
    elif st.session_state.page_mode == "detail":
        notice = get_notice_by_id(st.session_state.selected_notice)
        if notice:
            nid, title, content, created_at, is_pinned, tag, file_path, writer, updated_at = notice
            st.markdown(f"<h2 style='margin-bottom:0.3em;'>{title}</h2>", unsafe_allow_html=True)
            badges = []
            if is_pinned:
                badges.append('<span style="background:#ffe066;color:#222;padding:3px 12px;border-radius:8px;font-size:1em;font-weight:bold;margin-right:7px;">📌 중요</span>')
            if tag:
                badges.append(f'<span style="background:#e9ecef;color:#333;padding:3px 12px;border-radius:8px;font-size:1em;margin-right:7px;">{tag}</span>')
            is_today = (created_at[:10] == date.today().strftime("%Y-%m-%d"))
            if is_today:
                badges.append('<span style="background:#e64980;color:#fff;padding:3px 12px;border-radius:8px;font-size:1em;font-weight:bold;">NEW</span>')
            if file_path:
                badges.append('<span title="첨부파일 있음" style="margin-left:9px;font-size:1.12em;">📎</span>')
            st.markdown(" ".join(badges), unsafe_allow_html=True)
            st.markdown(f"<div style='color:#999;font-size:0.96em;margin-bottom:1em;'>작성일: {created_at[:16]} | 작성자: {writer or '-'}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1.07em;line-height:1.65;'>{content.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
            if file_path:
                file_name = os.path.basename(file_path)
                st.download_button(
                    label=f"{file_name}", #다운로드",   # ← 실제 파일명 표시!
                    #label=f"📎 {file_name} 다운로드",   # ← 실제 파일명 표시!
                    data=open(file_path, "rb").read(),
                    file_name=file_name,
                    mime=None  # 파일 종류에 따라 필요시 지정
                )
                # with open(file_path, "rb") as f:
                #     st.download_button("첨부파일 다운로드", f, file_name=file_name)
            #st.button("목록으로 돌아가기", on_click=lambda: st.session_state.update(page_mode="list", selected_notice=None))

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("수정하기", key=f"edit_{nid}"):
                    st.session_state.page_mode = "edit"
            with col2:
                if st.button("삭제하기", key=f"delete_{nid}"):
                    delete_notice(nid)
                    st.success("공지사항이 삭제되었습니다.")
                    st.session_state.page_mode = "list"
                    st.session_state.selected_notice = None
                    st.rerun()
            with col3:
                st.button("목록으로 돌아가기", key=f"back_{nid}", on_click=lambda: st.session_state.update(page_mode="list", selected_notice=None))


        else:
            st.warning("해당 공지사항을 찾을 수 없습니다.")
            st.button("목록으로 돌아가기", on_click=lambda: st.session_state.update(page_mode="list", selected_notice=None))
        


    elif st.session_state.page_mode == "edit":
        notice = get_notice_by_id(st.session_state.selected_notice)
        if notice:
            nid, title, content, created_at, is_pinned, tag, file_path, writer, updated_at = notice
            with st.form("edit_form"):
                new_title = st.text_input("제목", value=title)
                new_content = st.text_area("내용", value=content)
                new_is_pinned = st.checkbox("중요(상단 고정)", value=bool(is_pinned))
                new_tag = st.text_input("태그", value=tag or "")
                new_writer = st.text_input("작성자", value=writer or "관리자")
                submit_edit = st.form_submit_button("수정 완료")
                if submit_edit:
                    update_notice(nid, new_title, new_content, int(new_is_pinned), new_tag, file_path, new_writer)
                    st.success("수정 완료!")
                    st.session_state.page_mode = "detail"
                    st.rerun()
            st.button("수정 취소", on_click=lambda: st.session_state.update(page_mode="detail"))
            
    #create_table()

    # choice = st.radio(
    #     label="",
    #     options=["공지사항 목록", "공지사항 작성"],
    #     index=0,
    #     horizontal=True,
    # )

    # # 공지사항 목록
    # if choice == "공지사항 목록":
    #     st.subheader("📋 전체 공지사항")
    #     notices = get_notices()

    #     if not notices:
    #         st.info("등록된 공지사항이 없습니다.")
    #     else:
    #         for idx, (nid, title, content, created_at) in enumerate(notices):
    #             # 가장 최근 공지에 "NEW" 뱃지
    #             is_new = (idx == 0)
    #             new_badge = '<span class="new-badge">NEW</span>' if is_new else ''
    #             st.markdown(f"""
    #                 <div class="notice-card">
    #                     <div class="notice-header">📌 {title} {new_badge}</div>
    #                     <div class="notice-date">{created_at} | ID: {nid}</div>
    #                     <div class="notice-content">{content[:400]}{'...' if len(content)>400 else ''}</div>
    #                 </div>
    #             """, unsafe_allow_html=True)
    #     st.write(" ")  # 공백
    #     col1, col2 = st.columns([8, 1])
    #     with col2:
    #         if st.button("🔄 다시 갱신"):
    #             st.rerun()

    # 공지사항 작성
    # elif choice == "공지사항 작성":
    #     st.subheader("📝 새 공지사항 등록")
    #     with st.form(key="notice_form"):
    #         title = st.text_input("제목", placeholder="공지 제목을 입력하세요")
    #         content = st.text_area("내용", height=200, placeholder="공지 내용을 입력하세요")
    #         uploaded_file = st.file_uploader("첨부 파일 (선택)", type=["jpg", "jpeg", "png", "pdf"])
        
    #         submitted = st.form_submit_button("공지사항 등록")
    #         if submitted:
    #             if (not title) or (not content):
    #                 st.error("제목과 내용을 모두 입력해주세요.")
    #             else:
    #                 file_path = save_uploaded_file(uploaded_file) if uploaded_file else None
    #                 insert_notice(title, content, file_path)
    #                 st.success("공지사항이 등록되었습니다! 상단 '공지사항 목록'에서 확인하세요.")

if __name__ == "__main__":
    main()