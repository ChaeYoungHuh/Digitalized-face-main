import streamlit as st
import cv2
import numpy as np
import time
from back import FaceProcessor
import threading
import base64
from io import BytesIO
from PIL import Image
import platform
import os

# 성능 최적화를 위한 설정
CV_FRAME_WIDTH = 320  # 캡처 프레임 너비 (낮출수록 성능 향상)
CV_FRAME_HEIGHT = 240  # 캠처 프레임 높이 (낮출수록 성능 향상)
PROCESSING_INTERVAL = 4  # 매 N 프레임마다 처리 (높일수록 성능 향상)
GRID_COLS = 50  # 그리드 열 수
GRID_ROWS = 35  # 그리드 행 수
REFRESH_RATE = 120  # 화면 갱신 간격 (ms)
CAPTURE_DELAY = 30  # 캡처 간격 (ms)

# 카메라 상태 체크 함수
def check_camera_availability():
    """사용 가능한 카메라를 확인하고 시스템 정보를 반환합니다."""
    camera_info = []
    max_cameras = 5  # 최대 5개 카메라 인덱스 확인
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            status = "사용 가능" if ret else "열림 (프레임 읽기 실패)"
            camera_info.append(f"카메라 인덱스 {i}: {status}")
            cap.release()
        else:
            camera_info.append(f"카메라 인덱스 {i}: 사용 불가")
    
    # 시스템 정보 추가
    os_info = f"운영체제: {platform.system()} {platform.release()}"
    
    return camera_info, os_info

# 전역 변수 정의 (스레드 간 공유)
class SharedState:
    """스레드 간 공유되는 상태 정보를 저장하는 클래스"""
    def __init__(self):
        self.stop_thread = False
        self.text_grid = None
        self.processor = FaceProcessor()
        self.current_frame = None
        self.processed_frame = None
        self.face_detected = False
        self.webcam_error = None
        self.show_hands_coming = False  # 손이 60% 이상 차지할 때 표시
        self.show_darkside = False      # 눈이 60% 이상 가려졌을 때 표시
        self.hands_coverage = 0.0       # 손의 커버리지 비율
        self.eyes_covered = 0.0         # 눈이 가려진 비율

# 공유 상태 객체 생성
if 'shared_state' not in st.session_state:
    st.session_state.shared_state = SharedState()
    
# 웹캠 활성화 상태 초기화 - 자동으로 켜지도록 설정
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = True
if 'thread_active' not in st.session_state:
    st.session_state.thread_active = False
if 'show_debug' not in st.session_state:
    st.session_state.show_debug = False
# 성능 설정을 고정 값으로 설정
st.session_state.frame_delay = 50  # 50ms로 고정
st.session_state.refresh_rate = 100  # 100ms로 고정

# 스트림릿 페이지 설정
st.set_page_config(
    page_title="텍스트 얼굴",
    layout="wide",
    initial_sidebar_state="collapsed",  # 사이드바 숨김
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 이미지를 base64로 인코딩하는 함수
def get_image_base64(img):
    """OpenCV 이미지를 base64 문자열로 변환합니다."""
    # OpenCV 이미지를 PIL 이미지로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # PIL 이미지를 base64로 인코딩
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG", quality=70)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

# 눈 영역을 표시하는 함수
def draw_eye_regions(frame, face_bbox, eye_landmarks, mouth_landmarks=None, nose_landmarks=None, hand_landmarks=None, hand_convex_hulls=None):
    # 원본 이미지 복사
    img = frame.copy()
    
    # 얼굴 경계 상자 그리기
    if face_bbox:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 눈 랜드마크 그리기
    if eye_landmarks:
        for i, eye_points in enumerate(eye_landmarks):
            # 눈 영역 확장 (추상적 표현)
            min_x = np.min(eye_points[:, 0])
            min_y = np.min(eye_points[:, 1])
            max_x = np.max(eye_points[:, 0])
            max_y = np.max(eye_points[:, 1])
            
            # 영역 확장
            width = max_x - min_x
            height = max_y - min_y
            min_x = int(max(0, min_x - width * 0.2))
            min_y = int(max(0, min_y - height * 0.2))
            max_x = int(min(frame.shape[1], max_x + width * 0.2))
            max_y = int(min(frame.shape[0], max_y + height * 0.2))
            
            # 확장된 눈 경계를 사각형으로 그리기
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255, 40), -1)  # 반투명 사각형

            # 눈 윤곽선 그리기 (원래 형태)
            hull = cv2.convexHull(eye_points)
            cv2.drawContours(img, [hull], 0, (0, 0, 255), 1)
            cv2.fillPoly(img, [hull], (0, 0, 255, 80))  # 반투명으로 채우기
            
            # 눈 중심 계산
            eye_center = np.mean(eye_points, axis=0).astype(int)
            cv2.circle(img, tuple(eye_center), 2, (0, 255, 255), -1)
    
    # 입 랜드마크 그리기
    if mouth_landmarks is not None:
        # 원래 윤곽선
        hull = cv2.convexHull(mouth_landmarks)
        cv2.drawContours(img, [hull], 0, (255, 0, 255), 1)
        cv2.fillPoly(img, [hull], (255, 0, 255, 80))  # 반투명으로 채우기
        
        # 입 영역 확장 (추상적 표현)
        min_x = np.min(mouth_landmarks[:, 0])
        min_y = np.min(mouth_landmarks[:, 1])
        max_x = np.max(mouth_landmarks[:, 0])
        max_y = np.max(mouth_landmarks[:, 1])
        
        # 영역 확장
        width = max_x - min_x
        height = max_y - min_y
        min_x = int(max(0, min_x - width * 0.3))
        min_y = int(max(0, min_y - height * 0.3))
        max_x = int(min(frame.shape[1], max_x + width * 0.3))
        max_y = int(min(frame.shape[0], max_y + height * 0.3))
        
        # 확장된 입 경계를 사각형으로 그리기
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 255), 2)
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (255, 0, 255, 40), -1)  # 반투명 사각형
    
    # 코 랜드마크 그리기
    if nose_landmarks is not None:
        # 원래 윤곽선
        hull = cv2.convexHull(nose_landmarks)
        cv2.drawContours(img, [hull], 0, (0, 255, 0), 1)
        cv2.fillPoly(img, [hull], (0, 255, 0, 80))  # 반투명으로 채우기
        
        # 코 영역 확장 (추상적 표현)
        min_x = np.min(nose_landmarks[:, 0])
        min_y = np.min(nose_landmarks[:, 1])
        max_x = np.max(nose_landmarks[:, 0])
        max_y = np.max(nose_landmarks[:, 1])
        
        # 영역 확장
        width = max_x - min_x
        height = max_y - min_y
        min_x = int(max(0, min_x - width * 0.15))
        min_y = int(max(0, min_y - height * 0.15))
        max_x = int(min(frame.shape[1], max_x + width * 0.15))
        max_y = int(min(frame.shape[0], max_y + height * 0.15))
        
        # 확장된 코 경계를 사각형으로 그리기
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0, 40), -1)  # 반투명 사각형
    
    # 손 랜드마크 그리기
    if hand_landmarks:
        for hand_points in hand_landmarks:
            # 손 랜드마크 포인트 그리기
            for point in hand_points:
                cv2.circle(img, tuple(point), 2, (255, 0, 0), -1)
    
    # 손 윤곽선 그리기
    if hand_convex_hulls:
        for hull_points in hand_convex_hulls:
            cv2.drawContours(img, [hull_points], 0, (255, 0, 0), 2)
            cv2.fillPoly(img, [hull_points], (255, 0, 0, 80))  # 반투명으로 채우기
    
    return img

# 웹캠 처리 함수
def process_webcam(shared_state, frame_delay_ms=CAPTURE_DELAY):
    """웹캠 스트림을 처리하는 메인 함수"""
    # 여러 카메라 인덱스 시도
    camera_indexes = [0, 1, 2]  # 0: 기본 웹캠, 1,2: 다른 카메라 장치 시도
    cap = None
    
    for idx in camera_indexes:
        try:
            print(f"카메라 인덱스 {idx} 연결 시도 중...")
            cap = cv2.VideoCapture(idx)
            
            # 웹캠 해상도 설정 - 해상도 낮춤 (성능 최적화)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CV_FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CV_FRAME_HEIGHT)
            
            # 웹캠이 제대로 열렸는지 확인
            if cap.isOpened():
                # 테스트 프레임 읽기
                ret, test_frame = cap.read()
                if ret:
                    print(f"카메라 인덱스 {idx}에 성공적으로 연결되었습니다.")
                    break
                else:
                    print(f"카메라 인덱스 {idx}에 연결했지만 프레임을 읽을 수 없습니다.")
                    cap.release()
                    cap = None
            else:
                print(f"카메라 인덱스 {idx}를 열 수 없습니다.")
                cap = None
        except Exception as e:
            print(f"카메라 인덱스 {idx} 연결 중 오류 발생: {str(e)}")
            cap = None
    
    if cap is None or not cap.isOpened():
        error_msg = "사용 가능한 카메라를 찾을 수 없습니다!"
        shared_state.webcam_error = error_msg
        print(error_msg)
        print("확인사항:")
        print("1. 카메라 연결 상태 확인")
        print("2. 다른 앱에서 카메라를 사용 중인지 확인")
        print("3. 카메라 권한 설정 확인")
        return
    
    # 웹캠 오류 초기화
    shared_state.webcam_error = None
    
    print(f"웹캠 스레드 시작됨 (딜레이: {frame_delay_ms}ms)")
    
    try:
        frame_count = 0
        while not shared_state.stop_thread:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다!")
                break
            
            # 프레임 처리 성능 최적화
            frame = cv2.resize(frame, (CV_FRAME_WIDTH, CV_FRAME_HEIGHT))  # 해상도 낮춤
            
            # 원본 프레임 저장
            shared_state.current_frame = frame.copy()
            
            # PROCESSING_INTERVAL 프레임마다 처리 (성능 최적화)
            if frame_count % PROCESSING_INTERVAL == 0:
                # 프레임 처리 및 얼굴 감지
                face_bbox, eye_landmarks, mouth_landmarks, nose_landmarks, hand_landmarks, hand_convex_hulls = shared_state.processor.process_frame(frame)
                
                # 얼굴 감지 상태 업데이트
                shared_state.face_detected = face_bbox is not None
                
                # 처리된 이미지 생성 및 저장
                if face_bbox is not None:
                    processed_img = draw_eye_regions(frame, face_bbox, eye_landmarks, mouth_landmarks, nose_landmarks, hand_landmarks, hand_convex_hulls)
                    shared_state.processed_frame = processed_img
                    
                    # 그리드에 매핑
                    text_grid = shared_state.processor.map_face_to_grid(
                        face_bbox, eye_landmarks, mouth_landmarks, nose_landmarks, hand_landmarks, hand_convex_hulls, (GRID_ROWS, GRID_COLS)
                    )
                    
                    # 손과 눈 커버리지 계산
                    if text_grid is not None:
                        # 손 커버리지 계산 (1, -1, -2, -3 값의 비율)
                        hand_cells = np.sum((text_grid == '1') | (text_grid == '-1') | (text_grid == '-2') | (text_grid == '-3'))
                        total_cells = GRID_ROWS * GRID_COLS
                        hand_coverage = hand_cells / total_cells
                        shared_state.hands_coverage = hand_coverage
                        
                        # 눈 가려짐 비율 계산 (-1 값의 비율)
                        eye_cells = np.sum(text_grid == '0')  # 모든 눈 영역
                        covered_eye_cells = np.sum(text_grid == '-1')  # 가려진 눈 영역
                        
                        if eye_cells > 0:
                            eyes_covered_ratio = covered_eye_cells / eye_cells
                            shared_state.eyes_covered = eyes_covered_ratio
                        else:
                            shared_state.eyes_covered = 0.0
                        
                        # 알림 상태 업데이트
                        shared_state.show_hands_coming = hand_coverage >= 0.6  # 손이 60% 이상 차지할 때
                        shared_state.show_darkside = eyes_covered_ratio >= 0.6 if eye_cells > 0 else False  # 눈이 60% 이상 가려졌을 때
                    
                    # 그리드 업데이트
                    shared_state.text_grid = text_grid
            
            # 프레임 카운터 증가
            frame_count += 1
            
            # 프레임 딜레이
            time.sleep(frame_delay_ms / 1000.0)
    
    except Exception as e:
        print(f"웹캠 스레드 오류: {str(e)}")
    
    finally:
        # 자원 해제
        if cap is not None:
            cap.release()
        print("웹캠 스레드 종료됨")

# 웹캠 스레드 관리
if st.session_state.webcam_active and not st.session_state.thread_active:
    # 웹캠 스레드 시작
    thread = threading.Thread(
        target=process_webcam, 
        args=(st.session_state.shared_state, CAPTURE_DELAY)
    )
    thread.daemon = True  # 메인 스레드가 종료되면 같이 종료
    thread.start()
    st.session_state.thread_active = True
    st.session_state.thread = thread

# CSS 스타일
st.markdown(
    """
    <style>
    /* 전체 페이지 스타일 - 중앙 정렬 및 기본 레이아웃 */
    body {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
        margin: 0;
        padding: 0;
        background-color: #f5f5f5;
    }
    
    /* 텍스트 그리드 컨테이너 - 작품의 주요 표시 영역 */
    .text-grid-container {
        font-family: monospace;
        width: 100%;
        max-width: 1200px;
        overflow: hidden;
        padding: 0;
        margin: 5vh auto;
        line-height: 1.4;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-height: 80vh;
    }
    
    /* 텍스트 그리드 행 스타일 */
    .text-row {
        white-space: nowrap;
        display: block;
        margin: 0;
        padding: 0;
        line-height: 1.5em;
        height: 1.5em;
    }
    
    /* 각 텍스트 문자 스타일 */
    .text-item {
        display: inline-block;
        width: 1.4em;
        text-align: center;
        margin: 0;
        padding: 0;
        font-size: 3vh;
        font-weight: bold;
    }
    
    /* 얼굴 요소별 텍스트 스타일 */
    /* 눈 영역 (0) */
    .text-0 {
        color: blue;
    }
    
    /* 손 영역 (1) */
    .text-1 {
        color: red;
    }
    
    /* 손이 눈을 가리는 영역 (-1) */
    .text-minus1 {
        color: magenta;
        font-weight: bolder;
    }
    
    /* 입 영역 (O) */
    .text-o {
        color: purple;
    }
    
    /* 코 영역 (^) */
    .text-nose {
        color: green;
    }
    
    /* 손이 입을 가리는 영역 (-2) */
    .text-minus2 {
        color: orange;
        font-weight: bolder;
    }
    
    /* 손이 코를 가리는 영역 (-3) */
    .text-minus3 {
        color: teal;
        font-weight: bolder;
    }
    
    /* 눈을 감은 상태 (==) */
    .text-equal {
        color: purple;
    }
    
    /* 기본 텍스트 스타일 */
    .text-no {
        color: #444;
    }
    
    /* 공백 스타일 */
    .text-space {
        color: transparent;
    }
    
    /* Streamlit UI 요소 조정 */
    .main .block-container {
        padding: 0;
        max-width: 100%;
    }
    
    /* 불필요한 Streamlit 요소 숨기기 */
    header, footer {
        display: none !important;
    }
    
    /* 모바일 최적화 */
    @media (max-width: 768px) {
        .text-item {
            font-size: 2.5vh;
            width: 1.3em;
        }
        
        .text-row {
            line-height: 1.3em;
            height: 1.3em;
        }
    }
    
    /* 예술적 경고 메시지 스타일 */
    /* "hnds coming" 메시지 - 기술의 침입을 상징 */
    .hands-coming-message {
        position: absolute;
        color: red;
        font-size: 5vh;
        z-index: 1000;
        animation: blink 1s infinite;
    }
    
    /* "darkside" 메시지 - 디지털 감시의 어두운 면을 상징 */
    .darkside-message {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: black;
        font-size: 10vh;
        z-index: 1001;
        animation: pulse 2s infinite;
    }
    
    /* 경고 메시지 애니메이션 효과 */
    /* 깜빡이는 애니메이션 - 경고의 시급함을 표현 */
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        75% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    
    /* 맥박 애니메이션 - 생동감과 위협을 표현 */
    @keyframes pulse {
        0% { transform: translate(-50%, -50%) scale(1); }
        50% { transform: translate(-50%, -50%) scale(1.1); }
    }
    
    /* 경고 메시지 컨테이너 */
    .warning-container {
        position: relative;
        width: 100%;
        height: 100vh;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 디버깅 정보 표시
if st.session_state.show_debug and st.session_state.webcam_active:
    shared_state = st.session_state.shared_state
    
    # 웹캠 오류가 있는 경우 표시
    if shared_state.webcam_error:
        st.error(f"웹캠 오류: {shared_state.webcam_error}")
        
        # 카메라 상태 확인 버튼
        if st.button("카메라 상태 확인"):
            camera_info, os_info = check_camera_availability()
            st.code("\n".join(camera_info))
            st.code(os_info)
            
        # 문제 해결 가이드
        st.markdown("""
        ### 문제 해결 방법:
        1. **권한 확인**: MacOS에서 시스템 환경설정 > 보안 및 개인 정보 보호 > 카메라에서 브라우저 및 Python에 권한이 부여되었는지 확인하세요.
        2. **다른 카메라 앱 종료**: FaceTime, Photo Booth 등 카메라를 사용하는 다른 앱을 모두 종료하세요.
        3. **브라우저 새로고침**: 현재 브라우저 탭을 새로고침하세요.
        4. **컴퓨터 재시작**: 필요한 경우 컴퓨터를 재시작하세요.
        """)
    
    # 상태 정보 표시
    status_cols = st.columns(3)
    with status_cols[0]:
        st.write(f"얼굴 감지: {'감지됨 ✅' if shared_state.face_detected else '감지되지 않음 ❌'}")
    
    # 이미지 표시
    debug_cols = st.columns(2)
    
    # 원본 이미지 표시
    with debug_cols[0]:
        st.subheader("원본 이미지")
        if shared_state.current_frame is not None:
            img_str = get_image_base64(shared_state.current_frame)
            st.markdown(f'<img src="data:image/jpeg;base64,{img_str}" width="100%"/>', unsafe_allow_html=True)
        else:
            st.write("웹캠 이미지를 가져오는 중...")
    
    # 처리된 이미지 표시
    with debug_cols[1]:
        st.subheader("처리된 이미지 (얼굴/눈 감지)")
        if shared_state.processed_frame is not None:
            img_str = get_image_base64(shared_state.processed_frame)
            st.markdown(f'<img src="data:image/jpeg;base64,{img_str}" width="100%"/>', unsafe_allow_html=True)
        else:
            st.write("이미지 처리 중...")

# 텍스트 그리드 표시 컨테이너
grid_container = st.container()

# 텍스트 그리드 렌더링
def render_text_grid():
    """텍스트 그리드를 HTML로 렌더링합니다."""
    shared_state = st.session_state.shared_state
    
    # 경고 메시지 컨테이너 시작
    html = '<div class="warning-container">'
    
    # 손이 많이 차지할 때 "hnds coming" 메시지 표시
    if shared_state.show_hands_coming:
        # 랜덤 위치 생성
        random_left = np.random.randint(10, 70)
        random_top = np.random.randint(10, 70)
        html += f'<div class="hands-coming-message" style="left: {random_left}%; top: {random_top}%;">hnds coming</div>'
    
    # 눈이 많이 가려졌을 때 "darkside" 메시지 표시
    if shared_state.show_darkside:
        html += '<div class="darkside-message">darkside</div>'
    
    # 텍스트 그리드 부분
    if shared_state.text_grid is not None:
        grid_rows, grid_cols = shared_state.text_grid.shape
        
        html += '<div class="text-grid-container">'
        for r in range(grid_rows):
            html += '<div class="text-row">'
            for c in range(grid_cols):
                text = shared_state.text_grid[r, c]
                
                if text == '0':
                    html += f'<span class="text-item text-0">{text}</span>'
                elif text == '1':
                    html += f'<span class="text-item text-1">{text}</span>'
                elif text == '-1':
                    html += f'<span class="text-item text-minus1">*</span>'  # 별표로 표시
                elif text == '-2':
                    html += f'<span class="text-item text-minus2">X</span>'  # X로 표시
                elif text == '-3':
                    html += f'<span class="text-item text-minus3">!</span>'  # !로 표시
                elif text == 'O':
                    html += f'<span class="text-item text-o">{text}</span>'  # 입 표시
                elif text == '^':
                    html += f'<span class="text-item text-nose">{text}</span>'  # 코 표시
                elif text == '==':
                    html += f'<span class="text-item text-equal">{text}</span>'
                elif text == ' ':
                    html += f'<span class="text-item text-space">&nbsp;</span>'
                else:
                    html += f'<span class="text-item text-no">{text}</span>'
                    
            html += '</div>'
        html += '</div>'
    else:
        # 기본 그리드 생성 (모든 셀이 공백인 그리드)
        html += '<div class="text-grid-container">'
        for r in range(GRID_ROWS):
            html += '<div class="text-row">'
            for c in range(GRID_COLS):
                html += '<span class="text-item text-space">&nbsp;</span>'
            html += '</div>'
        html += '</div>'
    
    # 경고 메시지 컨테이너 종료
    html += '</div>'
    
    grid_container.markdown(html, unsafe_allow_html=True)

# 그리드 렌더링
render_text_grid()

# 자동 새로고침 (웹캠이 활성화된 경우)
if st.session_state.webcam_active:
    time.sleep(REFRESH_RATE / 1000.0)
    st.rerun()
else:
    # 웹캠이 비활성화되었고 스레드가 실행 중인 경우 정리
    if st.session_state.thread_active:
        st.session_state.shared_state.stop_thread = True
        if 'thread' in st.session_state:
            st.session_state.thread.join(timeout=1.0)
        st.session_state.thread_active = False 