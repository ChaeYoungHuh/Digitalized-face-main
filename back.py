# 텍스트 얼굴(Text Face): 디지털 실존주의 표현을 위한 처리 엔진
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull

class FaceProcessor:
    """
    얼굴과 손의 특징을 감지하고 처리하는 클래스
    
    이 프로세서는 단순한 얼굴 인식을 넘어, 인간의 물리적 존재를 디지털 공간으로
    변환하는 과정을 구현합니다. 얼굴, 눈, 입, 코와 같은 인간 정체성의 핵심 요소들을
    추상적인 텍스트 심볼로 재해석함으로써 디지털 자아에 대한 철학적 질문을 던집니다.
    
    MediaPipe를 사용하여 얼굴, 눈, 입, 코 그리고 손을 감지합니다.
    이렇게 감지된 특징들은 추상적인 텍스트 그리드로 변환되어 물리적 현실과
    디지털 표현 사이의 관계를 탐구합니다.
    """
    def __init__(self):
        # MediaPipe Face Detection 초기화 - 얼굴의 경계 상자 감지
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # MediaPipe Face Mesh 초기화 - 얼굴의 세부 랜드마크 감지 
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,         # 단일 얼굴만 처리 (단일 정체성)
            refine_landmarks=True,   # 정교한 랜드마크 추출
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # MediaPipe Hands 초기화 - 손의 위치와 제스처 감지
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,        # 최대 두 손 감지 (상호작용)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 눈 랜드마크 인덱스 (MediaPipe는 468개 랜드마크 사용)
        # 왼쪽 눈 - 인식의 창구이자 디지털 세계를 보는 관점
        self.LEFT_EYE_LANDMARKS = [
            # 왼쪽 눈 윤곽
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        ]
        
        # 오른쪽 눈 - 또 다른 인식의 창구, 현실 감각의 균형
        self.RIGHT_EYE_LANDMARKS = [
            # 오른쪽 눈 윤곽
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        ]
        
        # 입 랜드마크 인덱스 - 표현과 소통의 상징
        self.MOUTH_LANDMARKS = [
            # 입 외곽선
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146
        ]
        
        # 코 랜드마크 인덱스 - 현실 감각과 존재의 중심
        self.NOSE_LANDMARKS = [
            # 코 윤곽 
            1, 44, 49, 217, 197
        ]
        
        # 눈 영역 크기 계수 - 인식 범위의 확장
        self.EYE_WIDTH_FACTOR = 0.2
        self.EYE_HEIGHT_FACTOR = 0.2
        
        # 얼굴 특징 영역 확장 계수 - 디지털 표현의 추상화 정도
        self.MOUTH_EXPANSION = 1.5  # 입 영역 확장 계수
        self.NOSE_EXPANSION = 1.3   # 코 영역 확장 계수
        self.EYE_EXPANSION = 1.4    # 눈 영역 확장 계수
        
        # 손 관련 계수 - 상호작용과 개입의 강도
        self.HAND_SIZE_FACTOR = 0.2
        self.HAND_OUTLINE_THICKNESS = 1

    def process_frame(self, frame):
        """
        이미지 프레임에서 얼굴과 손 특징을 처리합니다.
        
        이 함수는 실시간으로 물리적 현실(카메라 이미지)을 디지털 추상화로 변환하는
        핵심 과정을 구현합니다. 얼굴의 각 부분과 손의 위치를 감지하여 디지털 자아의
        특징을 추출합니다.
        
        Args:
            frame: OpenCV 이미지 프레임 (물리적 현실의 디지털 표현)
            
        Returns:
            face_bbox: 얼굴 경계 상자 좌표 (x1, y1, x2, y2)
            eye_landmarks: 눈 랜드마크 좌표 리스트
            mouth_landmarks: 입 랜드마크 좌표 
            nose_landmarks: 코 랜드마크 좌표
            hand_landmarks: 손 랜드마크 좌표 리스트
            hand_convex_hulls: 손 외곽선 좌표 리스트
        """
        # 프레임 복사 및 RGB 변환 (MediaPipe는 RGB 형식 사용)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        
        # 결과 초기화
        face_bbox = None
        eye_landmarks = []
        mouth_landmarks = None
        nose_landmarks = None
        hand_landmarks = []
        hand_convex_hulls = []  # 손 외곽선 저장
        
        # 얼굴 검출 - 인간 정체성의 중심
        face_detection_results = self.face_detection.process(rgb_frame)
        face_mesh_results = self.face_mesh.process(rgb_frame)
        
        # 손 검출 - 물리적 영향과 개입의 수단
        hand_results = self.hands.process(rgb_frame)
        
        # 얼굴 검출 처리 - 정체성의 경계 설정
        if face_detection_results.detections:
            for detection in face_detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * frame_width)
                y = int(bbox.ymin * frame_height)
                w = int(bbox.width * frame_width)
                h = int(bbox.height * frame_height)
                
                # 얼굴 경계 상자 (좌상단, 우하단)
                face_bbox = (max(0, x), max(0, y), min(frame_width, x + w), min(frame_height, y + h))
                
        # Face mesh 결과 처리 - 정체성의 세부 특징 추출
        if face_mesh_results.multi_face_landmarks:
            face_landmarks = face_mesh_results.multi_face_landmarks[0]
            landmarks = np.array([(int(lm.x * frame_width), int(lm.y * frame_height)) for lm in face_landmarks.landmark])
            
            # 눈 랜드마크 추출 - 인식과 감각의 창구
            left_eye_points = np.array([landmarks[i] for i in self.LEFT_EYE_LANDMARKS])
            right_eye_points = np.array([landmarks[i] for i in self.RIGHT_EYE_LANDMARKS])
            
            # 입 랜드마크 추출 - 표현과 소통의 수단
            mouth_points = np.array([landmarks[i] for i in self.MOUTH_LANDMARKS])
            
            # 코 랜드마크 추출 - 현실 감각의 중심
            nose_points = np.array([landmarks[i] for i in self.NOSE_LANDMARKS])
            
            # 눈 랜드마크 저장
            eye_landmarks = [left_eye_points, right_eye_points]
            mouth_landmarks = mouth_points
            nose_landmarks = nose_points
        
        # 손 랜드마크 처리 - 물리적 상호작용의 매개체
        if hand_results.multi_hand_landmarks:
            for hand_landmarks_obj in hand_results.multi_hand_landmarks:
                # 손 랜드마크 좌표 추출
                hand_points = []
                for lm in hand_landmarks_obj.landmark:
                    x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                    hand_points.append((x, y))
                
                # 손 랜드마크를 numpy 배열로 변환
                hand_points = np.array(hand_points)
                
                # 손의 Convex Hull 계산 - 물리적 영향의 범위 정의
                if len(hand_points) >= 3:  # ConvexHull 계산에는 최소 3개의 점이 필요
                    hull = ConvexHull(hand_points)
                    hull_points = hand_points[hull.vertices]
                    hand_convex_hulls.append(hull_points)
                
                hand_landmarks.append(hand_points)
        
        # 얼굴 경계 상자, 눈 랜드마크, 입/코 랜드마크, 손 랜드마크, 손 외곽선 반환
        return face_bbox, eye_landmarks, mouth_landmarks, nose_landmarks, hand_landmarks, hand_convex_hulls
    
    def map_face_to_grid(self, face_bbox, eye_landmarks, mouth_landmarks, nose_landmarks, hand_landmarks, hand_convex_hulls, grid_size):
        """
        얼굴과 손 특징을 텍스트 그리드에 매핑합니다.
        
        이 함수는 3차원 얼굴과 손의 특징을 2차원 텍스트 그리드로 추상화하는
        핵심 변환 과정을 구현합니다. 물리적 현실의 복잡한 특징들이 단순한 문자 기호로
        재해석되는 과정에서 디지털 자아의 본질과 한계를 드러냅니다.
        
        Args:
            face_bbox: 얼굴 경계 상자 (x1, y1, x2, y2) - 정체성의 경계
            eye_landmarks: 눈 랜드마크 좌표 - 인식의 창구
            mouth_landmarks: 입 랜드마크 좌표 - 표현의 수단
            nose_landmarks: 코 랜드마크 좌표 - 감각의 중심
            hand_landmarks: 손 랜드마크 좌표 - 개입의 수단
            hand_convex_hulls: 손 외곽선 좌표 - 영향의 범위
            grid_size: 그리드 크기 (rows, cols) - 디지털 표현의 해상도
            
        Returns:
            text_grid: 얼굴과 손의 특징을 나타내는 추상적 텍스트 그리드
        """
        rows, cols = grid_size
        grid = np.full((rows, cols), '', dtype=object)  # 기본값을 공백으로 설정
        
        # 얼굴이 감지되지 않은 경우 빈 그리드 반환 - 부재의 표현
        if face_bbox is None:
            return grid
        
        # 얼굴 영역 비율 계산 - 디지털 공간에서의 비례 관계
        face_x1, face_y1, face_x2, face_y2 = face_bbox
        face_width = face_x2 - face_x1
        face_height = face_y2 - face_y1
        
        # 눈, 입, 코 위치를 저장할 마스크 생성 (특징 간 겹침 확인용)
        eye_mask = np.zeros((rows, cols), dtype=bool)
        mouth_mask = np.zeros((rows, cols), dtype=bool)
        nose_mask = np.zeros((rows, cols), dtype=bool)
        
        # 눈 랜드마크 처리 - 인식의 창구 매핑
        if eye_landmarks:
            for eye_points in eye_landmarks:
                # 눈 중심점 계산
                eye_center = np.mean(eye_points, axis=0).astype(int)
                
                # 눈 영역 확장 - 인식 범위의 확장
                min_x = np.min(eye_points[:, 0])
                min_y = np.min(eye_points[:, 1])
                max_x = np.max(eye_points[:, 0])
                max_y = np.max(eye_points[:, 1])
                
                # 영역 확장 - 추상화 과정
                width = max_x - min_x
                height = max_y - min_y
                min_x = int(max(0, min_x - width * (self.EYE_EXPANSION - 1) / 2))
                min_y = int(max(0, min_y - height * (self.EYE_EXPANSION - 1) / 2))
                max_x = int(min(face_x2, max_x + width * (self.EYE_EXPANSION - 1) / 2))
                max_y = int(min(face_y2, max_y + height * (self.EYE_EXPANSION - 1) / 2))
                
                # 확장된 경계를 기반으로 새로운 점 생성
                expanded_points = np.array([
                    [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]
                ])
                
                # 눈 좌표를 ROI 좌표계로 변환 - 물리적->디지털 공간 변환
                roi_eye_points = np.array([[int((p[0] - face_x1) / face_width * cols), int((p[1] - face_y1) / face_height * rows)] for p in expanded_points])
                
                # 빈 마스크 생성 (그리드 크기)
                mask = np.zeros((rows, cols), dtype=np.uint8)
                
                # 눈 폴리곤을 그리드에 그리기 - 디지털 표현으로 매핑
                if len(roi_eye_points) >= 3:  # 폴리곤 그리기에는 최소 3개의 점이 필요
                    # 눈 영역을 채우기 위한 폴리곤 그리기
                    cv2.fillPoly(mask, [roi_eye_points], 255)
                    
                    # 마스크를 그리드에 적용 - '0'은 열린 눈, 디지털 세계를 바라보는 시선
                    for y in range(rows):
                        for x in range(cols):
                            if mask[y, x] > 0:
                                grid[y, x] = '0'  # 눈 영역은 '0'으로 표시
                                eye_mask[y, x] = True
        
        # 입 랜드마크 처리 - 표현의 수단 매핑
        if mouth_landmarks is not None:
            # 입 중심점 계산
            mouth_center = np.mean(mouth_landmarks, axis=0).astype(int)
            
            # 입 영역 확장 - 표현 범위의 확장
            min_x = np.min(mouth_landmarks[:, 0])
            min_y = np.min(mouth_landmarks[:, 1])
            max_x = np.max(mouth_landmarks[:, 0])
            max_y = np.max(mouth_landmarks[:, 1])
            
            # 영역 확장 - 추상화 과정
            width = max_x - min_x
            height = max_y - min_y
            min_x = int(max(0, min_x - width * (self.MOUTH_EXPANSION - 1) / 2))
            min_y = int(max(0, min_y - height * (self.MOUTH_EXPANSION - 1) / 2))
            max_x = int(min(face_x2, max_x + width * (self.MOUTH_EXPANSION - 1) / 2))
            max_y = int(min(face_y2, max_y + height * (self.MOUTH_EXPANSION - 1) / 2))
            
            # 확장된 경계를 기반으로 새로운 점 생성
            expanded_points = np.array([
                [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]
            ])
            
            # 입 좌표를 ROI 좌표계로 변환 - 물리적->디지털 공간 변환
            roi_mouth_points = np.array([[int((p[0] - face_x1) / face_width * cols), int((p[1] - face_y1) / face_height * rows)] for p in expanded_points])
            
            # 빈 마스크 생성 (그리드 크기)
            mask = np.zeros((rows, cols), dtype=np.uint8)
            
            # 입 폴리곤을 그리드에 그리기 - 디지털 표현으로 매핑
            if len(roi_mouth_points) >= 3:
                # 입 영역을 채우기 위한 폴리곤 그리기
                cv2.fillPoly(mask, [roi_mouth_points], 255)
                
                # 마스크를 그리드에 적용 - 'O'는 입, 디지털 공간에서의 표현
                for y in range(rows):
                    for x in range(cols):
                        if mask[y, x] > 0 and not eye_mask[y, x]:  # 눈과 겹치지 않는 영역만
                            grid[y, x] = 'O'  # 입 영역은 'O'로 표시
                            mouth_mask[y, x] = True
        
        # 코 랜드마크 처리 - 현실 감각의 중심 매핑
        if nose_landmarks is not None:
            # 코 중심점 계산
            nose_center = np.mean(nose_landmarks, axis=0).astype(int)
            
            # 코 영역 확장 - 감각 범위의 확장
            min_x = np.min(nose_landmarks[:, 0])
            min_y = np.min(nose_landmarks[:, 1])
            max_x = np.max(nose_landmarks[:, 0])
            max_y = np.max(nose_landmarks[:, 1])
            
            # 영역 확장 - 추상화 과정
            width = max_x - min_x
            height = max_y - min_y
            min_x = int(max(0, min_x - width * (self.NOSE_EXPANSION - 1) / 2))
            min_y = int(max(0, min_y - height * (self.NOSE_EXPANSION - 1) / 2))
            max_x = int(min(face_x2, max_x + width * (self.NOSE_EXPANSION - 1) / 2))
            max_y = int(min(face_y2, max_y + height * (self.NOSE_EXPANSION - 1) / 2))
            
            # 확장된 경계를 기반으로 새로운 점 생성
            expanded_points = np.array([
                [min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]
            ])
            
            # 코 좌표를 ROI 좌표계로 변환 - 물리적->디지털 공간 변환
            roi_nose_points = np.array([[int((p[0] - face_x1) / face_width * cols), int((p[1] - face_y1) / face_height * rows)] for p in expanded_points])
            
            # 빈 마스크 생성 (그리드 크기)
            mask = np.zeros((rows, cols), dtype=np.uint8)
            
            # 코 폴리곤을 그리드에 그리기 - 디지털 표현으로 매핑
            if len(roi_nose_points) >= 3:
                # 코 영역을 채우기 위한 폴리곤 그리기
                cv2.fillPoly(mask, [roi_nose_points], 255)
                
                # 마스크를 그리드에 적용 - '^'는 코, 디지털 공간에서의 현실 감각
                for y in range(rows):
                    for x in range(cols):
                        # 눈이나 입과 겹치지 않는 영역만
                        if mask[y, x] > 0 and not eye_mask[y, x] and not mouth_mask[y, x]:
                            grid[y, x] = '^'  # 코 영역은 '^'로 표시
                            nose_mask[y, x] = True
        
        # 손 윤곽선 처리 - 물리적 개입과 상호작용 매핑
        if hand_convex_hulls and face_bbox:
            for hull_points in hand_convex_hulls:
                # 손이 얼굴 ROI 내에 있는지 확인 - 물리적 개입의 범위 확인
                # 손의 중심점 계산
                hand_center_x = int(np.mean(hull_points[:, 0]))
                hand_center_y = int(np.mean(hull_points[:, 1]))
                
                # 얼굴 영역 내의 손만 처리 - 정체성에 대한 직접적 영향
                if (hand_center_x >= face_x1 and hand_center_x <= face_x2 and 
                    hand_center_y >= face_y1 and hand_center_y <= face_y2):
                    
                    # 손 좌표를 ROI 좌표계로 변환 - 물리적->디지털 공간 변환
                    roi_hull_points = np.array([[int((p[0] - face_x1) / face_width * cols), int((p[1] - face_y1) / face_height * rows)] for p in hull_points])
                    
                    # 빈 마스크 생성 (그리드 크기)
                    mask = np.zeros((rows, cols), dtype=np.uint8)
                    
                    # 손 폴리곤을 그리드에 그리기 - 디지털 표현으로 매핑
                    if len(roi_hull_points) >= 3:
                        # 손 영역을 채우기 위한 폴리곤 그리기
                        cv2.fillPoly(mask, [roi_hull_points], 255)
                        
                        # 마스크를 그리드에 적용 - 손의 영향에 따른 기호 변형
                        for y in range(rows):
                            for x in range(cols):
                                if mask[y, x] > 0:
                                    # 현재 위치가 눈 영역인지 확인 - 인식에 대한 개입
                                    if eye_mask[y, x]:
                                        grid[y, x] = '-1'  # 손이 눈을 가리면 '-1'로 표시 (인식 방해)
                                    elif mouth_mask[y, x]:
                                        grid[y, x] = '-2'  # 손이 입을 가리면 '-2'로 표시 (표현 방해)
                                    elif nose_mask[y, x]:
                                        grid[y, x] = '-3'  # 손이 코를 가리면 '-3'로 표시 (감각 방해)
                                    else:
                                        grid[y, x] = '1'  # 일반 손 영역은 '1'로 표시 (단순 개입)
        
        return grid
