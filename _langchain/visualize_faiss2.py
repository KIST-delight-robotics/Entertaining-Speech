import matplotlib.pyplot as plt
from matplotlib import rc
import faiss
import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sqlite3
import os

def determine_optimal_clusters(vectors, max_clusters=10):
    """
    엘보우 방법과 실루엣 점수를 사용하여 최적의 클러스터 수를 결정합니다.
    
    Parameters:
    - vectors: 클러스터링할 데이터 벡터.
    - max_clusters: 테스트할 최대 클러스터 수.
    
    Returns:
    - optimal_clusters: 최적의 클러스터 수.
    """
    wss = []
    silhouette_scores = []
    cluster_range = range(2, max_clusters+1)
    
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(vectors)
        wss.append(kmeans.inertia_)
        score = silhouette_score(vectors, labels)
        silhouette_scores.append(score)
        print(f"클러스터 수: {k}, WSS: {kmeans.inertia_:.2f}, 실루엣 점수: {score:.4f}")
    
    # # 엘보우 그래프
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(cluster_range, wss, 'bx-')
    # plt.xlabel('클러스터 수')
    # plt.ylabel('WSS')
    # plt.title('엘보우 방법을 이용한 최적의 클러스터 수 결정')
    
    # # 실루엣 점수 그래프
    # plt.subplot(1, 2, 2)
    # plt.plot(cluster_range, silhouette_scores, 'bx-')
    # plt.xlabel('클러스터 수')
    # plt.ylabel('실루엣 점수')
    # plt.title('실루엣 점수를 이용한 최적의 클러스터 수 결정')
    
    # plt.tight_layout()
    # plt.show()
    
    # 최적의 클러스터 수 선택 (실루엣 점수가 가장 높은 값)
    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"최적의 클러스터 수는 {optimal_clusters}입니다.")
    return optimal_clusters

def visualize_faiss_vectors_umap(faiss_index, file_names=None, clusters=None):
    """
    FAISS 벡터를 UMAP으로 시각화하는 함수.
    
    Parameters:
    - faiss_index: FAISS 인덱스 객체.
    - file_names: 벡터와 연결된 파일 이름 리스트 (선택적).
    - clusters: 클러스터 수. None일 경우 최적의 클러스터 수를 자동으로 결정.
    """
    total_vectors = faiss_index.ntotal
    if total_vectors == 0:
        print("FAISS 인덱스에 저장된 벡터가 없습니다.")
        return
    
    # IndexIDMap일 경우 기본 인덱스 접근
    if isinstance(faiss_index, faiss.IndexIDMap):
        base_index = faiss_index.index
    else:
        base_index = faiss_index
    
    # 모든 벡터를 저장할 배열 초기화
    vectors = np.zeros((total_vectors, base_index.d), dtype='float32')
    
    # 벡터 재구성
    print("FAISS 인덱스에서 벡터 재구성 중...")
    base_index.reconstruct_n(0, total_vectors, vectors)
    
    # 클러스터 수 결정
    if clusters is None:
        print("최적의 클러스터 수를 결정하는 중...")
        clusters = determine_optimal_clusters(vectors, max_clusters=10)
    
    # UMAP으로 차원 축소
    print("UMAP으로 차원 축소 중...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.25, n_components=2, random_state=42)
    reduced_vectors = reducer.fit_transform(vectors)
    
    # KMeans 클러스터링
    print(f"KMeans로 클러스터링 중 (클러스터 수: {clusters})...")
    kmeans = KMeans(n_clusters=clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    
    # 한국어 폰트 설정
    rc('font', family='NanumGothic')  # Windows의 경우 "맑은 고딕"
    # macOS는 'AppleGothic', Ubuntu는 'NanumGothic'을 사용
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

    # 시각화
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], 
                          c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"FAISS Vectors Visualization (UMAP, Clusters={clusters})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    
    # 파일 이름 레이블 추가 (옵션)
    if file_names and len(file_names) == total_vectors:
        for i, file_name in enumerate(file_names):
            # 주요 포인트에만 레이블 추가
            if i % 50 == 0:  # 50번째 데이터마다 레이블 추가
                plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], file_name,
                         fontsize=6, alpha=0.7)
    
    plt.show()

if __name__ == "__main__":
    # FAISS 인덱스 로드
    faiss_index = faiss.read_index("faiss_index.bin")
    
    # IndexIDMap 여부 확인
    if isinstance(faiss_index, faiss.IndexIDMap):
        print("FAISS 인덱스를 IndexIDMap으로 로드했습니다.")
    else:
        print("FAISS 인덱스를 기본 인덱스로 로드했습니다.")
    
    # SQLite 데이터베이스에서 파일 이름 가져오기
    db_path = './mp3_database.db'
    if not os.path.exists(db_path):
        print(f"데이터베이스 파일이 존재하지 않습니다: {db_path}")
        file_names = None
    else:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT file_name FROM mp3_files ORDER BY id")
        file_names = [row[0] for row in cursor.fetchall()]
        conn.close()
    
    # 벡터 수와 파일 이름 수 일치 여부 확인
    if file_names and len(file_names) != faiss_index.ntotal:
        print("경고: 파일 이름의 수가 벡터의 수와 일치하지 않습니다.")
        file_names = None  # 파일 이름 레이블 비활성화
    
    # FAISS 벡터 시각화 (클러스터 수 자동 결정)
    visualize_faiss_vectors_umap(faiss_index, file_names=file_names, clusters=None)
