import streamlit as st
import heapq
import math

# --- 1. Fungsi Heuristik (Manhattan Distance) ---
def heuristic(a, b):
    """
    Menghitung jarak Manhattan antara dua titik (x, y).
    Jarak Manhattan: |x1 - x2| + |y1 - y2|
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# --- 2. Algoritma A* Search ---
def a_star_search(graph, start, goal):
    """
    Implementasi A* Search di Grid 2D.
    graph: Kamus yang memetakan node ke tetangga dan biayanya.
    start: Koordinat node awal (tuple).
    goal: Koordinat node tujuan (tuple).
    """
    # Priority Queue: (f_cost, g_cost, node, path)
    # f_cost: biaya total (g + h)
    # g_cost: biaya aktual dari start
    # node: koordinat node saat ini
    # path: jalur yang telah ditempuh hingga node ini
    frontier = [(0, 0, start, [start])]
    
    # Biaya terendah dari start ke node, yang ditemukan sejauh ini
    cost_so_far = {start: 0}
    
    # Kumpulan node yang sudah dieksplorasi (untuk menghindari loop tak terbatas)
    explored = set()
    
    while frontier:
        # Ambil node dengan f_cost terendah
        f_cost, g_cost, current_node, current_path = heapq.heappop(frontier)
        
        if current_node in explored:
            continue
            
        explored.add(current_node)

        # Cek apakah tujuan tercapai
        if current_node == goal:
            return current_path, cost_so_far[goal], explored

        # Eksplorasi tetangga
        for next_node in graph.get(current_node, []):
            new_g_cost = cost_so_far[current_node] + graph[current_node][next_node]
            
            # Jika node belum pernah dikunjungi ATAU ditemukan jalur yang lebih murah
            if next_node not in cost_so_far or new_g_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_g_cost
                h_cost = heuristic(next_node, goal)
                new_f_cost = new_g_cost + h_cost
                
                new_path = current_path + [next_node]
                heapq.heappush(frontier, (new_f_cost, new_g_cost, next_node, new_path))
                
    return None, 0, explored # Tidak ditemukan jalur

# --- 3. Fungsi Pembuatan Grid Sederhana (Node dan Biaya) ---
def create_grid_graph(size_x, size_y, obstacle_nodes=None):
    """Membuat representasi graf dari grid sederhana (4 arah pergerakan)."""
    graph = {}
    if obstacle_nodes is None:
        obstacle_nodes = set()
        
    nodes = [(x, y) for x in range(size_x) for y in range(size_y)]
    
    for x, y in nodes:
        if (x, y) in obstacle_nodes:
            continue # Abaikan node obstacle
            
        neighbors = {}
        # Gerakan 4 arah: Atas, Bawah, Kiri, Kanan
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Cek Batasan dan Obstacle
            if 0 <= nx < size_x and 0 <= ny < size_y and (nx, ny) not in obstacle_nodes:
                neighbors[(nx, ny)] = 1 # Biaya 1 untuk setiap langkah
        
        if neighbors:
            graph[(x, y)] = neighbors
            
    return graph

# --- 4. Aplikasi Streamlit ---

st.title("⭐ Visualisasi A* Search (Grid 2D)")
st.markdown("Cari jalur terpendek dari **Mulai** ke **Tujuan** di Grid.")

# Pengaturan Grid
size_x = st.slider("Ukuran Lebar Grid (X):", 5, 15, 10)
size_y = st.slider("Ukuran Tinggi Grid (Y):", 5, 15, 10)

# Input Koordinat
st.sidebar.header("Koordinat")
start_x = st.sidebar.slider("Mulai X:", 0, size_x - 1, 0)
start_y = st.sidebar.slider("Mulai Y:", 0, size_y - 1, 0)
goal_x = st.sidebar.slider("Tujuan X:", 0, size_x - 1, size_x - 1)
goal_y = st.sidebar.slider("Tujuan Y:", 0, size_y - 1, size_y - 1)

start_node = (start_x, start_y)
goal_node = (goal_x, goal_y)

# Obstacle (Simulasi sederhana dengan input teks)
st.sidebar.header("Obstacle (Hambatan)")
obstacle_input = st.sidebar.text_area(
    "Masukkan koordinat obstacle (x, y) dipisahkan koma dan baris baru:", 
    "2,2\n3,2\n4,2\n4,3\n4,4"
)

# Parsing Obstacle
obstacle_nodes = set()
try:
    for line in obstacle_input.split('\n'):
        if line.strip():
            x, y = map(int, line.strip().split(','))
            if 0 <= x < size_x and 0 <= y < size_y:
                 obstacle_nodes.add((x, y))
except ValueError:
    st.error("Format obstacle tidak valid. Gunakan format 'x,y' per baris.")

# Validasi Input
if start_node == goal_node:
    st.warning("Mulai dan Tujuan tidak boleh sama.")
elif start_node in obstacle_nodes or goal_node in obstacle_nodes:
    st.error("Mulai atau Tujuan tidak boleh berada di area Obstacle.")
else:
    grid_graph = create_grid_graph(size_x, size_y, obstacle_nodes)

    if st.button("Jalankan A* Search"):
        path, total_cost, explored_nodes = a_star_search(grid_graph, start_node, goal_node)

        # --- Visualisasi Grid ---
        st.subheader("Visualisasi Grid dan Jalur")
        
        # Inisialisasi grid (0=kosong, 1=obstacle, 2=explored, 3=path, 4=start, 5=goal)
        grid = [[0 for _ in range(size_y)] for _ in range(size_x)]
        
        # Tandai Obstacle, Start, dan Goal
        for ox, oy in obstacle_nodes:
            grid[ox][oy] = 1 # Obstacle
            
        grid[start_node[0]][start_node[1]] = 4 # Start
        grid[goal_node[0]][goal_node[1]] = 5 # Goal
        
        # Tandai node yang dieksplorasi (jika bukan start/goal/obstacle)
        for ex, ey in explored_nodes:
            if grid[ex][ey] == 0:
                grid[ex][ey] = 2 # Explored
        
        # Tandai Jalur Ditemukan
        if path:
            for px, py in path:
                if grid[px][py] < 3: # Jangan menimpa Start/Goal
                    grid[px][py] = 3 # Path
            
            st.success(f"🎉 Jalur Ditemukan! Total Biaya (Langkah): **{total_cost}**")
            st.code(f"Jalur: {path}")

        else:
            st.error("Jalur tidak ditemukan!")
            st.code("Tidak ada jalur dari Mulai ke Tujuan.")

        # Menampilkan Grid sebagai DataFrame (visualisasi tabel)
        # Transpose grid untuk menampilkan X sebagai kolom dan Y sebagai baris, lebih intuitif.
        
        # Membuat label yang lebih deskriptif
        color_map = {
            0: "⬜️ (Kosong)", 
            1: "⬛️ (Obstacle)", 
            2: "🟦 (Dieksplorasi)", 
            3: "🟩 (Jalur A*)", 
            4: "🟠 (Mulai)", 
            5: "🔴 (Tujuan)"
        }
        
        # Mengubah nilai numerik menjadi emoji untuk visualisasi
        emoji_grid = [[color_map[grid[x][y]] for x in range(size_x)] for y in range(size_y)]
        
        # Membuat DataFrame dan melakukan transpose
        df = st.dataframe(
            emoji_grid,
            column_config={i: st.column_config.Column(label=f"X={i}", width="small") for i in range(size_x)},
            hide_index=True,
            use_container_width=False,
            height=size_y * 35 # Menyesuaikan tinggi
        )
        st.markdown(
            """
            *Arah sumbu Y adalah vertikal (baris), X adalah horizontal (kolom).*
            """
        )