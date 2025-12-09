import streamlit as st
import heapq
import math
import pandas as pd
import time

# --- 1. Fungsi Heuristik (Manhattan Distance) ---
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# --- 2. Algoritma A* Search dengan Log Langkah ---
def a_star_search_logged(graph, start, goal):
    
    frontier = [(0, 0, start, [start])] # (f_cost, g_cost, node, path)
    cost_so_far = {start: 0}
    explored = set()
    
    # Log utama untuk visualisasi langkah per langkah
    step_log = [] 
    
    while frontier:
        f_cost, g_cost, current_node, current_path = heapq.heappop(frontier)
        
        if current_node in explored:
            continue
            
        explored.add(current_node)
        
        # Log: Node saat ini sedang dieksplorasi
        step_log.append({
            'type': 'explore',
            'node': current_node,
            'g': g_cost,
            'h': heuristic(current_node, goal),
            'f': f_cost,
            'path_so_far': current_path
        })

        if current_node == goal:
            # Log: Tujuan tercapai
            step_log.append({
                'type': 'goal',
                'node': current_node,
                'g': g_cost,
                'h': 0,
                'f': f_cost,
                'path_so_far': current_path
            })
            return current_path, cost_so_far[goal], explored, step_log

        # Eksplorasi tetangga
        for next_node in graph.get(current_node, []):
            new_g_cost = cost_so_far[current_node] + graph[current_node][next_node]
            
            if next_node not in cost_so_far or new_g_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_g_cost
                h_cost = heuristic(next_node, goal)
                new_f_cost = new_g_cost + h_cost
                new_path = current_path + [next_node]
                
                heapq.heappush(frontier, (new_f_cost, new_g_cost, next_node, new_path))
                
    return None, 0, explored, step_log # Tidak ditemukan jalur

# --- 3. Fungsi Pembuatan Grid Sederhana (Graf) ---
def create_grid_graph(size_x, size_y, obstacle_nodes=None):
    # (Fungsi ini tetap sama dari kode sebelumnya)
    graph = {}
    if obstacle_nodes is None: obstacle_nodes = set()
    nodes = [(x, y) for x in range(size_x) for y in range(size_y)]
    for x, y in nodes:
        if (x, y) in obstacle_nodes: continue
        neighbors = {}
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size_x and 0 <= ny < size_y and (nx, ny) not in obstacle_nodes:
                neighbors[(nx, ny)] = 1
        if neighbors: graph[(x, y)] = neighbors
    return graph

# --- 4. Fungsi Visualisasi Bergerak (Simulasi) ---
def visualize_movement(size_x, size_y, start, goal, obstacle, log_data):
    
    # Placeholder untuk visualisasi yang akan diperbarui
    viz_container = st.empty()
    
    color_map = {
        0: "⬜️", 1: "⬛️", 2: "🟦", 3: "🟩", 4: "🟠", 5: "🔴", 6: "🚗"
    }

    # Fungsi untuk menggambar grid
    def draw_grid(current_pos=None, current_path=None):
        grid = [[0 for _ in range(size_y)] for _ in range(size_x)]
        
        # Tandai dasar
        for ox, oy in obstacle: grid[ox][oy] = 1
        grid[start[0]][start[1]] = 4 # Start
        grid[goal[0]][goal[1]] = 5 # Goal
        
        # Tandai jalur yang sudah terbukti (setelah Goal ditemukan)
        if current_path:
            for px, py in current_path:
                 if grid[px][py] < 3: grid[px][py] = 3

        # Tandai posisi mobil saat ini (hanya saat eksplorasi)
        if current_pos and grid[current_pos[0]][current_pos[1]] < 4:
            grid[current_pos[0]][current_pos[1]] = 6 # Mobil

        # Konversi ke emoji dan buat DataFrame
        emoji_grid = [[color_map[grid[x][y]] for x in range(size_x)] for y in range(size_y)]
        df = pd.DataFrame(emoji_grid).T
        
        # Gunakan 'with' untuk memperbarui container
        with viz_container.container():
            st.dataframe(
                df,
                column_config={i: st.column_config.Column(label=f"X={i}", width="small") for i in range(size_x)},
                hide_index=False,
                use_container_width=True,
                height=size_y * 40
            )

    # Inisialisasi tampilan awal
    draw_grid(current_pos=start)
    time.sleep(1)
    
    # Jalankan simulasi langkah demi langkah
    for i, log in enumerate(log_data):
        if log['type'] == 'explore':
            # Tampilkan pergerakan mobil ke node yang dieksplorasi
            draw_grid(current_pos=log['node'])
            st.info(f"Langkah {i+1}: Mengeksplorasi node {log['node']}. f(n)={log['f']}")
            time.sleep(0.3) # Jeda visual
        
        elif log['type'] == 'goal':
            # Setelah tujuan tercapai, tampilkan jalur final
            st.balloons()
            st.success(f"🎉 Langkah {i+1}: Tujuan {goal} ditemukan! Jalur terpendek ditemukan.")
            draw_grid(current_pos=goal, current_path=log['path_so_far'])
            
            # Tampilkan detail akhir
            df_status = pd.DataFrame(log_data)
            st.subheader("Detail Eksplorasi Node (g, h, f)")
            st.dataframe(df_status, use_container_width=True)

            return # Akhiri loop setelah tujuan tercapai
            
    st.error("Pencarian Selesai: Jalur tidak ditemukan.")


# --- 5. Konfigurasi Aplikasi Streamlit ---

st.set_page_config(layout="wide")
st.title("🚗 Simulasi A* Search Langkah Per Langkah")

col_input, col_view = st.columns([1, 3])

with col_input:
    st.header("Pengaturan Grid")
    size_x = st.slider("Ukuran Lebar Grid (X):", 5, 15, 10, key='sw')
    size_y = st.slider("Ukuran Tinggi Grid (Y):", 5, 15, 10, key='sh')

    st.subheader("Koordinat")
    start_x = st.slider("Mulai X:", 0, size_x - 1, 0, key='sx')
    start_y = st.slider("Mulai Y:", 0, size_y - 1, 0, key='sy')
    goal_x = st.slider("Tujuan X:", 0, size_x - 1, size_x - 1, key='gx')
    goal_y = st.slider("Tujuan Y:", 0, size_y - 1, size_y - 1, key='gy')
    
    start_node = (start_x, start_y)
    goal_node = (goal_x, goal_y)

    st.subheader("Obstacle (Hambatan)")
    obstacle_input = st.text_area(
        "Masukkan koordinat obstacle (x, y) dipisahkan koma dan baris baru:", 
        "2,2\n3,2\n4,2\n4,3\n4,4", height=100
    )

obstacle_nodes = set()
try:
    for line in obstacle_input.split('\n'):
        if line.strip():
            x, y = map(int, line.strip().split(','))
            if 0 <= x < size_x and 0 <= y < size_y:
                 obstacle_nodes.add((x, y))
except ValueError:
    with col_input: st.error("Format obstacle tidak valid. Gunakan format 'x,y' per baris.")

if start_node == goal_node:
    with col_input: st.warning("Mulai dan Tujuan tidak boleh sama.")
elif start_node in obstacle_nodes or goal_node in obstacle_nodes:
    with col_input: st.error("Mulai atau Tujuan tidak boleh berada di area Obstacle.")
else:
    grid_graph = create_grid_graph(size_x, size_y, obstacle_nodes)

    with col_input:
        if st.button("Mulai Simulasi A*"):
            path, total_cost, explored_nodes, step_log = a_star_search_logged(grid_graph, start_node, goal_node)
            
            with col_view:
                st.subheader("Simulasi Pergerakan")
                visualize_movement(size_x, size_y, start_node, goal_node, obstacle_nodes, step_log)
